from contextlib import contextmanager
from functools import partial
from typing import Callable

from torch import Tensor, nn

from .leace import LeaceEraser
from .utils import assert_type, is_norm_layer, mangle_module_path


class ConceptScrubber:
    """Wrapper for a dictionary mapping module paths to `LeaceEraser` objects."""

    def __init__(self, pre_hook: bool = False):
        super().__init__()

        self.erasers: dict[str, LeaceEraser] = {}
        self.pre_hook = pre_hook

    @contextmanager
    def scrub(self, model):
        """Add hooks to the model which apply the erasers during a forward pass."""

        def scrub_hook(key: str, x: Tensor):
            eraser = assert_type(LeaceEraser, self.erasers[key])
            return eraser(x).type_as(x)

        with self.apply_hook(model, scrub_hook):
            yield self

    @contextmanager
    def apply_hook(
        self,
        model: nn.Module,
        hook_fn: Callable[[str, Tensor], Tensor | None],
    ):
        """Apply a `hook_fn` to each submodule in `model` that we're scrubbing."""

        def post_wrapper(_, __, output, name: str) -> Tensor | None:
            key = mangle_module_path(name)
            return hook_fn(key, output)

        def pre_wrapper(_, inputs, name: str) -> tuple[Tensor | None, ...]:
            x, *extras = inputs
            key = mangle_module_path(name)
            return hook_fn(key, x), *extras

        # Unwrap the base model if necessary. This is needed to ensure we don't try to
        # scrub right before the unembedding layer
        from transformers import PreTrainedModel

        if isinstance(model, PreTrainedModel):
            model = assert_type(PreTrainedModel, model.base_model)

        handles = [
            (
                mod.register_forward_pre_hook(partial(pre_wrapper, name=name))
                if self.pre_hook
                else mod.register_forward_hook(partial(post_wrapper, name=name))
            )
            for name, mod in model.named_modules()
            if is_norm_layer(mod) and mangle_module_path(name) in self.erasers
        ]
        assert len(handles) == len(self.erasers), "Not all erasers can be applied"

        try:
            yield self
        finally:
            # Make sure to remove the hooks even if an exception is raised
            for handle in handles:
                handle.remove()
