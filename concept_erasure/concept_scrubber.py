from contextlib import contextmanager
from functools import partial
from typing import Callable, Literal

import torch
from torch import Tensor, nn
from transformers import PreTrainedModel

from .concept_eraser import ConceptEraser
from .utils import assert_type, is_norm_layer, mangle_module_path


class ConceptScrubber(nn.Module):
    @classmethod
    def from_model(
        cls,
        model: PreTrainedModel,
        z_dim: int = 1,
        affine: bool = True,
        module_suffix: str = "",
        proj_type: Literal["leace", "orth"] = "leace",
        pre_hook: bool = False,
    ):
        d_model = model.config.hidden_size

        scrubber = cls()
        scrubber.erasers.update(
            {
                mangle_module_path(name): ConceptEraser(
                    d_model,
                    z_dim,
                    affine=affine,
                    device=model.device,
                    proj_type=proj_type,
                )
                # Note that we are unwrapping the base model here
                for name, mod in model.base_model.named_modules()
                if is_norm_layer(mod) and name.endswith(module_suffix)
            }
        )
        return scrubber

    def __init__(self, pre_hook: bool = False):
        super().__init__()

        self.erasers = nn.ModuleDict()
        self.pre_hook = pre_hook

    @contextmanager
    def record(
        self,
        model: PreTrainedModel,
        label: Tensor | None = None,
    ):
        """Update erasers with the activations of the model, using the given label.

        This method adds a forward pre-hook to each layer of the model, which
        records its input activations. These are used to update the erasers.
        """

        def record_hook(eraser: ConceptEraser, x: Tensor):
            eraser.update(x, label)

        with self.apply_hook(model, record_hook):
            yield self

    @contextmanager
    def scrub(self, model, dry_run: bool = False):
        """Add hooks to the model which apply the erasers during a forward pass."""

        def scrub_hook(eraser: ConceptEraser, x: Tensor):
            return eraser(x).type_as(x) if not dry_run else x

        with self.apply_hook(model, scrub_hook):
            yield self

    @contextmanager
    def random_scrub(self, model):
        eraser = assert_type(ConceptEraser, next(iter(self.erasers.values())))
        d = eraser.mean_x.shape[0]

        u = eraser.mean_x.new_zeros(d, eraser.z_dim)
        u = nn.init.orthogonal_(u)
        P = torch.eye(d, device=u.device) - u @ u.T

        def scrub_hook(eraser: ConceptEraser, x: Tensor):
            mean = eraser.mean_x

            if eraser.affine:
                _x = (x.type_as(mean) - mean) @ P.T + mean
            else:
                _x = x.type_as(mean) @ P.T

            return _x.type_as(x)

        with self.apply_hook(model, scrub_hook):
            yield self

    @contextmanager
    def apply_hook(
        self,
        model: nn.Module,
        hook_fn: Callable[[ConceptEraser, Tensor], Tensor | None],
    ):
        """Apply a `hook_fn` to each submodule in `model` that we're scrubbing."""

        def post_wrapper(_, __, output, name: str) -> Tensor | None:
            key = mangle_module_path(name)
            eraser = assert_type(ConceptEraser, self.erasers[key])
            return hook_fn(eraser, output)

        def pre_wrapper(_, inputs, name: str) -> tuple[Tensor | None, ...]:
            x, *extras = inputs
            key = mangle_module_path(name)
            eraser = assert_type(ConceptEraser, self.erasers[key])
            return hook_fn(eraser, x), *extras

        # Unwrap the base model if necessary
        if isinstance(model, PreTrainedModel):
            model = model.base_model

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
