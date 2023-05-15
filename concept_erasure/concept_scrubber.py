from contextlib import contextmanager
from typing import Literal, Sequence

from torch import Tensor, nn
from transformers import PreTrainedModel

from .concept_eraser import ConceptEraser
from .utils import assert_type, get_transformer_layers


class ConceptScrubber(nn.Module):
    def __init__(
        self,
        model: PreTrainedModel,
        y_dim: int = 1,
        *,
        cov_type: Literal["eye", "diag", "full"] = "full",
        rank: int | None = None,
        shrinkage: float = 0.0,
    ):
        super().__init__()

        d_model = model.config.hidden_size
        num_layers = model.config.num_hidden_layers

        self.x_dim = d_model
        self.y_dim = y_dim

        self.erasers = nn.ModuleList(
            [
                ConceptEraser(
                    d_model,
                    y_dim,
                    device=model.device,
                    cov_type=cov_type,
                    rank=rank,
                    shrinkage=shrinkage,
                )
                for _ in range(num_layers)
            ]
        )

    def clear_x(self):
        """Clear the running statistics of X."""
        for eraser in self.erasers:
            assert isinstance(eraser, ConceptEraser)
            eraser.clear_x()

    @contextmanager
    def record(self, model: PreTrainedModel, label: Tensor | None = None):
        """Update erasers with the activations of the model, using the given label.

        This method adds a forward pre-hook to each layer of the model, which
        records its input activations. These are used to update the erasers.
        """
        handles = []
        layers = get_transformer_layers(model)

        for _eraser, layer in zip(self.erasers, layers):
            eraser = assert_type(ConceptEraser, _eraser)

            # Called before every layer forward pass
            def record_hook(_, args):
                eraser.update(args[0], label)
                return args

            handles.append(layer.register_forward_pre_hook(record_hook))

        try:
            yield self
        finally:
            # Make sure to remove the hooks even if an exception is raised
            for handle in handles:
                handle.remove()

    @contextmanager
    def scrub(self, model, layer_indices: Sequence[int] = ()):
        """Add hooks to the model which apply the erasers during a forward pass."""

        handles = []
        layers = get_transformer_layers(model)

        for _eraser, layer in zip(self.erasers, layers):
            eraser = assert_type(ConceptEraser, _eraser)

            # Called before every layer forward pass
            def apply_hook(_, args):
                x, *extras = args
                return (eraser(x), *extras)

            handles.append(layer.register_forward_pre_hook(apply_hook))

        try:
            yield self
        finally:
            # Make sure to remove the hooks even if an exception is raised
            for handle in handles:
                handle.remove()

    @contextmanager
    def random_scrub(self, model, layer_indices: Sequence[int] = ()):
        handles = []
        layers = get_transformer_layers(model)

        eraser = assert_type(ConceptEraser, self.erasers[0])
        d = eraser.mean_x.shape[0]
        u = eraser.mean_x.new_zeros(d, eraser.rank)
        u = nn.init.orthogonal_(u)

        for _eraser, layer in zip(self.erasers, layers):
            eraser = assert_type(ConceptEraser, _eraser)

            # Called before every layer forward pass
            def apply_hook(_, args):
                x, *extras = args
                mean = eraser.mean_x

                P = eraser.proj_for_subspace(u)
                x = (x - mean) @ P.T + mean
                return (eraser(x), *extras)

            handles.append(layer.register_forward_pre_hook(apply_hook))

        try:
            yield self
        finally:
            # Make sure to remove the hooks even if an exception is raised
            for handle in handles:
                handle.remove()
