from contextlib import contextmanager
from typing import Literal, Sequence

from torch import Tensor, nn
from transformers import PreTrainedModel

from .concept_eraser import ConceptEraser
from .utils import get_transformer_layers


class ConceptScrubber:
    def __init__(
        self,
        model: PreTrainedModel,
        y_dim: int = 1,
        *,
        cov_type: Literal["eye", "diag", "full"] = "full",
        rank: int | None = None,
        shrinkage: float = 0.0,
    ):
        d_model = model.config.hidden_size
        layers = get_transformer_layers(model)

        self.x_dim = d_model
        self.y_dim = y_dim

        self.erasers = {
            layer: ConceptEraser(
                d_model,
                y_dim,
                device=model.device,
                cov_type=cov_type,
                rank=rank,
                shrinkage=shrinkage,
            )
            for layer in layers
        }

    def clear_x(self):
        """Clear the running statistics of X."""
        for eraser in self.erasers.values():
            eraser.clear_x()

    @contextmanager
    def record(self, label: Tensor | None = None):
        """Update erasers with the activations of the model, using the given label.

        This method adds a forward pre-hook to each layer of the model, which
        records its input activations. These are used to update the erasers.
        """

        # Called before every layer forward pass
        def record_hook(layer, args):
            x, *extras = args

            self.erasers[layer].update(x, label)
            return (x, *extras)

        handles = {
            layer: layer.register_forward_pre_hook(record_hook)
            for layer in self.erasers
        }

        try:
            yield self
        finally:
            # Make sure to remove the hooks even if an exception is raised
            for handle in handles.values():
                handle.remove()

    @contextmanager
    def scrub(self, layer_indices: Sequence[int] = ()):
        """Add hooks to the model which apply the erasers during a forward pass."""

        # Called before every layer forward pass
        def apply_hook(layer, args):
            x, *extras = args
            x = self.erasers[layer](x)

            return (x, *extras)

        if layer_indices:
            layer_list = list(self.erasers.keys())
            layers = [layer_list[i] for i in layer_indices]
        else:
            layers = self.erasers.keys()

        handles = {
            layer: layer.register_forward_pre_hook(apply_hook) for layer in layers
        }

        try:
            yield self
        finally:
            # Make sure to remove the hooks even if an exception is raised
            for handle in handles.values():
                handle.remove()

    @contextmanager
    def random_scrub(self, layer_indices: Sequence[int] = ()):
        eraser = next(iter(self.erasers.values()))

        d = eraser.mean_x.shape[0]
        u = eraser.mean_x.new_zeros(d, eraser.rank)
        u = nn.init.orthogonal_(u)

        # Called after every layer forward pass
        def apply_hook(layer, args):
            x, *extras = args
            mean = self.erasers[layer].mean_x

            P = self.erasers[layer].proj_for_subspace(u)
            x = (x - mean) @ P.T + mean

            return (x, *extras)

        if layer_indices:
            layer_list = list(self.erasers.keys())
            layers = [layer_list[i] for i in layer_indices]
        else:
            layers = self.erasers.keys()

        handles = {
            layer: layer.register_forward_pre_hook(apply_hook) for layer in layers
        }

        try:
            yield self
        finally:
            # Make sure to remove the hooks even if an exception is raised
            for handle in handles.values():
                handle.remove()
