from contextlib import contextmanager
from functools import partial
from typing import Literal, Sequence

import torch
from torch import Tensor, nn
from transformers import PreTrainedModel

from .concept_eraser import ConceptEraser
from .utils import assert_type, is_norm_layer


class ConceptScrubber(nn.Module):
    def __init__(
        self,
        model: PreTrainedModel,
        y_dim: int = 1,
        *,
        affine: bool = True,
        cov_type: Literal["eye", "diag", "full"] = "full",
        dtype: torch.dtype | None = None,
        max_rank: int | None = None,
    ):
        super().__init__()

        d_model = model.config.hidden_size
        num_layers = sum(is_norm_layer(mod) for mod in model.modules())

        self.x_dim = d_model
        self.y_dim = y_dim

        self.erasers = nn.ModuleList(
            [
                ConceptEraser(
                    d_model,
                    y_dim,
                    affine=affine,
                    device=model.device,
                    dtype=dtype,
                    cov_type=cov_type,
                    max_rank=max_rank,
                )
                for _ in range(num_layers)
            ]
        )

    @contextmanager
    def record(
        self,
        model: PreTrainedModel,
        label: Tensor | None = None,
        eraser_indices: Sequence[int] = (),
    ):
        """Update erasers with the activations of the model, using the given label.

        This method adds a forward pre-hook to each layer of the model, which
        records its input activations. These are used to update the erasers.
        """
        handles = []
        norm_layers = [mod for mod in model.modules() if is_norm_layer(mod)]

        def record_hook(_, __, output, layer_idx):
            x, *_ = output
            eraser = assert_type(ConceptEraser, self.erasers[layer_idx])
            eraser.update(x, label)

        for i, norm_layer in enumerate(norm_layers):
            # Skip layers which are not in the given indices
            if eraser_indices and i not in eraser_indices:
                continue

            hook_fn = partial(record_hook, layer_idx=i)
            handles.append(norm_layer.register_forward_hook(hook_fn))

        try:
            yield self
        finally:
            # Make sure to remove the hooks even if an exception is raised
            for handle in handles:
                handle.remove()

    @contextmanager
    def scrub(
        self,
        model,
        dry_run: bool = False,
        eraser_indices: Sequence[int] = (),
        return_hiddens: bool = False,
    ):
        """Add hooks to the model which apply the erasers during a forward pass."""

        handles = []
        hiddens = []
        norm_layers = [mod for mod in model.modules() if is_norm_layer(mod)]

        def apply_hook(_, __, x, layer_idx):
            eraser = assert_type(ConceptEraser, self.erasers[layer_idx])
            x_ = eraser(x).type_as(x) if not dry_run else x
            if return_hiddens:
                hiddens.append(x_)

            return x_

        for i, norm_layer in enumerate(norm_layers):
            # Skip layers which are not in the given indices
            if eraser_indices and i not in eraser_indices:
                continue

            hook_fn = partial(apply_hook, layer_idx=i)
            handles.append(norm_layer.register_forward_hook(hook_fn))

        try:
            yield hiddens
        finally:
            # Make sure to remove the hooks even if an exception is raised
            for handle in handles:
                handle.remove()

    @contextmanager
    def random_scrub(self, model, eraser_indices: Sequence[int] = ()):
        handles = []
        norm_layers = [mod for mod in model.modules() if is_norm_layer(mod)]

        eraser = assert_type(ConceptEraser, self.erasers[0])
        d = eraser.mean_x.shape[0]
        u = eraser.mean_x.new_zeros(d, eraser.max_rank)
        u = nn.init.orthogonal_(u)

        def apply_hook(_, __, x, layer_idx):
            eraser = assert_type(ConceptEraser, self.erasers[layer_idx])
            mean = eraser.mean_x

            P = eraser.proj_for_subspace(u)
            if eraser.affine:
                _x = (x.type_as(mean) - mean) @ P.T + mean
            else:
                _x = x.type_as(mean) @ P.T

            return _x.type_as(x)

        for i, norm_layer in enumerate(norm_layers):
            # Skip layers which are not in the given indices
            if eraser_indices and i not in eraser_indices:
                continue

            hook_fn = partial(apply_hook, layer_idx=i)
            handles.append(norm_layer.register_forward_hook(hook_fn))

        try:
            yield self
        finally:
            # Make sure to remove the hooks even if an exception is raised
            for handle in handles:
                handle.remove()
