from typing import Any, Type, TypeVar, cast

from torch import nn
from transformers import PreTrainedModel

T = TypeVar("T")


def assert_type(typ: Type[T], obj: Any) -> T:
    """Assert that an object is of a given type at runtime and return it."""
    if not isinstance(obj, typ):
        raise TypeError(f"Expected {typ.__name__}, got {type(obj).__name__}")

    return cast(typ, obj)


def get_transformer_layers(model: PreTrainedModel) -> nn.ModuleList:
    """Return the `nn.ModuleList` containing the transformer layers in a model."""
    assert not model.config.is_encoder_decoder, "Encoder-decoder models not supported"

    lists = [mod for mod in model.modules() if isinstance(mod, nn.ModuleList)]
    if not lists:
        raise ValueError("Could not find transformer layers")

    # Return the module list with the most parameters
    return max(lists, key=lambda mod: sum(p.numel() for p in mod.parameters()))


def is_norm_layer(module: nn.Module) -> bool:
    """Return `True` if the module is a normalization layer."""
    cls_name = type(module).__name__
    return cls_name.endswith("LayerNorm") or cls_name.endswith("RMSNorm")
