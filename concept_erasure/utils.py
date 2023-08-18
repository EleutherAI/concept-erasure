from typing import Any, Type, TypeVar, cast

from torch import nn

T = TypeVar("T")


def assert_type(typ: Type[T], obj: Any) -> T:
    """Assert that an object is of a given type at runtime and return it."""
    if not isinstance(obj, typ):
        raise TypeError(f"Expected {typ.__name__}, got {type(obj).__name__}")

    return cast(typ, obj)


def is_norm_layer(module: nn.Module) -> bool:
    """Return `True` if the module is a normalization layer."""
    cls_name = type(module).__name__
    return cls_name.endswith("LayerNorm") or cls_name.endswith("RMSNorm")


def mangle_module_path(name: str) -> str:
    """Mangle a module path to make it a valid key in a `nn.ModuleDict`."""
    # This is a weird edge case we probably don't need to support
    assert "-" not in name, "Module path cannot contain `-`"
    return name.replace(".", "-")
