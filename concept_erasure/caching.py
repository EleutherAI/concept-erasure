from functools import wraps
from typing import Callable


def cached_property(func: Callable) -> property:
    """Decorator that converts a method into a lazily-evaluated cached property"""
    # Create a secret attribute name for the cached property
    attr_name = "_cached_" + func.__name__

    @property
    @wraps(func)
    def _cached_property(self):
        # If the secret attribute doesn't exist, compute the property and set it
        if not hasattr(self, attr_name):
            setattr(self, attr_name, func(self))

        # Otherwise, return the cached property
        return getattr(self, attr_name)

    return _cached_property


def invalidates_cache(dependent_prop_name: str) -> Callable:
    """Invalidates a cached property when the decorated function is called"""
    attr_name = "_cached_" + dependent_prop_name

    # The actual decorator
    def _invalidates_cache(func: Callable) -> Callable:
        # The wrapper function
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            # Check if the secret attribute exists; if so delete it so that
            # the cached property is recomputed
            if hasattr(self, attr_name):
                delattr(self, attr_name)

            return func(self, *args, **kwargs)

        return wrapper

    return _invalidates_cache
