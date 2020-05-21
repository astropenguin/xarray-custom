__all__ = ["add_methods_to_accessor"]


# standard library
from functools import wraps
from typing import Optional


# dependencies
from xarray import register_dataarray_accessor


# main functions
def add_methods_to_accessor(cls: type, accessor: Optional[str] = None) -> type:
    """Create a DataArray accessor and add methods in a class to it.

    Args:
        cls: Custom DataArray class.
        accessor: Name of a DataArray accessor.

    Returns:
        cls: Same as ``cls`` in the arguments.

    """
    if accessor is None:
        return cls

    class Accessor:
        def __init__(self, _accessed):
            self._accessed = _accessed

    def convert(method):
        @wraps(method)
        def accessor_method(self, *args, **kwargs):
            return method(self._accessed, *args, **kwargs)

        return accessor_method

    for name, obj in cls.__dict__.items():
        if not callable(obj):
            continue

        if name.startswith("__"):
            continue

        setattr(Accessor, name, convert(obj))

    register_dataarray_accessor(accessor)(Accessor)
    return cls
