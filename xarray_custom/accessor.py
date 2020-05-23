"""Module for accessors of custom DataArray classes.

This module provides a function (``add_methods_to_accessor``)
to add user-defined methods to an accessor at decoration.

"""
__all__ = ["add_methods_to_accessor"]


# standard library
import re
from functools import wraps
from types import FunctionType
from typing import Any, Optional


# dependencies
from xarray import register_dataarray_accessor


# constants
SPECIAL_NAME = "^__.+__$"


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

    for name in dir(cls):
        obj = getattr(cls, name)

        if is_user_defined_method(obj):
            setattr(Accessor, name, convert(obj))

    register_dataarray_accessor(accessor)(Accessor)
    return cls


def is_user_defined_method(obj: Any) -> bool:
    """Return whether ``obj`` is a user-defined method."""
    if not isinstance(obj, FunctionType):
        return False

    if re.search(SPECIAL_NAME, obj.__name__):
        return False

    return True
