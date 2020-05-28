"""Module for accessors of custom DataArray classes.

This module provides a function (``add_methods_to_accessor``)
to add user-defined methods to an accessor at decoration.

"""
__all__ = ["add_methods_to_accessor"]


# standard library
import re
from functools import wraps
from types import FunctionType
from typing import Any, Callable, Optional


# dependencies
from xarray import DataArray, register_dataarray_accessor


# constants
SPECIAL_NAME = "^__.+__$"


# main functions
def add_methods_to_accessor(
    cls: type, accessor: Optional[str] = None, override: bool = False,
) -> type:
    """Create a DataArray accessor and add methods in a class to it.

    Args:
        cls: Custom DataArray class.
        accessor: Name of a DataArray accessor.
        override: Whether overriding a DataArray accessor of the same name
            if it is already registered in DataArray. If False (default),
            this function tries to add methods to it if the namespace
            has no conflicts. Otherwise an AttributeError is raised.

    Returns:
        cls: Same as ``cls`` in the arguments.

    Raises:
        AttributeError: Raised if this function fails to add methods
            due to some conflicts between method names and the
            namespace of the existing DataArray accessor.

    """
    if accessor is None:
        return cls

    if not hasattr(DataArray, accessor) or override:
        Accessor = create_accessor(accessor)
    else:
        Accessor = getattr(DataArray, accessor)

    for name in dir(cls):
        obj = getattr(cls, name)

        if is_user_defined_method(obj):
            Accessor._add_method(name, obj)

    return cls


# helper functions
def create_accessor(accessor: str) -> type:
    """Create a new DataArray accessor with special methods."""

    class Accessor:
        def __init__(self, _accessed: DataArray) -> None:
            self._accessed = _accessed

        @classmethod
        def _add_method(cls, name: str, method: Callable) -> None:
            if hasattr(cls, name):
                raise AttributeError(f"Method {name!r} already exists.")

            @wraps(method)
            def accessor_method(self, *args, **kwargs):
                return method(self._accessed, *args, **kwargs)

            setattr(cls, name, accessor_method)

    register_dataarray_accessor(accessor)(Accessor)
    return Accessor


def is_user_defined_method(obj: Any) -> bool:
    """Return whether ``obj`` is a user-defined method."""
    if not isinstance(obj, FunctionType):
        return False

    if re.search(SPECIAL_NAME, obj.__name__):
        return False

    return True
