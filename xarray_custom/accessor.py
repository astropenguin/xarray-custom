__all__ = ["add_methods_to_accessor"]


# standard library
import re
from functools import wraps
from typing import Any


# dependencies
from xarray import register_dataarray_accessor


# constants
SPECIAL_METHOD = "^__.+__$"


# main functions
def add_methods_to_accessor(cls: type) -> type:
    """Create a DataArray accessor and add methods in a class to it.

    Args:
        cls: Custom DataArray class.

    Returns:
        cls: Same as ``cls`` in the arguments.

    """

    class Accessor:
        def __init__(self, _accessed):
            self._accessed = _accessed

    def convert(method):
        @wraps(method)
        def accessor_method(self, *args, **kwargs):
            return method(self._accessed, *args, **kwargs)

        return accessor_method

    for name, obj in cls.__dict__.items():
        if is_user_defined_method(obj):
            setattr(Accessor, name, convert(obj))

    register_dataarray_accessor(cls.accessor)(Accessor)
    return cls


def is_user_defined_method(obj: Any) -> bool:
    """Return whether ``obj`` is a user-defined method."""
    if not callable(obj):
        return False

    if re.search(SPECIAL_METHOD, obj.__name__):
        return False

    return True
