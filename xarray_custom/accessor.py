"""Module for accessors of custom DataArray classes.

Currently this module only provides ``register_accessor`` function.

- register_accessor: Register an accessor for a custom DataArray class.

"""
__all__ = ["register_accessor"]


# standard library
from functools import wraps
from itertools import chain
from types import FunctionType
from typing import Any, List, Sequence


# dependencies
from xarray import DataArray, register_dataarray_accessor


# main functions
def register_accessor(cls):
    """Register an accessor for a custom DataArray class.

    Methods and attributes in the class can be accessed
    via the accessor like ``dataarray.<accessor>.<method>``.

    Args:
        cls: Custom DataArray class.

    Returns:
        This function returns nothing.

    """
    if cls.accessor is None:
        return

    # if an accessor is already registed
    if hasattr(DataArray, cls.accessor):
        Accessor = getattr(DataArray, cls.accessor)

        if not issubclass(Accessor, AccessorBase):
            raise ValueError(f"Invalid accessor: {cls.accessor}")

        Accessor._dataarrayclasses.insert(0, cls)
        return

    # otherwise, create a new accessor
    class Accessor(AccessorBase):
        _dataarrayclasses = DataArrayClasses([cls])

    register_dataarray_accessor(cls.accessor)(Accessor)


# main classes
class DataArrayClasses(list):
    """List-like class for storing DataArray classes.

    This class (instance) is only used in ``AccessorBase``
    for registering multiple DataArray classes to an accessor.

    Args:
        dataarrayclasses: Sequence of DataArray classes.

    Returns:
        List-like object which stores DataArray classes.

    """

    def __init__(self, dataarrayclasses: Sequence[type]) -> None:
        super().__init__(dataarrayclasses)

    def __getattr__(self, name: str) -> Any:
        """Get an attribute of DataArray classes."""
        if name not in dir(self):
            raise AttributeError

        for dataarrayclass in self:
            if hasattr(dataarrayclass, name):
                return getattr(dataarrayclass, name)

    def __dir__(self) -> List[str]:
        """List names in a union of DataArray classes' namespaces."""
        return list(set(chain.from_iterable(map(dir, self))))


class AccessorBase:
    """Base class for DataArray accessors.

    This class has a hidden attribute ``_dataarrayclasses``
    where multiple DataArray classes can be registered.
    Methods and attributes in the classes can be then accessed
    via the accessor like ``dataarray.<accessor>.<method>``.

    Args:
        dataarray: DataArray to be accessed.

    Returns:
        Accessor for a custom DataArray class.

    """

    _dataarrayclasses: DataArrayClasses

    def __init__(self, dataarray: DataArray) -> None:
        self._dataarray = dataarray

    def __getattr__(self, name: str) -> Any:
        """Get a method or an attribute of DataArray classes."""
        obj = getattr(self._dataarrayclasses, name)

        if not isinstance(obj, FunctionType):
            return obj

        @wraps(obj)
        def method(self, *args, **kwargs):
            return obj(self._dataarray, *args, **kwargs)

        setattr(type(self), name, method)
        return getattr(self, name)

    def __dir__(self) -> List[str]:
        """List names in a union of DataArray classes' namespaces."""
        return dir(self._dataarrayclasses)
