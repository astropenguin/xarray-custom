"""Module for DataArray accessor classes."""
__all__ = ["add_accessors"]


# standard library
from collections import defaultdict
from functools import lru_cache
from itertools import chain
from inspect import getsource, signature
from re import sub
from textwrap import dedent
from types import FunctionType
from typing import Any, Callable, List, Optional
from uuid import uuid4


# dependencies
from xarray import DataArray, register_dataarray_accessor


# main features
def add_accessors(cls: type, name: Optional[str] = None) -> type:
    """Add unique and common accessors to a DataArray class.

    Args:
        cls: DataArray class to which accessors are added.
        name: Name of a common accessor. If not specified,
            only an unique accessor is added to the class.

    Returns:
        The same DataArray class as the input.

    """

    class UniqueAccessor(UniqueAccessorBase):
        _dataarrayclass = cls

    class CommonAccessor(CommonAccessorBase):
        _dataarrayclass = cls
        _name = name

    return cls


# helper features
class CommonAccessorBase:
    """Base class for common accessors of DataArray classes."""

    _dataarrayclasses = defaultdict(list)
    _dataarrayclass: type
    _name: str

    def __init_subclass__(cls):
        """Initialize a subclass with a bound DataArray class."""
        if not cls._name:
            return

        if cls._name not in cls._dataarrayclasses:
            register_dataarray_accessor(cls._name)(cls)

        cls._dataarrayclasses[cls._name].insert(0, cls._dataarrayclass)

    def __init__(self, dataarray: DataArray) -> None:
        """Initialize an instance with a DataArray to be accessed."""
        self._dataarray = dataarray

    def __getattr__(self, name: str) -> Any:
        """Get a method or an attribute of the DataArray class."""
        for dataarrayclass in self._dataarrayclasses[self._name]:
            bound = dataarrayclass._accessor(self._dataarray)

            if hasattr(bound, name):
                return getattr(bound, name)

        raise AttributeError(f"Any DataArray class has no attribute {name!r}")

    def __dir__(self) -> List[str]:
        """List names in the union namespace of DataArray classes."""
        dirs = map(dir, self._dataarrayclasses[self._name])
        return list(set(chain.from_iterable(dirs)))


class UniqueAccessorBase:
    """Base class for unique accessors of DataArray classes."""

    _dataarrayclass: type
    _name: str

    def __init_subclass__(cls) -> None:
        """Initialize a subclass with a bound DataArray class."""
        cls._dataarrayclass._accessor = cls
        cls._name = "_accessor_" + uuid4().hex[:16]
        register_dataarray_accessor(cls._name)(cls)

    def __init__(self, dataarray: DataArray) -> None:
        """Initialize an instance with a DataArray to be accessed."""
        self._dataarray = dataarray

    @lru_cache(None)
    def __bind_function(self, func: Callable) -> Callable:
        """Convert a function to a method of an instance."""
        first_arg = list(signature(func).parameters)[0]

        pattern = rf"(?<!\w){first_arg}\."
        repl = rf"{first_arg}.{self._name}."
        source = dedent(getsource(func))

        exec(sub(pattern, repl, source), func.__globals__, locals())
        return locals()[func.__name__].__get__(self._dataarray)

    def __getattr__(self, name: str) -> Any:
        """Get a method or an attribute of the DataArray class."""
        try:
            return getattr(self._dataarray, name)
        except AttributeError:
            obj = getattr(self._dataarrayclass, name)

        if isinstance(obj, FunctionType):
            return self.__bind_function(obj)

        if isinstance(obj, property):
            return self.__bind_function(obj.fget)

        return obj

    def __dir__(self) -> List[str]:
        """List names in the namespace of the DataArray class."""
        return dir(self._dataarrayclass)
