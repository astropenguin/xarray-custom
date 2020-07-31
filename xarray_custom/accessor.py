"""Module for DataArray accessor classes."""


# standard library
from functools import lru_cache
from inspect import getsource, signature
from re import sub
from textwrap import dedent
from types import FunctionType
from typing import Any, Callable, List, Type
from uuid import uuid4

# dependencies
from xarray import DataArray, register_dataarray_accessor


# helper features
class UniqueAccessorMeta(type):
    """Metaclass only for the UniqueAccessorBase class."""

    def __repr__(cls) -> str:
        return f"UniqueAccessor({cls._name!r})"


class UniqueAccessorBase(metaclass=UniqueAccessorMeta):
    """Base for DataArrayClass unique accessors."""

    _name: str = ""
    _dataarrayclass: type

    def __init_subclass__(cls) -> None:
        """Initialize a subclass with a bound DataArray class."""
        cls._name = "_accessor_" + uuid4().hex[:16]
        cls._dataarrayclass._accessor = cls
        register_dataarray_accessor(cls._name)(cls)

    def __init__(self, dataarray: DataArray) -> None:
        """Initialize an instance with a DataArray to be accessed."""
        self._dataarray = dataarray

    @lru_cache(None)
    def __bind_function(self, func: Callable) -> Callable:
        """Bind a function to an instance to use it as a method."""
        first_arg = list(signature(func).parameters)[0]

        pattern = rf"(?<!\w){first_arg}\."
        repl = rf"{first_arg}.{self._name}."
        source = dedent(getsource(func))

        exec(sub(pattern, repl, source), func.__globals__, locals())
        return locals()[func.__name__].__get__(self._dataarray)

    def __getattr__(self, name: str) -> Any:
        """Get a bound method or an attribute of the DataArray class."""
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
