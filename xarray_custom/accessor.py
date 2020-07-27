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


# main classes
class DataArrayAccessorBase:
    """Base class for DataArray accessor classes."""

    def __init_subclass__(cls, dataarrayclass: Type):
        """Initialize a subclass with a bound DataArray class."""
        cls.__id = uuid4().hex[:16]
        cls.__name = "_accessor_" + cls.__id
        cls.__dataarrayclass = dataarrayclass

        register_dataarray_accessor(cls.__name)(cls)

    def __init__(self, dataarray: DataArray):
        """Initialize an instance with a DataArray to be accessed."""
        self.__dataarray = dataarray

    @lru_cache(None)
    def __bind_function(self, func: Callable) -> Callable:
        """Bind a function to an instance to use it as a method."""
        first_arg = list(signature(func).parameters)[0]

        pattern = rf"(?<!\w){first_arg}\."
        repl = rf"{first_arg}.{self.__name}."
        source = dedent(getsource(func))

        exec(sub(pattern, repl, source), func.__globals__, locals())
        return locals()[func.__name__].__get__(self.__dataarray)

    def __dir__(self) -> List[str]:
        """List names in the namespace of the DataArray class."""
        return dir(self.__dataarrayclass)

    def __getattr__(self, name: str) -> Any:
        """Get a bound method or an attribute of the DataArray class."""
        try:
            return getattr(self.__dataarray, name)
        except AttributeError:
            obj = getattr(self.__dataarrayclass, name)

        if isinstance(obj, FunctionType):
            return self.__bind_function(obj)

        if isinstance(obj, property):
            return self.__bind_function(obj.fget)

        return obj
