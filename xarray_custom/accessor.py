"""Module for DataArray accessor classes."""


# standard library
from functools import lru_cache
from itertools import chain
from inspect import getsource, signature
from re import sub
from textwrap import dedent
from types import FunctionType
from typing import Any, Callable, List
from uuid import uuid4


# dependencies
from xarray import DataArray, register_dataarray_accessor


# helper features
class CommonAccessorMeta(type):
    """Metaclass only for the CommonAccessorBase class."""

    __accessors: dict = {}

    def __new__(meta, name: str, bases: tuple, namespace: dict) -> type:
        _name = namespace.get("_name")
        _dataarrayclass = namespace.pop("_dataarrayclass", None)

        cls = super().__new__(meta, name, bases, namespace)

        if _name not in meta.__accessors:
            cls._dataarrayclasses = []
            meta.__accessors[_name] = cls
            register_dataarray_accessor(cls._name)(cls)

        cls = meta.__accessors[_name]

        if _dataarrayclass is not None:
            cls._dataarrayclasses.insert(0, _dataarrayclass)

        return cls

    def __repr__(cls) -> str:
        return f"Accessor({cls._name!r})"


class CommonAccessorBase(metaclass=CommonAccessorMeta):
    """Base for DataArrayClass common accessors."""

    _name: str = ""
    _dataarrayclass: type

    def __init__(self, dataarray: DataArray) -> None:
        """Initialize an instance with a DataArray to be accessed."""
        self._dataarray = dataarray

    def __getattr__(self, name: str) -> Any:
        """Get a method or an attribute of the DataArray class."""
        for dataarrayclass in self._dataarrayclasses:
            bound = dataarrayclass.bind(self._dataarray)

            if hasattr(bound, name):
                return getattr(bound, name)

        raise AttributeError(f"Any DataArray class has no attribute {name!r}")

    def __dir__(self) -> List[str]:
        """List names in the union namespace of DataArray classes."""
        dirs = map(dir, self._dataarrayclasses)
        return list(set(chain.from_iterable(dirs)))


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
