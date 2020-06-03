__all__ = ["DataArrayClass"]


# standard library
from typing import Callable


# dependencies
from .special import __new__, empty, zeros, ones, full


# main classes
class DataArrayClassMeta(type):
    def __new__(cls, name, bases, attrs):
        attrs = {**attrs, **dict(__new__=__new__)}
        return super().__new__(cls, name, bases, attrs)

    @property
    def empty(cls) -> Callable:
        cls._empty = classmethod(empty)
        return cls._empty

    @property
    def zeros(cls) -> Callable:
        cls._zeros = classmethod(zeros)
        return cls._zeros

    @property
    def ones(cls) -> Callable:
        cls._ones = classmethod(ones)
        return cls._ones

    @property
    def full(cls) -> Callable:
        cls._full = classmethod(full)
        return cls._full

    @property
    def __doc__(cls) -> str:
        return "No description."

    def __repr__(cls) -> str:
        return cls.__name__


class DataArrayClass(metaclass=DataArrayClassMeta):
    """No description."""

    dims = NotImplemented
    dtype = None
