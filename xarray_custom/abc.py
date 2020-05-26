"""Module for abstract base class (ABC).

This module provides an abstract base class, ``DataArrayClass``,
to ensure that a class is a subclass of it at decoration.

"""
__all__ = ["DataArrayClass"]


# standard library
from abc import ABC


# abstract base class
class DataArrayClass(ABC):
    """Abstract base class of a DataArray class."""

    pass
