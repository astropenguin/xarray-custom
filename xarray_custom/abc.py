"""Module for abstract base class of custom DataArray.

This module provides an abstract base class ``DataArrayClass`` to ensure
that a class decorated by ``dataarrayclass`` is a subclass of it.

"""
__all__ = ["DataArrayClass"]


# standard library
from abc import ABC


# abstract base class
class DataArrayClass(ABC):
    pass
