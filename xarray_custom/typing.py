"""Module for type hints of xarray DataArray.

This module provides type hints of xarray DataArray
which are intended to be used other modules of the package.

"""
__all__ = ["Attrs", "Dims", "Dtype", "Name", "Shape"]


# standard library
from typing import (
    Hashable,
    Mapping,
    Sequence,
    Tuple,
    Union,
)


# type hints
Dims = Union[Sequence[Hashable], Hashable]
Dtype = Union[type, str]
Shape = Union[Tuple[int], int]
Name = Hashable
Attrs = Mapping
