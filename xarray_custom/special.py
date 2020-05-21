__all__ = ["add_special_methods"]


# standard library
from typing import Any, Optional


# dependencies
import numpy as np
from xarray import DataArray
from .typing import Attrs, Dtype, Name, Shape


# constants
ORDER = "C"


# main functions
def add_special_methods(cls: type) -> type:
    """Add special methods to a custom DataArray class.

    Args:
        cls: Custom DataArray class.

    Returns:
        cls: Same as ``cls`` in the arguments.

    """
    cls.zeros = zeros
    cls.ones = ones
    cls.empty = empty
    cls.full = full

    return cls


# special class methods
@classmethod
def zeros(
    cls: type,
    shape: Shape,
    dtype: Optional[Dtype] = None,
    order: str = ORDER,
    name: Optional[Name] = None,
    attrs: Optional[Attrs] = None,
    **coords,
) -> DataArray:
    """Create a custom DataArray filled with zeros.

    Args:
        shape: Shape of a custom DataArray. The length of it must match
            that of ``dims`` defined by the class.
        dtype: Datatype of a custom DataArray. Default is 64-bit float.
        order: Order of data in memory. Either ``'C'`` (row-major; C-style)
            or ``'F'`` (column-major; Fortran-style) is accepted.
        name: Name of a custom DataArray.
        attrs: Attributes of a custom DataArray. Default is an empty dict.
        **coords: Coordinates of a custom DataArray defined by the class.

    Returns:
        dataarray: Custom DataArray filled with zeros.

    """
    return cls(np.zeros(shape, dtype, order), name, attrs, **coords)


@classmethod
def ones(
    cls: type,
    shape: Shape,
    dtype: Optional[Dtype] = None,
    order: str = ORDER,
    name: Optional[Name] = None,
    attrs: Optional[Attrs] = None,
    **coords,
) -> DataArray:
    """Create a custom DataArray filled with ones.

    Args:
        shape: Shape of a custom DataArray. The length of it must match
            that of ``dims`` defined by the class.
        dtype: Datatype of a custom DataArray. Default is 64-bit float.
        order: Order of data in memory. Either ``'C'`` (row-major; C-style)
            or ``'F'`` (column-major; Fortran-style) is accepted.
        name: Name of a custom DataArray.
        attrs: Attributes of a custom DataArray. Default is an empty dict.
        **coords: Coordinates of a custom DataArray defined by the class.

    Returns:
        dataarray: Custom DataArray filled with ones.

    """
    return cls(np.ones(shape, dtype, order), name, attrs, **coords)


@classmethod
def empty(
    cls: type,
    shape: Shape,
    dtype: Optional[Dtype] = None,
    order: str = ORDER,
    name: Optional[Name] = None,
    attrs: Optional[Attrs] = None,
    **coords,
) -> DataArray:
    """Create a custom DataArray filled with uninitialized values.

    Args:
        shape: Shape of a custom DataArray. The length of it must match
            that of ``dims`` defined by the class.
        dtype: Datatype of a custom DataArray. Default is 64-bit float.
        order: Order of data in memory. Either ``'C'`` (row-major; C-style)
            or ``'F'`` (column-major; Fortran-style) is accepted.
        name: Name of a custom DataArray.
        attrs: Attributes of a custom DataArray. Default is an empty dict.
        **coords: Coordinates of a custom DataArray defined by the class.

    Returns:
        dataarray: Custom DataArray filled with uninitialized values.

    """
    return cls(np.empty(shape, dtype, order), name, attrs, **coords)


@classmethod
def full(
    cls: type,
    shape: Shape,
    fill_value: Any,
    dtype: Optional[Dtype] = None,
    order: str = ORDER,
    name: Optional[Name] = None,
    attrs: Optional[Attrs] = None,
    **coords,
) -> DataArray:
    """Create a custom DataArray filled with ``fill_value``.

    Args:
        shape: Shape of a custom DataArray. The length of it must match
            that of ``dims`` defined by the class.
        fill_value: Scalar value to fill a custom DataArray.
        dtype: Datatype of a custom DataArray. Default is 64-bit float.
        order: Order of data in memory. Either ``'C'`` (row-major; C-style)
            or ``'F'`` (column-major; Fortran-style) is accepted.
        name: Name of a custom DataArray.
        attrs: Attributes of a custom DataArray. Default is an empty dict.
        **coords: Coordinates of a custom DataArray defined by the class.

    Returns:
        dataarray: Custom DataArray filled with ``fill_value``.

    """
    return cls(np.full(shape, fill_value, dtype, order), name, attrs, **coords)
