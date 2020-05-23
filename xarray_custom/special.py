__all__ = ["add_special_methods"]


# standard library
from typing import Any, Optional


# dependencies
import numpy as np
from xarray import DataArray
from .typing import Attrs, Dtype, Name, Shape


# constants
ORDER: str = "C"


# main functions
def add_special_methods(cls: type) -> type:
    """Add special methods to a custom DataArray class.

    Args:
        cls: Custom DataArray class.

    Returns:
        cls: Same as ``cls`` in the arguments.

    """
    cls.__new__ = __new__
    cls.zeros = zeros
    cls.ones = ones
    cls.empty = empty
    cls.full = full

    return cls


# special class methods
def __new__(
    cls: type,
    data: Any,
    name: Optional[Name] = None,
    attrs: Optional[Attrs] = None,
    **coords,
) -> DataArray:
    """Create a custom DataArray with ``dims={dims}`` and ``dtype={dtype}``.

    Args:
        data: Values of the DataArray. Its shape must be consistent with
            ``dims``. It is casted to ``dtype`` if it is defined as a
            class attribute (an error is raised if it cannot be casted).
        name: Name of the DataArray.
        attrs: Attributes of the DataArray. Default is an empty dict.
        **coords: Coordinates of the DataArray defined by the class.

    Returns:
        dataarray: Custom DataArray.

    """
    dataarray = DataArray(data, dims=cls.dims, name=name, attrs=attrs)

    if cls.dtype is not None:
        dataarray = dataarray.astype(cls.dtype)

    for name, ctype in cls.ctypes.items():
        shape = [dataarray.sizes[dim] for dim in ctype.dims]

        if name in coords:
            dataarray.coords[name] = ctype.full(shape, coords[name])
            continue

        if hasattr(cls, name):
            dataarray.coords[name] = ctype.full(shape, getattr(cls, name))
            continue

        raise ValueError(
            f"No default value for a coordinate {repr(name)}. "
            "The value must be given as a keyword argument."
        )

    return dataarray


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
        shape: Shape of the DataArray. The length of it must match
            that of ``dims`` defined by the class.
        dtype: Datatype of the DataArray. Default is 64-bit float.
            It is ignored if ``dtype`` is defined as a class attribute.
        order: Order of data in memory. Either ``'C'`` (row-major; C-style)
            or ``'F'`` (column-major; Fortran-style) is accepted.
        name: Name of the DataArray.
        attrs: Attributes of the DataArray. Default is an empty dict.
        **coords: Coordinates of the DataArray defined by the class.

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
        shape: Shape of the DataArray. The length of it must match
            that of ``dims`` defined by the class.
        dtype: Datatype of the DataArray. Default is 64-bit float.
            It is ignored if ``dtype`` is defined as a class attribute.
        order: Order of data in memory. Either ``'C'`` (row-major; C-style)
            or ``'F'`` (column-major; Fortran-style) is accepted.
        name: Name of the DataArray.
        attrs: Attributes of the DataArray. Default is an empty dict.
        **coords: Coordinates of the DataArray defined by the class.

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
        shape: Shape of the DataArray. The length of it must match
            that of ``dims`` defined by the class.
        dtype: Datatype of the DataArray. Default is 64-bit float.
            It is ignored if ``dtype`` is defined as a class attribute.
        order: Order of data in memory. Either ``'C'`` (row-major; C-style)
            or ``'F'`` (column-major; Fortran-style) is accepted.
        name: Name of the DataArray.
        attrs: Attributes of the DataArray. Default is an empty dict.
        **coords: Coordinates of the DataArray defined by the class.

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
        shape: Shape of the DataArray. The length of it must match
            that of ``dims`` defined by the class.
        fill_value: Scalar value to fill a custom DataArray.
        dtype: Datatype of the DataArray. Default follows ``fill_value``.
            It is ignored if ``dtype`` is defined as a class attribute.
        order: Order of data in memory. Either ``'C'`` (row-major; C-style)
            or ``'F'`` (column-major; Fortran-style) is accepted.
        name: Name of the DataArray.
        attrs: Attributes of the DataArray. Default is an empty dict.
        **coords: Coordinates of the DataArray defined by the class.

    Returns:
        dataarray: Custom DataArray filled with ``fill_value``.

    """
    return cls(np.full(shape, fill_value, dtype, order), name, attrs, **coords)
