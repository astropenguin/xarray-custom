# standard library
from functools import wraps
from types import MethodType
from typing import Any, Optional


# dependencies
import numpy as np
from xarray import DataArray, register_dataarray_accessor
from .typing import Attrs, Dtype, Name, Shape


# constants
ORDER = "C"


# helper functions
def move_methods_to_accessor(cls: type, accessor_name: str) -> None:
    """Create a DataArray accessor and move methods in a class to it.

    Args:
        cls: Custom DataArray class.
        accessor_name: Name of a custom DataArray accessor.

    Returns:
        This function returns nothing.

    """

    # empty accessor
    class Accessor:
        def __init__(self, accessed):
            self.accessed = accessed

    # move methods to accessor
    for name in dir(cls):
        method = getattr(cls, name)

        if not isinstance(method, MethodType):
            continue

        if method.__name__.startswith("__"):
            continue

        @wraps(method)
        def accessor_method(self, *args, **kwargs):
            return method(self.accessed, *args, **kwargs)

        setattr(Accessor, name, accessor_method)
        delattr(cls, name)

    # register accessor with given name
    register_dataarray_accessor(accessor_name)(Accessor)


# special class methods
@classmethod
def zeros(
    cls: type,
    shape: Shape,
    dtype: Optional[Dtype] = None,
    order: str = ORDER,
    name: Optional[Name] = None,
    attrs: Optional[Attrs] = None,
    **coords
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
    **coords
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
    **coords
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
    **coords
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
