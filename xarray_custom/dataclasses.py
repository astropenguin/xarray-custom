"""Module for dataclasses of xarray DataArray.

This module provides functions which helps to create a custom DataArray class
of fixed dimensions, datatype, and coordinates. Two functions are available:

- ``dataarrayclass``: Class decorator which updates a custom DataArray class.
- ``coordtype``: Create a custom DataArray class for the definition of a coordinate.

Example:
    Here is an example to create a custom DataArray class::

        @dataarrayclass(('x', 'y'), accessor='custom')
        class CustomDataArray:
            x: coordtype('x', int) = 0
            y: coordtype('y', int) = 1
            z: coordtype(('x', 'y'), str) = 'spam'

            def double(self):
                return self * 2

    The code style is similar to that of Python's dataclass.
    A DataArray instance is then created using the class::

        dataarray = CustomDataArray([[0, 1], [2, 3]], x=[2, 2])
        print(dataarray)

        # <xarray.DataArray (x: 2, y: 2)>
        # array([[0, 1],
        #        [2, 3]])
        # Coordinates:
        # * x        (x) int64 2 2
        # * y        (y) int64 1 1
        #   z        (x, y) <U1 'spam' 'spam' 'spam' 'spam'

    Because ``dims`` and coordinates are pre-defined, it is much
    easier to create a DataArray with given data and coordinates.
    Custom methods can be used via an accessor::

        doubled = dataarray.custom.double()
        print(doubled)

        # <xarray.DataArray (x: 2, y: 2)>
        # array([[0, 2],
        #        [4, 6]])
        # Coordinates:
        # * x        (x) int64 2 2
        # * y        (y) int64 1 1
        #   z        (x, y) <U1 'spam' 'spam' 'spam' 'spam'

    Like NumPy, there are several special class methods
    to create a DataArray filled with some values::

        shape = 3, 3
        empty = CustomDataArray.empty(shape, ...)
        zeros = CustomDataArray.zeros(shape, ...)
        ones = CustomDataArray.ones(shape, ...)
        full = CustomDataArray.full(shape, fill_value=5, ...)


"""
__all__ = ["coordtype", "dataarrayclass"]


# standard library
from functools import wraps
from types import FunctionType
from typing import Any, Callable, Optional


# dependencies
import numpy as np
from xarray import DataArray, register_dataarray_accessor
from .typing import Attrs, Dims, Dtype, Name, Shape


# constants
ORDER = "C"


# class decorators
def dataarrayclass(
    dims: Dims,
    dtype: Optional[Dtype] = None,
    accessor: Optional[str] = None,
    docstring_style: str = "google",
) -> Callable:
    """Class decorator which updates a custom DataArray class.

    Args:
        dims: Dimensions of a custom DataArray.
        dtype: Datatype of a custom DataArray. Default is ``None``,
            which means that an input of any datatype is accepted.
        accessor: Name of a DataArray accessor.
            Methods in the decorated class are moved into the accessor.
        docstring_style: Style of docstrings of special methods.
            ``'google'`` is only available (``'numpy'`` will be added).

    Returns:
        decorator: Class decorator.

    Examples:
        To create a custom DataArray class::

            @dataarrayclass(('x', 'y'), accessor='custom')
            class CustomDataArray:
                x: coordtype('x', int) = 0
                y: coordtype('y', int) = 1
                z: coordtype(('x', 'y'), str) = 'spam'

                def double(self):
                    return self * 2

    """

    def decorator(cls: type) -> type:
        # create a custom __new__ method
        cls.__new__ = get_new(cls, dims, dtype)

        # move methods in the class to accessor
        move_methods_to_accessor(cls, accessor)

        # add special class attributes
        cls.dims = dims
        cls.dtype = dtype

        # add special class methods
        cls.zeros = zeros
        cls.ones = ones
        cls.empty = empty
        cls.full = full

        return cls

    return decorator


def coordtype(dims: Dims, dtype: Optional[Dtype] = None) -> type:
    """Create a custom DataArray class for the definition of a coordinate.

    Args:
        dims: Dimensions of a coordinate.
        dtype: Datatype of a custom DataArray. Default is ``None``,
            which means that an input of any datatype is accepted.

    Returns:
        CoordType: Custom DataArray class for a coordinate.

    """

    @dataarrayclass(dims, dtype)
    class CoordType:
        pass

    return CoordType


# helper functions
def get_new(cls: type, dims: Dims, dtype: Optional[Dtype] = None) -> Callable:
    """Create a custom __new__ method for the class.

    Args:
        cls: Custom DataArray class.
        dims: Dimensions of a custom DataArray.
        dtype: Datatype of a custom DataArray. Default is ``None``,
            which means that an input of any datatype is accepted.

    Returns:
        __new__: A method to create a DataArray instance.

    """

    def __new__(
        cls: type,
        data: Any,
        name: Optional[Name] = None,
        attrs: Optional[Attrs] = None,
        **coords,
    ) -> DataArray:
        # create custom DataArray without coordinates
        dataarray = DataArray(data, dims=dims, name=name, attrs=attrs)

        if dtype is not None:
            dataarray = dataarray.astype(dtype)

        if not hasattr(cls, "__annotations__"):
            return dataarray

        # add coordinates if they are defined in the class
        for name, coordtype in cls.__annotations__.items():
            shape = [dataarray.sizes[dim] for dim in coordtype.dims]

            if name in coords:
                dataarray.coords[name] = coordtype.full(shape, coords[name])
                continue

            if hasattr(cls, name):
                default = getattr(cls, name)
                dataarray.coords[name] = coordtype.full(shape, default)
                continue

            raise ValueError(
                f"No default value for a coordinate {repr(name)}. "
                "The value must be given as a keyword argument."
            )

        return dataarray

    return __new__


def move_methods_to_accessor(cls: type, accessor: Optional[str] = None) -> None:
    """Create a DataArray accessor and move methods in a class to it.

    Args:
        cls: Custom DataArray class.
        accessor: Name of a custom DataArray accessor.

    Returns:
        This function returns nothing.

    """
    # empty accessor
    class Accessor:
        def __init__(self, accessed):
            self.accessed = accessed

    # accessor method converter
    def to_accessor_method(func):
        @wraps(func)
        def wrapped(self, *args, **kwargs):
            return func(self.accessed, *args, **kwargs)

        return wrapped

    # move methods to accessor
    for name in dir(cls):
        obj = getattr(cls, name)

        if not isinstance(obj, FunctionType):
            continue

        if obj.__name__.startswith("__"):
            continue

        setattr(Accessor, name, to_accessor_method(obj))
        delattr(cls, name)

    # register accessor with given name
    if accessor is not None:
        register_dataarray_accessor(accessor)(Accessor)


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
