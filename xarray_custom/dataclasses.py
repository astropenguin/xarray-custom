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
from typing import Any, Callable, Optional


# dependencies
from xarray import DataArray
from .typing import Attrs, Dims, Dtype, Name
from .accessor import add_methods_to_accessor
from .special import add_special_methods


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
        add_methods_to_accessor(cls, accessor)

        # add special class attributes
        cls.dims = dims
        cls.dtype = dtype

        # add special class methods
        add_special_methods(cls)

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
