"""Module for creating custom DataArray classes.

This module provides functions which help to create a custom DataArray class
with fixed dimensions, datatype, and coordinates. Two functions are available:

- ``dataarrayclass``: Class decorator which updates a custom DataArray class.
- ``ctype``: Create a custom DataArray class for the definition of a coordinate.

Examples:
    To create a custom DataArray class to represent images::
        @dataarrayclass(accessor='img')
        class Image:
            dims = 'x', 'y'
            dtype = float
            x: ctype('x', int) = 0
            y: ctype('y', int) = 0
            w: ctype(('x', 'y'), float) = 1.0

            def normalize(self):
                return self / self.max()

    The code style is similar to that of Python's dataclass.
    A DataArray is then created using the class::
        image = Image([[0, 1], [2, 3]], x=[0, 1], y=[0, 1])
        print(image)

        # <xarray.DataArray (x: 2, y: 2)>
        # array([[0., 1.],
        #        [2., 3.]])
        # Coordinates:
        #   * x        (x) int64 0 1
        #   * y        (y) int64 0 1
        #     w        (x, y) float64 1.0 1.0 1.0 1.0

    Because ``dims``, ``dtype``, and coordinates are pre-defined,
    it is much easier to create a DataArray with given data.
    Custom methods can be used via an accessor::
        normalized = image.img.normalize()
        print(normalized)

        # <xarray.DataArray (x: 2, y: 2)>
        # array([[0.        , 0.33333333],
        #        [0.66666667, 1.        ]])
        # Coordinates:
        #   * x        (x) int64 0 1
        #   * y        (y) int64 0 1
        #     w        (x, y) float64 1.0 1.0 1.0 1.0

    Like NumPy, several special class methods are available
    to create a DataArray filled with some values::
        ones = Image.ones((2, 2))
        print(ones)

        # <xarray.DataArray (x: 2, y: 2)>
        # array([[1., 1.],
                 [1., 1.]])
        # Coordinates:
        #   * x        (x) int64 0 0
        #   * y        (y) int64 0 0
        #     w        (x, y) float64 1.0 1.0 1.0 1.0

"""
__all__ = ["ctype", "dataarrayclass"]


# standard library
from typing import Callable, Optional, Union


# dependencies
from .accessor import add_methods_to_accessor
from .ensuring import ensure_dataarrayclass
from .special import add_special_methods
from .typing import Dims, Dtype


# main functions
def ctype(dims: Dims, dtype: Optional[Dtype] = None) -> type:
    """Create a custom DataArray class for the definition of a coordinate.

    Args:
        dims: Dimensions of a coordinate.
        dtype: Datatype of a custom DataArray. Default is ``None``,
            which means that an input of any datatype is accepted.

    Returns:
        ctype: Custom DataArray class for a coordinate.

    """
    return dataarrayclass(type("ctype", (), dict(dims=dims, dtype=dtype)))


def dataarrayclass(
    cls: Optional[type] = None,
    *,
    accessor: Optional[str] = None,
    strict_dims: bool = False,
    strict_dtype: bool = False,
    docstring_style: str = "google",
) -> Union[type, Callable]:
    """Class decorator which updates a custom DataArray class.

    Keyword Args:
        accessor: Name of an accessor for the custom DataArray.
            User-defined methods in the class are added to the accessor.
        docstring_style: Style of docstrings of special methods.
            ``'google'`` is only available (``'numpy'`` will be added).
        strict_dims: Whether ``dims`` is consistent with superclasses.
        strict_dtype: Whether ``dtype`` is consistent with superclasses.

    Returns:
        decorator: Returned if any keyword-only arguments are given.
        decorated: Returned if no keyword-only arguments are given.

    Examples:
        To create a custom DataArray class to represent images::
            @dataarrayclass(accessor='img')
            class Image:
                dims = 'x', 'y'
                dtype = float
                x: ctype('x', int) = 0
                y: ctype('y', int) = 0

                def normalize(self):
                    return self / self.max()
    """

    def decorator(cls: type) -> type:
        ensure_dataarrayclass(cls, strict_dims, strict_dtype)
        add_methods_to_accessor(cls, accessor)
        add_special_methods(cls)

        return cls

    if cls is None:
        return decorator
    else:
        return decorator(cls)
