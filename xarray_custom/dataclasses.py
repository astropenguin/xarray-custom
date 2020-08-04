"""Module for creating custom DataArray classes.

This module provides functions which help to create a custom DataArray class
with fixed dimensions, datatype, and coordinates. Two functions are available:

- ``dataarrayclass``: Class decorator to construct a custom DataArray class.
- ``coord``: Create a DataArray class for the definition of a coordinate.

Examples:
    To create a custom DataArray class to represent images::

        @dataarrayclass
        class Image:
            \"\"\"DataArray class to represent images.\"\"\"

            accessor = 'img'
            dims = 'x', 'y'
            dtype = float

            x: coord('x', int) = 0
            y: coord('y', int) = 0

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

    Like NumPy, several special class methods are available
    to create a DataArray filled with some values::

        ones = Image.ones((2, 2))
        print(ones)

        # <xarray.DataArray (x: 2, y: 2)>
        # array([[1., 1.],
        #        [1., 1.]])
        # Coordinates:
        #   * x        (x) int64 0 0
        #   * y        (y) int64 0 0

    Inheriting a custom DataArray class is possible to
    create a derivative DataArray class::

        class WeightedImage(Image):
            accessor = 'wimg'
            w: coord(('x', 'y'), float) = 1.0

        zeros = Weightedimage.zeros((2, 2))
        print(zeros)

        # <xarray.DataArray (x: 2, y: 2)>
        # array([[1., 1.],
        #        [1., 1.]])
        # Coordinates:
        #   * x        (x) int64 0 0
        #   * y        (y) int64 0 0
        #     w        (x, y) float64 1.0 1.0 1.0 1.0

"""
__all__ = ["coord", "dataarrayclass"]


# standard library
from typing import Optional


# dependencies
from .bases import DataArrayClassBase
from .typing import Dims, Dtype


# main functions
def coord(
    dims: Optional[Dims] = None,
    dtype: Optional[Dtype] = None,
    desc: Optional[str] = None,
    **_
) -> type:
    """Create a DataArray class for the definition of a coordinate.

    Args:
        dims: Dimensions of the coordinate.
        dtype: Datatype of the coordinate. Default is ``None``,
            which means that an input of any datatype is accepted.
        desc: Short description of the coordinate.

    Returns:
        DataArray class for the coordinate.

    """
    if dims is None:
        dims = ()

    if desc is None:
        namespace = dict(dims=dims, dtype=dtype)
    else:
        namespace = dict(dims=dims, dtype=dtype, desc=desc)

    return type("Coord", (DataArrayClassBase,), namespace)


def dataarrayclass(cls: type) -> type:
    """Class decorator to construct a custom DataArray class.

    A class can define properties of DataArray to be created.
    The following class variables are accepted (see examples).

    - ``dims`` (str or tuple of str): Name(s) of dimension(s).
    - ``dtype`` (str or type): Datatype of DataArray.
    - ``desc`` (str): Short description of DataArray. Users can
        alternatively define it by a docstring (``__doc__``).
    - ``accessor`` (str): Name of accessor (if necessary).

    The coordinates of DataArray can also be defined as class
    variables using the ``coord`` function (see examples).

    Finally users can define custom instance methods which can
    be used by an accessor whose name is defined by ``accessor``.

    Args:
        cls: Class to be decorated.

    Returns:
        Decorated class as a DataArray class.

    Examples:
        To create a custom DataArray class to represent images::

            @dataarrayclass
            class Image:
                \"\"\"DataArray class to represent images.\"\"\"

                accessor = 'img'
                dims = 'x', 'y'
                dtype = float

                x: coord('x', int) = 0
                y: coord('y', int) = 0

                def normalize(self):
                    return self / self.max()

    """
    return type(cls.__name__, (DataArrayClassBase,), cls.__dict__.copy())
