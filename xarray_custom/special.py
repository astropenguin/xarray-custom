"""Module for special methods of custom DataArray classes.

This module provides the following methods which are added to
a DataArray class at its creation. They have two placeholders
in their docstrings ({summary} and {coords_args}), which are
automatically filled according to the class attributes.

- new: Create a custom DataArray from data and coordinates.
- empty: Create a custom DataArray filled with uninitialized values.
- zeros: Create a custom DataArray filled with zeros.
- ones: Create a custom DataArray filled with ones.
- full: Create a custom DataArray filled with ``fill_value``.

"""
__all__ = ["new", "empty", "zeros", "ones", "full"]

# standard library
from typing import Any, Callable, Optional


# dependencies
import numpy as np
from xarray import DataArray
from .typing import Attrs, Dtype, Name, Shape


# constants
ORDER: str = "C"


def add_classmethods(cls: type, updater: Optional[Callable] = None) -> type:
    """Add special class methods to a DataArray class.

    Args:
        updater: Function to update docstrings of the class methods.
            Its input must be a docstring (``str``) and its output
            must be the updated docstring (``str``). If not specified
            (by default), any docstrings will not be updated.

    Returns:
        The same DataArray class as the input.

    """
    cls.__new__ = new.copy().set(updater)

    cls.empty = classmethod(empty.copy().set(updater))
    cls.zeros = classmethod(zeros.copy().set(updater))
    cls.ones = classmethod(ones.copy().set(updater))
    cls.full = classmethod(full.copy().set(updater))

    return cls


def new(
    cls: type,
    data: Any,
    name: Optional[Name] = None,
    attrs: Optional[Attrs] = None,
    **coords,
) -> DataArray:
    """\
    Create a custom DataArray from data and coordinates.

    {summary}

    Args:
        data: Values of the DataArray. Its shape must match class ``dims``.
            If class ``dtype`` is defined, it will be casted to that type.
            If it cannot be casted, a ``ValueError`` will be raised.
        name: Name of the DataArray. Default is class ``name``.
        attrs: Attributes of the DataArray. Default is class ``attrs``.
        **coords: Coordinates of the DataArray defined by the class.

    Returns:
        Custom DataArray.

    Keyword Args:
    {coords_args}

    See Also:
        - empty: Create a custom DataArray filled with uninitialized values.
        - zeros: Create a custom DataArray filled with zeros.
        - ones: Create a custom DataArray filled with ones.
        - full: Create a custom DataArray filled with ``fill_value``.

    """
    if cls.dims is None:
        raise ValueError("Dimensions (dims) are not defined.")

    name = name or cls.name
    attrs = attrs or cls.attrs
    dataarray = DataArray(data, dims=cls.dims, name=name, attrs=attrs)

    if cls.dtype is not None:
        dataarray = dataarray.astype(cls.dtype)

    for name, ctype in cls.coords.items():
        shape = [dataarray.sizes[dim] for dim in ctype.dims]

        if name in coords:
            dataarray.coords[name] = ctype.full(shape, coords[name])
            continue

        if hasattr(cls, name):
            dataarray.coords[name] = ctype.full(shape, getattr(cls, name))
            continue

        raise ValueError(
            f"A value for a coordinate {name} is not defined by default. "
            f"This must be given as a keyword argument ({name}=<value>)."
        )

    return dataarray


def zeros(
    cls: type,
    shape: Shape,
    dtype: Optional[Dtype] = None,
    order: str = ORDER,
    name: Optional[Name] = None,
    attrs: Optional[Attrs] = None,
    **coords,
) -> DataArray:
    """\
    Create a custom DataArray filled with zeros.

    {summary}

    Args:
        shape: Shape of the DataArray. It must match class ``dims``.
        dtype: Datatype of the DataArray. Default is 64-bit float.
            It is ignored if class ``dtype`` is defined.
        order: Order of data in memory. Either ``'C'`` (row-major; C-style)
            or ``'F'`` (column-major; Fortran-style) is accepted.
        name: Name of the DataArray. Default is class ``name``.
        attrs: Attributes of the DataArray. Default is class ``attrs``.
        **coords: Coordinates of the DataArray defined by the class.

    Returns:
        Custom DataArray filled with zeros.

    Keyword Args:
    {coords_args}

    """
    return cls(np.zeros(shape, dtype, order), name, attrs, **coords)


def ones(
    cls: type,
    shape: Shape,
    dtype: Optional[Dtype] = None,
    order: str = ORDER,
    name: Optional[Name] = None,
    attrs: Optional[Attrs] = None,
    **coords,
) -> DataArray:
    """\
    Create a custom DataArray filled with ones.

    {summary}

    Args:
        shape: Shape of the DataArray. It must match class ``dims``.
        dtype: Datatype of the DataArray. Default is 64-bit float.
            It is ignored if class ``dtype`` is defined.
        order: Order of data in memory. Either ``'C'`` (row-major; C-style)
            or ``'F'`` (column-major; Fortran-style) is accepted.
        name: Name of the DataArray. Default is class ``name``.
        attrs: Attributes of the DataArray. Default is class ``attrs``.
        **coords: Coordinates of the DataArray defined by the class.

    Returns:
        Custom DataArray filled with ones.

    Keyword Args:
    {coords_args}

    """
    return cls(np.ones(shape, dtype, order), name, attrs, **coords)


def empty(
    cls: type,
    shape: Shape,
    dtype: Optional[Dtype] = None,
    order: str = ORDER,
    name: Optional[Name] = None,
    attrs: Optional[Attrs] = None,
    **coords,
) -> DataArray:
    """\
    Create a custom DataArray filled with uninitialized values.

    {summary}

    Args:
        shape: Shape of the DataArray. It must match class ``dims``.
        dtype: Datatype of the DataArray. Default is 64-bit float.
            It is ignored if class ``dtype`` is defined.
        order: Order of data in memory. Either ``'C'`` (row-major; C-style)
            or ``'F'`` (column-major; Fortran-style) is accepted.
        name: Name of the DataArray. Default is class ``name``.
        attrs: Attributes of the DataArray. Default is class ``attrs``.
        **coords: Coordinates of the DataArray defined by the class.

    Returns:
        Custom DataArray filled with uninitialized values.

    Keyword Args:
    {coords_args}

    """
    return cls(np.empty(shape, dtype, order), name, attrs, **coords)


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
    """\
    Create a custom DataArray filled with ``fill_value``.

    {summary}

    Args:
        shape: Shape of the DataArray. It must match class ``dims``.
        fill_value: Scalar value to fill the custom DataArray.
        dtype: Datatype of the DataArray. Default is 64-bit float.
            It is ignored if class ``dtype`` is defined.
        order: Order of data in memory. Either ``'C'`` (row-major; C-style)
            or ``'F'`` (column-major; Fortran-style) is accepted.
        name: Name of the DataArray. Default is class ``name``.
        attrs: Attributes of the DataArray. Default is class ``attrs``.
        **coords: Coordinates of the DataArray defined by the class.

    Returns:
        Custom DataArray filled with ``fill_value``.

    Keyword Args:
    {coords_args}

    """
    return cls(np.full(shape, fill_value, dtype, order), name, attrs, **coords)
