"""Module for special class methods of DataArray classes."""
__all__ = ["add_classmethods"]


# standard library
from typing import Any, Callable, Optional


# dependencies
import numpy as np
from xarray import DataArray
from .docstring import updatable_doc
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


@updatable_doc
def new(
    cls: type,
    data: Any,
    name: Optional[Name] = None,
    attrs: Optional[Attrs] = None,
    **coords,
) -> DataArray:
    """Create a custom DataArray from data and coordinates.

    {cls.doc}

    Args:
        data: Values of the DataArray. Its shape must match class ``dims``.
            If class ``dtype`` is defined, it will be casted to that type.
            If it cannot be casted, a ``ValueError`` will be raised.
        name: Name of the DataArray. Default is class ``name``.
        attrs: Attributes of the DataArray. Default is class ``attrs``.
        **coords: Coordinates of the DataArray defined by the class.

    Returns:
        Custom DataArray.

    {cls.coords.doc}

    """
    dataarray = DataArray(data, dims=cls.dims, name=name, attrs=attrs)

    if cls.dtype is not None:
        dataarray = dataarray.astype(cls.dtype)

    for name, coord in cls.coords.items():
        shape = [dataarray.sizes[dim] for dim in coord.dims]

        if name in coords:
            dataarray.coords[name] = coord.full(shape, coords[name])
            continue

        if hasattr(cls, name):
            dataarray.coords[name] = coord.full(shape, getattr(cls, name))
            continue

        raise ValueError(
            f"Default value for a coordinate {name} is not defined. "
            f"It must be given as a keyword argument ({name}=<value>)."
        )

    return dataarray


@updatable_doc
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

    {cls.doc}

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

    {cls.coords.doc}

    """
    return cls(np.empty(shape, dtype, order), name, attrs, **coords)


@updatable_doc
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

    {cls.doc}

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

    {cls.coords.doc}

    """
    return cls(np.zeros(shape, dtype, order), name, attrs, **coords)


@updatable_doc
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

    {cls.doc}

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

    {cls.coords.doc}

    """
    return cls(np.ones(shape, dtype, order), name, attrs, **coords)


@updatable_doc
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

    {cls.doc}

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

    {cls.coords.doc}

    """
    return cls(np.full(shape, fill_value, dtype, order), name, attrs, **coords)
