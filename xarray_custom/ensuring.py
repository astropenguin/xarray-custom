"""Module for ensuring custom DataArray classes.

This module provides a function (``ensure_dataarrayclass``)
to ensure that a class is valid for ``DataArrayClass`` at decoration.

"""
__all__ = ["ensure_dataarrayclass"]


# standard library
import re

# dependencies
from .abc import DataArrayClass


# constants
CTYPES = "ctypes"
DESC = "desc"
DIMS = "dims"
DTYPE = "dtype"


# main functions
def ensure_dataarrayclass(
    cls: type, strict_dims: bool = False, strict_dtype: bool = False
) -> type:
    """Ensure that a class is valid for ``DataArrayClass``.

    This function makes sure that the class has required attributes
    (``ctypes``, ``dims``, ``doc``, and ``dtypes``) and is a subclass
    of ``DataArrayClass``. Options (``strict_*``) check more strictly
    whether ``dims`` and ``dtype`` are consistent with superclasses.

    Args:
        cls: Class to be ensured.
        strict_dims: Whether ``dims`` is consistent with superclasses.
        strict_dtype: Whether ``dtype`` is consistent with superclasses.

    Returns:
        cls: Same object as ``cls`` in the arguments.

    """
    ensure_subclass(cls)
    ensure_dims(cls, strict_dims)
    ensure_dtype(cls, strict_dtype)
    ensure_ctypes(cls)
    ensure_desc(cls)

    return cls


# helper functions
def ensure_ctypes(cls: type) -> type:
    """Ensure that a class has a valid ``ctypes`` attribute.

    If the attribute does not exist in the class,
    an empty dictionary is first set to initialize it.
    Then ``ctype``s are picked up from annotations of
    class attributes in the class and its superclasses.

    Args:
        cls: Class to be ensured.

    Returns:
        cls: Same object as ``cls`` in the arguments.

    """
    if not hasattr(cls, CTYPES):
        cls.ctypes = {}

    for sub in reversed(cls.mro()):
        if not hasattr(sub, "__annotations__"):
            continue

        for name, type_ in sub.__annotations__.items():
            if issubclass(type_, DataArrayClass):
                cls.ctypes[name] = type_

    return cls


def ensure_dims(cls: type, strict: bool = True) -> type:
    """Ensure that a class has a valid ``dims`` attribute.

    If the attribute does not exist in the class,
    an error is raised because it is mandatory.

    Args:
        cls: Class to be ensured.
        strict: Whether ``dims`` is consistent with superclasses.

    Returns:
        cls: Same object as ``cls`` in the arguments.

    Raises:
        AttributeError: Raised if ``dims`` does not exist.
        ValueError: Raised if ``dims`` is not a superset of (``strict=False``)
            or not equal to (``True``) any ``dims`` of superclasses.

    """
    if not hasattr(cls, DIMS):
        raise AttributeError(f"Must have {DIMS} attribute.")

    for sub in cls.mro():
        if not hasattr(sub, DIMS):
            continue

        if strict and set(cls.dims) != set(sub.dims):
            raise ValueError("Dims must be a superset of any of superclasses.")
        elif set(cls.dims) < set(sub.dims):
            raise ValueError("Dims must be equal to any of superclasses.")

    return cls


def ensure_desc(cls: type) -> type:
    """Ensure that a class has a valid ``desc`` attribute.

    If the attribute does not exist in the class,
    ``__doc__ or 'No description.'`` is set.
    Line breaks and indents are replaced with whitespaces.

    Args:
        cls: Class to be ensured.

    Returns:
        cls: Same object as ``cls`` in the arguments.

    """
    if not hasattr(cls, DESC):
        cls.desc = cls.__doc__ or "No description."

    cls.desc = re.sub(r"\n\s*", " ", cls.desc)
    return cls


def ensure_dtype(cls: type, strict: bool = True) -> type:
    """Ensure that a class has a valid ``dtype`` attribute.

    If the attribute does not exist in the class, ``None`` is set.

    Args:
        cls: Class to be ensured.
        strict: Whether ``dtype`` is consistent with superclasses.

    Returns:
        cls: Same object as ``cls`` in the arguments.

    Raises:
        ValueError: Raised if ``strict==True`` and ``dtype``
            is not equal to any of superclasses.

    """
    if not hasattr(cls, DTYPE):
        cls.dtype = None

    for sub in cls.mro():
        if not hasattr(sub, DTYPE):
            continue

        if strict and cls.dtype != sub.dtype:
            raise ValueError("Dtype must be equal to any of superclasses.")

    return cls


def ensure_subclass(cls: type) -> type:
    """Ensure that a class is a subclass of ``DataArrayClass``.

    Args:
        cls: Class to be ensured.

    Returns:
        cls: Same object as ``cls`` in the arguments.

    """
    DataArrayClass.register(cls)
    return cls
