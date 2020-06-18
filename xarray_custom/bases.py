__all__ = ["DataArrayClass"]


# standard library
from functools import wraps
from typing import Callable, Dict, Optional, Union
from typing import get_type_hints


# dependencies
from .docstring import update_doc
from .special import empty, zeros, ones, full, new
from .typing import Attrs, Dims, Dtype, Name


# helper classes
class classproperty(property):
    pass


class DataArrayClassMeta(type):
    def __new__(meta, name, bases, namespace):
        for key, obj in namespace.copy().items():
            if isinstance(obj, classmethod):
                setattr(meta, key, obj.__func__)
                namespace.pop(key)

            if isinstance(obj, classproperty):
                setattr(meta, key, obj)
                namespace.pop(key)

        return super().__new__(meta, name, bases, namespace)


# main classes
class DataArrayClass(metaclass=DataArrayClassMeta):
    # class attributes for DataArray
    attrs: Optional[Attrs] = None
    dims: Optional[Dims] = None
    dtype: Optional[Dtype] = None
    name: Optional[Name] = None
    desc: Optional[str] = None
    accessor: Optional[str] = None

    @wraps(new)
    def __new__(cls, *args, **kwargs):
        return new(cls, *args, **kwargs)

    @classproperty
    def new(cls) -> Callable:
        return update_doc(new, cls).__get__(cls)

    @classproperty
    def empty(cls) -> Callable:
        return update_doc(empty, cls).__get__(cls)

    @classproperty
    def zeros(cls) -> Callable:
        return update_doc(zeros, cls).__get__(cls)

    @classproperty
    def ones(cls) -> Callable:
        return update_doc(ones, cls).__get__(cls)

    @classproperty
    def full(cls) -> Callable:
        return update_doc(full, cls).__get__(cls)

    @classproperty
    def data(cls) -> Dict[str, Union[Dims, Dtype]]:
        """Properties of multi-dimensional data."""
        return dict(dims=cls.dims, dtype=cls.dtype)

    @classproperty
    def coords(cls) -> Dict[str, type]:
        """Properties of coordinates of data."""
        coords = {}

        for name, hint in get_type_hints(cls).items():
            if not isinstance(hint, type):
                continue

            if issubclass(hint, DataArrayClass):
                coords[name] = hint

        return coords

    @classproperty
    def __doc__(cls) -> str:
        """Updatable class docstring."""
        return update_doc(new, cls).__doc__

    @classmethod
    def __repr__(cls) -> str:
        """Updatable class repr string."""
        dims = str(cls.dims).replace("'", "")
        dtype = str(cls.dtype).replace("'", "")

        return f"{cls.__name__}(dims={dims}, dtype={dtype})"
