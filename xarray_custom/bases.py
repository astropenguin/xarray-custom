__all__ = ["DataArrayClass"]


# standard library
from typing import Callable, Dict, List, Optional, Union
from typing import get_type_hints


# dependencies
from .special import __new__ as new
from .special import empty, zeros, ones, full
from .typing import Attrs, Dims, Dtype, Name


# main classes
class DataArrayClassMeta(type):
    # class attributes for DataArray
    attrs: Optional[Attrs] = None
    dims: Optional[Dims] = None
    dtype: Optional[Dtype] = None
    name: Optional[Name] = None

    # class attributes for options
    accessor: Optional[str] = None
    desc: str = "No description."

    def __init__(cls, name, bases, attrs) -> None:
        def __new__(cls, *args, **kwargs):
            return cls.new(*args, **kwargs)

        cls.__new__ = __new__

    @property
    def data(cls) -> Dict[str, Union[Dims, Dtype]]:
        """Properties of multi-dimensional data."""
        return dict(dims=cls.dims, dtype=cls.dtype)

    @property
    def coords(cls) -> Dict[str, "DataArrayClassMeta"]:
        """Properties of coordinates of data."""
        coords = {}

        for name, hint in get_type_hints(cls).items():
            if isinstance(hint, DataArrayClassMeta):
                coords[name] = hint

        return coords

    @property
    def new(cls) -> Callable:
        cls._new = classmethod(new)
        return cls._new

    @property
    def empty(cls) -> Callable:
        cls._empty = classmethod(empty)
        return cls._empty

    @property
    def zeros(cls) -> Callable:
        cls._zeros = classmethod(zeros)
        return cls._zeros

    @property
    def ones(cls) -> Callable:
        cls._ones = classmethod(ones)
        return cls._ones

    @property
    def full(cls) -> Callable:
        cls._full = classmethod(full)
        return cls._full

    @property
    def __doc__(cls) -> str:
        return cls.new.__doc__

    def __repr__(cls) -> str:
        try:
            return cls.__class_repr__()
        except AttributeError:
            return super().__repr__()

    def __dir__(cls) -> List[str]:
        dir_cls = super().__dir__()
        dir_meta = dir(type(cls))

        return list(set(dir_cls) | set(dir_meta))


class DataArrayClass(metaclass=DataArrayClassMeta):
    """Base of custom DataArray class."""

    @classmethod
    def __class_repr__(cls):
        dims = str(cls.dims).replace("'", "")

        try:
            dtype = cls.dtype.__name__
        except AttributeError:
            dtype = cls.dtype

        return f"{cls.__name__}(dims={dims}, dtype={dtype})"
