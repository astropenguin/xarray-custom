"""Module for providing the base class of DataArray classes."""
__all__ = ["DataArrayClassBase"]


# standard library
from re import sub
from itertools import chain
from typing import get_type_hints, List, Optional
from textwrap import TextWrapper


# dependencies
from .typing import Dims, Dtype
from .accessor import add_accessors
from .special import add_classmethods


# constants
DOC_WIDTH: int = 78
DOC_INDENT: str = " " * 4


class classproperty(property):
    """Decorator to convert a function as a class property."""

    pass


class DataArrayClassMeta(type):
    """Metaclass only for the ``DataArrayClassBase`` class."""

    def __new__(meta, name: str, bases: tuple, namespace: dict) -> type:
        """Create a DataArray class as an instance of the metaclass.

        This method (1) convert instance methods to class properties
        and (2) use the class ``__doc__`` as the ``desc`` attribute.

        """
        for key, obj in namespace.copy().items():
            if isinstance(obj, classproperty):
                setattr(meta, key, obj)
                namespace.pop(key)

        if isinstance(namespace.get("dims"), str):
            namespace["dims"] = (namespace["dims"],)

        if namespace.get("__doc__") and not namespace.get("desc"):
            namespace["desc"] = namespace["__doc__"]

        return super().__new__(meta, name, bases, namespace)

    def __init__(cls, name: str, bases: tuple, namespace: dict) -> None:
        """Initialize a DataArray class with class-specific customization.

        This method (1) adds special class methods (e.g., ``__new__``, ``ones``)
        with a docstring updater and (2) adds unique and common accessors.

        """

        def updater(doc):
            return doc.format(cls=cls)

        add_classmethods(cls, updater)
        add_accessors(cls, cls.accessor)

    def __repr__(cls) -> str:
        """Customizable repr feature of a DataArray class.

        If a DataArray class has a ``__class_repr__`` classmethod,
        the default class ``__repr__`` behavior is overridden by it.

        """
        try:
            return cls.__class_repr__()
        except AttributeError:
            return super().__repr__()

    def __dir__(cls) -> List[str]:
        """List names in the namespace of a DataArray class."""
        dirs = super().__dir__(), dir(type(cls))
        return list(set(chain.from_iterable(dirs)))


class DataArrayClassBase(metaclass=DataArrayClassMeta):
    """Base class for DataArray classes."""

    desc: str = "No description."
    dims: Optional[Dims] = None
    dtype: Optional[Dtype] = None
    accessor: Optional[str] = None

    @classproperty
    def doc(cls) -> "Doc":
        return Doc.from_dataarrayclass(cls)

    @classproperty
    def coords(cls) -> "Coords":
        """Dictionary of coordinate definitions."""
        return Coords.from_dataarrayclass(cls)

    @classproperty
    def __doc__(cls) -> str:
        """Updatable class docstring."""
        return cls.__new__.__doc__


class Coords(dict):
    """Class for the coordinate definitions of a DataArray class."""

    wrapper = TextWrapper(DOC_WIDTH, DOC_INDENT, DOC_INDENT * 2)

    @classmethod
    def from_dataarrayclass(cls, dataarrayclass: type) -> "Coords":
        """Create a Coords instance from a DataArray class."""
        coords = {}

        for name, hint in get_type_hints(dataarrayclass).items():
            if not isinstance(hint, type):
                continue

            if issubclass(hint, DataArrayClassBase):
                coords[name] = hint

        return cls(coords)

    @property
    def doc(self) -> str:
        """Create the Google-style docstring of an instance."""
        if not self:
            return ""

        docs = ["Keyword Args:"]

        for name, coord in self.items():
            doc = f"{name}: {coord.doc.unwrap}"
            docs.append("\n".join(self.wrapper.wrap(doc)))

        return "\n".join(docs)


class Doc(str):
    """Class for the docstring of a DataArray class."""

    wrapper = TextWrapper(DOC_WIDTH - len(DOC_INDENT), "", "")

    def __new__(cls, doc: str) -> "Doc":
        """Create an instance from a docstring."""
        return super().__new__(cls, "\n".join(cls.wrapper.wrap(doc)))

    @classmethod
    def from_dataarrayclass(cls, dataarrayclass: type) -> "Doc":
        """Create an Doc instace from a DataArray class."""
        desc = sub(r"[\n|\s]+", " ", dataarrayclass.desc)
        dims = f"dims={dataarrayclass.dims!r}"
        dtype = f"dims={dataarrayclass.dtype!r}"

        return cls(f"({dims}, {dtype}) {desc}")

    @property
    def unwrap(self):
        """Convert an instance to an unwrap docstring."""
        return sub(r"[\n|\s]+", " ", self)
