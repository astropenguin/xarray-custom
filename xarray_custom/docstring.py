"""Module for formatting docstrings of custom DataArray classes.

This module provides the following functions which are used in
the metaclass (``DataArrayClassMeta``) of custom DataArray classes.

- ensure_desc: Make sure that a class has a valid ``desc`` attribute.
- update_doc: Copy a function and update docstring according to a class attributes.

"""
__all__ = ["ensure_desc", "update_doc"]


import re
from functools import wraps
from textwrap import dedent, TextWrapper
from typing import Callable, List


# constants
WIDTH = 80
INDENT = " " * 4


# main functions
def ensure_desc(cls):
    """Make sure that a class has a valid ``desc`` attribute.

    If it is None, then metaclass ``desc`` is set.
    Line breaks and indents are replaced with whitespaces.

    Args:
        cls: Custom DataArray class.

    Returns:
        This function returns nothing.

    """
    if cls.desc is None:
        cls.desc = type(cls).desc
    else:
        cls.desc = re.sub(r"\n\s*", " ", cls.desc)


def update_doc(func: Callable, cls: type) -> Callable:
    """Copy a function and update docstring according to a class attributes.

    Args:
        func: Function to be copied and updated.
        cls: Custom DataArray class whose attributes are used for updating.

    Returns:
        Copied function whose docstring is updated.

    """
    formatter = dict(
        coords_args=wrap(create_coords_args(cls), INDENT, INDENT * 2),
        summary=wrap(create_summary(cls), "- ", "  "),
    )

    @wraps(func)
    def copied(*args, **kwargs):
        return func(*args, **kwargs)

    copied.__doc__ = dedent(copied.__doc__).format(**formatter)
    return copied


# helper functions
def create_summary(cls: type) -> List[str]:
    """Create docstrings to summarize a class."""
    dims = str(cls.dims).replace("'", "")

    try:
        dtype = cls.dtype.__name__
    except AttributeError:
        dtype = cls.dtype

    yield f"desc: {cls.desc}"
    yield f"dims: {dims}"
    yield f"dtype: {dtype}"
    yield f"accessor: {cls.accessor}"
    yield f"coords: {', '.join(cls.coords)}"


def create_coords_args(cls: type) -> List[str]:
    """Create docstrings of coordinates of a class."""
    for name, ctype in cls.coords.items():
        dims = str(ctype.dims).replace("'", "")
        dtype = str(ctype.dtype).replace("'", "")

        yield f"{name}: (dims={dims}, dtype={dtype}) {ctype.desc}"


def wrap(docs: List[str], initial_indent: str, subsequent_indent: str) -> str:
    """Wrap and join docstrings with indents.

    Args:
        docs: Docstrings to be wrapped and joined.
        initial_indent: String to be prepended to the first line
            of each wrapped docstring.
        subsequent_indent: String to be prepended to the second,
            third, ... lines of each wrapped docstring.

    Returns:
        Wrapped and joined string.

    """
    wrapper = TextWrapper(WIDTH, initial_indent, subsequent_indent)
    return "\n".join("\n".join(wrapper.wrap(doc)) for doc in docs)
