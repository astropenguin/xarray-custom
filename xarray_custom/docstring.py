"""Module for formatting docstrings of custom DataArray classes.

This module provides a function (``format_doc``) which formats
docstrings of special methods in a custom DataArray class.

"""
__all__ = ["format_doc"]


# standard library
from functools import wraps
from textwrap import dedent, indent, TextWrapper
from typing import Callable


# constants
HALF_INDENT = " " * 2
INDENT = " " * 4
WIDTH = 80


# main functions
def format_doc(func: Callable, cls: type) -> Callable:
    """Copy a function and format its docstring according to a class attributes.

    Args:
        func: Function to be copied and formatted.
        cls: Custom DataArray class whose attributes are used for formatting.

    Returns:
        func: Copied function whose docstring is updated.

    """
    summary = create_summary(cls)
    coords_doc = create_coords_doc(cls)

    doc = dedent(func.__doc__)
    doc = doc.format(summary=summary, coords_doc=coords_doc)

    func = copy(func)
    func.__doc__ = doc
    return func


# helper functions
def copy(func: Callable) -> Callable:
    """Copy a function as a decorated one."""

    @wraps(func)
    def copied(*args, **kwargs):
        return func(*args, **kwargs)

    return copied


def create_summary(cls: type) -> str:
    """Create a docstring to summarize a class."""
    coords = [f"``{name}``" for name in cls.ctypes]

    docs = [
        f"- **desc:** {cls.desc}",
        f"- **dims:** ``{cls.dims!r}``",
        f"- **dtype:** ``{cls.dtype!r}``",
        f"- **coords:** {', '.join(coords)}",
    ]

    return "\n".join(indent_wrap(doc, False) for doc in docs)


def create_coords_doc(cls: type) -> str:
    """Create a docstring of coordinates of a class."""
    docs = []

    for name, ctype in cls.ctypes.items():
        docs.append(
            indent_wrap(
                f"{name}: "
                f"(dims: ``{ctype.dims!r}``, "
                f"dtype: ``{ctype.dtype!r}``) "
                f"{ctype.desc}"
            ),
        )

    return "\n".join(indent(doc, INDENT) for doc in docs)


def indent_wrap(doc: str, inside_section: bool = True) -> str:
    """Wrap a docstring with leading indents from the second line.

    Args:
        doc: String to be wrapped.
        inside_section: Whether the wrapped ``doc`` is used inside
            a section (e.g., Args or Returns). If False (outside),
            half indent (two whitespaces) is alternatively used.

    Returns:
        wrapped: Wrapped string.

    """
    indent = INDENT if inside_section else HALF_INDENT
    return "\n".join(TextWrapper(WIDTH, "", indent).wrap(doc))
