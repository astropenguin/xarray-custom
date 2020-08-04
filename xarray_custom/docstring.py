"""Module for making docstrings of functions updatable."""
__all__ = ["updatable_doc"]


# standard library
from functools import wraps
from textwrap import dedent
from typing import Callable


def updatable_doc(func: Callable) -> Callable:
    """Decorator for making the docstring of a function updatable.

    A decorated function has the ``set`` method, by which an update
    function for the docstring is registered to the function.

    Args:
        func: A function whose docstring should be updatable.

    Returns:
        Decorated function with the ``copy`` and ``set`` methods.

    Methods:
        copy: Copy the decorated function and return a new one.
        set: Set an update function to the decorated function.

    Examples::

        @updatable_doc
        def func():
            \"\"\"Docstring.\"\"\"
            pass

        # set an updater function
        func.set(lambda doc: doc.replace(".", "!"))

        # show the help of the decorated function
        help(func) # -> Docstring!

    """

    @wraps(func)
    def decorated(*args, **kwargs):
        return func(*args, **kwargs)

    def copy(self):
        return updatable_doc(self)

    def set(self, updater):
        self.__doc__.updater = updater
        return self

    decorated.__doc__ = UpdatableDoc(decorated.__doc__)
    decorated.copy = copy.__get__(decorated)
    decorated.set = set.__get__(decorated)

    return decorated


class UpdatableDoc(str):
    """Subclass of string for making docstrings updatable.

    An instance of it can have an update function as an attribute,
    which is used before running the ``expandtabs``, ``repr``, and ``str``
    methods. As a result, the output string can be dinamically updated.

    """

    def __new__(cls, doc: str) -> "UpdatableDoc":
        """Create an instance from a docstring."""
        return super().__new__(cls, cls.dedent(doc))

    def __init__(self, doc: str) -> None:
        """Initialize an instance with a None attribute."""
        self.updater = None

    def to_str(self):
        """Convert an instace to a normal string."""
        return super().__str__()

    @staticmethod
    def dedent(doc: str) -> str:
        """Custom dedent function for docstrings."""
        try:
            first, others = doc.split("\n", 1)
        except ValueError:
            first, others = doc, ""

        return first.lstrip() + "\n" + dedent(others)

    def update(self):
        """Update a docstring using the updater."""
        if self.updater is None:
            return self.to_str()
        else:
            return self.updater(self.to_str())

    def expandtabs(self, *args, **kwargs) -> str:
        """Update a docstring before running expandtabs().

        This is used to update a docstring in the builtin help().

        """
        return self.update().expandtabs(*args, **kwargs)

    def __str__(self) -> str:
        """Update a docstring before returning str."""
        return str(self.update())

    def __repr__(self) -> str:
        """Update a docstring before returning repr."""
        return repr(self.update())
