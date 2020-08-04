"""Module for utilities which help to create custom DataArray classes.

Currently this module only provides ``include`` class decorator
which can include a custom DataArray definition written in a file.

- include: Class decorator to include a custom DataArray definition in a file.

"""
__all__ = ["include"]


# standard library
import json
import re
from pathlib import Path
from typing import Any, Callable, Dict, Union


# dependencies
import toml
import yaml
from .dataclasses import coord


# constants
ATTRS = (
    "accessor",
    "desc",
    "dims",
    "dtype",
)
COORDS = "coords"
DEFAULT = "default"
JSON_RE = r"\.json$"
TOML_RE = r"\.toml$"
YAML_RE = r"\.ya?ml$"


# main functions
def include(path: Union[Path, str]) -> Callable:
    """Class decorator to include a custom DataArray definition in a file.

    File format of either JSON, TOML, or YAML is accepted.
    The following ``key=value`` pairs can be included if available.

    - ``dims=<array of string>``: Dimensions of the DataArray.
    - ``dtype=<string>``: Datatype of the DataArray.
    - ``desc=<string>``: Short description of the DataArray.
    - ``coords=<map of coord>``: Definition of coordinates (coords).
      Each coord is a map which can have the following ``key=value`` pairs.

      - ``dims=<array of string>``: Dimensions of a coordinate.
      - ``dtype=<string>``: Datatype of a coordinate.
      - ``desc=<string>``: Short description of a coordinate.
      - ``default=<any>``: Default value of a coordinate.

    Args:
        path: Path or filename of the file.

    Returns:
        Decorator to include the definition.

    Examples:
        If a definition is written in ``dataarray.toml``::

            # dataarray.toml

            dims = [ "x", "y" ]
            dtype = "float"
            desc = "DataArray class to represent images."

            [coords.x]
            dims = "x"
            dtype = "int"
            default = 0

            [coords.y]
            dims = "y"
            dtype = "int"
            default = 0

        then the following two class definitions are equivalent::

            @dataarrayclass(accessor='img')
            @include('dataarray.toml')
            class Image:
                pass

        ::

            @dataarrayclass(accessor='img')
            class Image:
                \"\"\"DataArray class to represent images.\"\"\"

                dims = 'x', 'y'
                dtype = float
                x: coord('x', int) = 0
                y: coord('y', int) = 0

    """
    path = Path(path).expanduser()
    loader = choose_loader_from(path)

    def decorator(cls: type) -> type:
        config = loader(path)
        coords = config.get(COORDS, {})

        for name in ATTRS:
            if name in config:
                setattr(cls, name, config[name])

        for name, values in coords.items():
            cls.__annotations__[name] = coord(**values)

            if DEFAULT in values:
                setattr(cls, name, values[DEFAULT])

        return cls

    return decorator


# helper functions
def choose_loader_from(path: Path) -> Callable:
    """Choose file loader based on a filename."""
    if re.search(JSON_RE, path.name):
        return load_json
    elif re.search(TOML_RE, path.name):
        return load_toml
    elif re.search(YAML_RE, path.name):
        return load_yaml
    else:
        raise ValueError("Invalid file format.")


def load_json(path: Path) -> Dict[str, Any]:
    """Load a JSON file to create a dictionary."""
    with path.open() as f:
        return json.load(f)


def load_toml(path: Path) -> Dict[str, Any]:
    """Load a TOML file to create a dictionary."""
    return toml.load(path)


def load_yaml(path: Path) -> Dict[str, Any]:
    """Load a YAML file to create a dictionary."""
    with path.open() as f:
        return yaml.load(f, Loader=yaml.SafeLoader)
