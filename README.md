# xarray-custom

[![PyPI](https://img.shields.io/pypi/v/xarray-custom.svg?label=PyPI&style=flat-square)](https://pypi.org/pypi/xarray-custom/)
[![Python](https://img.shields.io/pypi/pyversions/xarray-custom.svg?label=Python&color=yellow&style=flat-square)](https://pypi.org/pypi/xarray-custom/)
[![Test](https://img.shields.io/github/workflow/status/astropenguin/xarray-custom/Test?logo=github&label=Test&style=flat-square)](https://github.com/astropenguin/xarray-custom/actions)
[![License](https://img.shields.io/badge/license-MIT-blue.svg?label=License&style=flat-square)](LICENSE)
[![DOI](https://img.shields.io/badge/DOI-10.5281/zenodo.3971531-blue?style=flat-square)](https://doi.org/10.5281/zenodo.3971531)

:zap: Data classes for custom xarray creation

## TL;DR

xarray-custom is a third-party Python package which helps to create custom DataArray classes in the same manner as [the Python's native dataclass].
Here is an introduction code of what the package provides:

```python
from xarray_custom import coord, dataarrayclass

@dataarrayclass
class Image:
    """DataArray class to represent images."""

    dims = 'x', 'y'
    dtype = float
    accessor = 'img'

    x: coord('x', int) = 0
    y: coord('y', int) = 0

    def normalize(self):
        return self / self.max()
```

The key features are:

```python
# create a custom DataArray
image = Image([[0, 1], [2, 3]], x=[0, 1], y=[0, 1])

# use a custom method via an accessor
normalized = image.img.normalize()

# create a custom DataArray filled with ones
ones = Image.ones((2, 2), x=[0, 1], y=[0, 1])
```

- Custom DataArray instances with fixed dimensions, datatype, and coordinates can easily be created.
- NumPy-like special functions like ``ones()`` are provided as class methods.
- Custom DataArray methods can be available via a custom accessor.

## Requirements

- **Python:** 3.6, 3.7, or 3.8 (tested by the author)
- **Dependencies:** See [pyproject.toml](pyproject.toml)

## Installation

```shell
$ pip install xarray-custom
```

## License

Copyright (c) 2020 Akio Taniguchi

- xarray-custom is distributed under the MIT License
- xarray-custom uses an icon of [xarray] distributed under the Apache 2.0 license

<!-- References -->
[xarray]: https://github.com/pydata/xarray
[the Python's native dataclass]: https://docs.python.org/3/library/dataclasses.html
