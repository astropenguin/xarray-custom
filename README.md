# xarray-custom

[![PyPI](https://img.shields.io/pypi/v/xarray-custom.svg?label=PyPI&style=flat-square)](https://pypi.org/pypi/xarray-custom/)
[![Python](https://img.shields.io/pypi/pyversions/xarray-custom.svg?label=Python&color=yellow&style=flat-square)](https://pypi.org/pypi/xarray-custom/)
[![Test](https://img.shields.io/github/workflow/status/astropenguin/xarray-custom/Test?logo=github&label=Test&style=flat-square)](https://github.com/astropenguin/xarray-custom/actions)
[![License](https://img.shields.io/badge/license-MIT-blue.svg?label=License&style=flat-square)](LICENSE)

:zap: Data classes for custom xarray creation

## TL;DR

xarray-custom is a third-party Python package which helps to create custom DataArray classes in the same manner as [the Python's native dataclass].
Here is an introduction code of what the package provides:

```python
from xarray_custom import ctype, dataarrayclass

@dataarrayclass(accessor='img')
class Image:
    """DataArray class to represent images."""

    dims = 'x', 'y'
    dtype = float
    x: ctype('x', int) = 0
    y: ctype('y', int) = 0

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

<!-- References -->
[the Python's native dataclass]: https://docs.python.org/3/library/dataclasses.html
