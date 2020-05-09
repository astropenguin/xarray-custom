# xarray-custom

[![PyPI](https://img.shields.io/pypi/v/xarray-custom.svg?label=PyPI&style=flat-square)](https://pypi.org/pypi/xarray-custom/)
[![Python](https://img.shields.io/pypi/pyversions/xarray-custom.svg?label=Python&color=yellow&style=flat-square)](https://pypi.org/pypi/xarray-custom/)
[![Test](https://img.shields.io/github/workflow/status/astropenguin/xarray-custom/Test?logo=github&label=Test&style=flat-square)](https://github.com/astropenguin/xarray-custom/actions)
[![License](https://img.shields.io/badge/license-MIT-blue.svg?label=License&style=flat-square)](LICENSE)

:zap: Data classes for custom xarray constructors

## TL;DR

xarray-custom is a Python package which helps to create custom DataArray classes in the same manner as [the Python's native dataclass](https://docs.python.org/3/library/dataclasses.html).
Here is an introduction code of what the package provides:

```python
from xarray_custom import coordtype, dataarrayclass


@dataarrayclass(('x', 'y'), float, 'custom')
class CustomDataArray:
    x: coordtype('x', int)
    y: coordtype('y', int)
    z: coordtype(('x', 'y'), str) = 'spam'

    def double(self):
        """Custom DataArray method which doubles values."""
        return self * 2


dataarray = CustomDataArray([[0, 1], [2, 3]], x=[2, 2], y=[3, 3])
onesarray = CustomDataArray.ones(shape=(3, 3))
doubled = dataarray.custom.double()
```

The key points are:

- Custom DataArray instances with fixed dimensions and coordinates can easily be created.
- Default values and dtype can be specified via a class decorator and class variable annotations.
- NumPy-like special factory functions like ``ones()`` are provided as class methods.
- Custom DataArray methods can be used via a custom accessor.
