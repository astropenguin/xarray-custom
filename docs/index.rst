.. xarray-custom documentation master file, created by
   sphinx-quickstart on Sat May  9 22:13:03 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

xarray-custom documentation
=============================

**Date**: |today| **Version**: |release|

.. toctree::
   :maxdepth: 2
   :caption: Contents:

xarray-custom is a Python package which helps to create custom DataArray classes in the same manner as `the Python's native dataclass <https://docs.python.org/3/library/dataclasses.html>`__.
Here is an introduction code of what the package provides::

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

The key points are:

* Custom DataArray instances with fixed dimensions and coordinates can easily be created.
* Default values and dtype can be specified via a class decorator and class variable annotations.
* NumPy-like special factory functions like ``ones()`` are provided as class methods.
* Custom DataArray methods can be used via a custom accessor.

Contents
========

* :ref:`genindex`
* :ref:`modindex`
