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

xarray-custom is a third-party Python package which helps to create custom DataArray classes in the same manner as `the Python's native dataclass <https://docs.python.org/3/library/dataclasses.html>`__.
Here is an introduction code of what the package provides::

    from xarray_custom import coordtype, dataarrayclass

    @dataarrayclass(accessor='img')
    class Image:
        dims = 'x', 'y'
        dtype = float
        x: ctype('x', int) = 0
        y: ctype('y', int) = 0

        def normalize(self):
            return self / self.max()

    # create a custom DataArray
    image = Image([[0, 1], [2, 3]], x=[0, 1], y=[0, 1])

    # use a custom method via an accessor
    normalized = image.img.normalize()

    # create a custom DataArray filled with ones
    ones = Image.ones((2, 2), x=[0, 1], y=[0, 1])

The key points are:

* Custom DataArray instances with fixed dimensions, datatype, and coordinates can easily be created.
* NumPy-like special functions like ``ones()`` are provided as class methods.
* Custom DataArray methods can be available via a custom accessor.

Contents
========

* :ref:`genindex`
* :ref:`modindex`
