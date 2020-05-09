# dependencies
import numpy as np
from xarray_custom import coordtype, dataarrayclass


# test functions
def test_coordtype():
    dims = ("x",)
    dtype = int

    CoordType = coordtype(dims, dtype)
    coord = CoordType([1.0, 2.0])

    assert coord.dims == dims
    assert coord.dtype == dtype


def test_dataarrayclass():
    dims = ("x", "y")
    dtype = float
    accessor = "custom"

    @dataarrayclass(dims, dtype, accessor)
    class CustomDataArray:
        x: coordtype(dims[0], int)
        y: coordtype(dims[1], int)
        z: coordtype(dims, str) = "a"

        def double(self):
            return self * 2

    shape = 2, 2
    data = np.arange(4).reshape(shape)
    default_z = np.full(shape, "a")

    dataarray = CustomDataArray(data, x=[0, 1], y=[0, 1])

    assert dataarray.dims == dims
    assert dataarray.dtype == dtype
    assert (dataarray.custom.double() == data * 2).all()
    assert (dataarray.z == default_z).all()
