# dependencies
import numpy as np
from xarray_custom import coord, dataarrayclass


# constants
DIMS = "x", "y"
DTYPE = float
DATA = np.array([[0, 1], [2, 3]], DTYPE)


@dataarrayclass
class Image:
    dims = DIMS
    dtype = DTYPE
    accessor = "img"
    x: coord(DIMS[0], int) = 0
    y: coord(DIMS[1], int) = 0

    def normalize(self):
        return self / self.max()


# test functions
def test_dataarrayclass():
    data = np.array([[0, 1], [2, 3]], DTYPE)
    image = Image(data, x=[0, 1], y=[0, 1])

    assert image.dims == DIMS
    assert image.dtype == DTYPE
    assert (image == DATA).all()


def test_custom_methods():
    data = np.array([[0, 1], [2, 3]], DTYPE)
    image = Image(data, x=[0, 1], y=[0, 1])

    assert (image.img.normalize() == DATA / DATA.max()).all()


def test_special_methods():
    shape = 2, 2

    assert (Image.zeros(shape) == np.zeros(shape)).all()
    assert (Image.ones(shape) == np.ones(shape)).all()
    assert (Image.full(shape, 1) == np.full(shape, 1)).all()


def test_inheritance():
    class WeightedImage(Image):
        accessor = "wimg"
        w: coord(DIMS, float) = 1.0

    image = WeightedImage(DATA)
    print(vars(WeightedImage))

    assert (image.w == np.ones_like(DATA)).all()
    assert (image.wimg.normalize() == DATA / DATA.max()).all()
