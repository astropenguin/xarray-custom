# dependencies
from xarray_custom import __author__, __version__


# test functions
def test_version():
    assert __version__ == "0.6.2"


def test_author():
    assert __author__ == "Akio Taniguchi"
