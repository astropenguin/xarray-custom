# standard library
from typing import (
    Hashable,
    Mapping,
    Sequence,
    Tuple,
    Union,
)


# type hints
Dims = Union[Sequence[Hashable], Hashable]
Dtype = Union[type, str]
Shape = Union[Tuple[int], int]
Name = Hashable
Attrs = Mapping
