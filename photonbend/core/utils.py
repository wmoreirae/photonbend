import numpy as np
import numpy.typing as npt
from typing import Callable, Tuple


# New non-numba version
def make_complex(x: npt.NDArray[np.core.float64], y: npt.NDArray[np.core.float64], sparse: bool = True):
    """Converts 2 numpy arrays of floats into a single array of complex.

    of the same size containing np.float64


    """
    zx = x * 0
    zy = y * 0
    fx = x + zy
    fy = y + zx
    ans = np.concatenate([fx.reshape(*fx.shape, 1), fy.reshape(*fy.shape, 1)], 2).view(
        dtype=np.core.complex128).reshape(fx.shape[:2])
    return ans
