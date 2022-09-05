import numpy as np
import numpy.typing as npt


def make_complex(x: npt.NDArray[np.core.float64], y: npt.NDArray[np.core.float64], sparse: bool = True):
    """Converts 2 arrays of float64 into a single array of complex128.

    Converts two arrays containing float64 to a single array of
    complex128. The arrays must have to either have the same dimensions or be sparse
    arrays.

    Args:
        x (np.ndarray[float64]): An array whose elements are going to be
            the real part of the numbers.
        y (np.ndarray[float64]): An array whose elements are going to be
            the imaginary part of the numbers.
        sparse (bool): Optional component describing if the algorithm
            should handle the sparse case. Default is True.
    Returns:
        A numpy ndarray containing complex128.
    """

    zx = x * 0
    zy = y * 0
    fx = x + zy
    fy = y + zx
    ans = np.concatenate([fx.reshape(*fx.shape, 1), fy.reshape(*fy.shape, 1)], 2).view(
        dtype=np.core.complex128).reshape(fx.shape[:2])
    return ans
