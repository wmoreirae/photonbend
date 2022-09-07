#   Copyright (c) 2022. Edson Moreira
#
#   Permission is hereby granted, free of charge, to any person obtaining a copy
#   of this software and associated documentation files (the "Software"), to deal
#   in the Software without restriction, including without limitation the rights
#   to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
#   copies of the Software, and to permit persons to whom the Software is
#   furnished to do so, subject to the following conditions:
#
#   The above copyright notice and this permission notice shall be included in
#   all copies or substantial portions of the Software.
#
#   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#   IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#   FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
#   AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#   LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
#   OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
#   SOFTWARE.

import numpy as np
import numpy.typing as npt


def make_complex(
    x: npt.NDArray[np.float64],
    y: npt.NDArray[np.float64],
    sparse: bool = True,
) -> npt.NDArray[np.complex128]:
    """Converts 2 arrays of float64 into a single array of complex128.

    Converts two arrays containing float64 to a single array of complex128.
    The arrays must have to either have the same dimensions or be sparse arrays.

    Args:
        x (np.ndarray[float64]): An array whose elements are going to be the
            real part of the numbers.
        y (np.ndarray[float64]): An array whose elements are going to be the
        imaginary part of the numbers.
        sparse (bool): Optional component describing if the algorithm should
            handle the sparse case. Default is True.
    Returns:
        A numpy ndarray containing complex128.
    """

    zx = x * 0
    zy = y * 0
    fx = x + zy
    fy = y + zx
    ans: npt.NDArray[np.complex128] = (
        np.concatenate([fx.reshape(*fx.shape, 1), fy.reshape(*fy.shape, 1)], 2)
        .view(dtype=np.complex128)
        .reshape(fx.shape[:2])
    )
    return ans
