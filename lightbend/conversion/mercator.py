#  Copyright (c) 2022. Edson Moreira
#
#  Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
#  documentation files (the "Software"), to deal in the Software without restriction, including without limitation
#  the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and
#  to permit persons to whom the Software is furnished to do so, subject to the following conditions:
#
#  The above copyright notice and this permission notice shall be included in all copies or substantial portions
#  of the Software.
#
#  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING
#  BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
#  NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
#  DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
#  OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

import numpy as np
from numba import njit, prange

from lightbend.core import SphereImage
from lightbend.utils import degrees_to_radians, radians_to_degrees


@njit(parallel=True)
def make_panoramic(source: SphereImage, desired_width):
    radius = desired_width / (np.pi * 2)

    destiny_width = int(np.round(radius * np.pi * 2))
    destiny_height = int(np.round(radius * np.log(np.tan(np.pi / 4 + degrees_to_radians(89 / 2))))) * 2
    destiny_center_row = (destiny_height / 2) - 0.5

    angle_of_a_column = (np.pi * 2) / destiny_width

    destiny_array = np.zeros((destiny_height, destiny_width, 3), np.core.uint8)
    p = False
    for row in prange(destiny_height):

        row_delta = destiny_center_row - row
        source_yl_theta = 2 * np.arctan(np.exp(row_delta / radius)) - np.pi / 2
        latitude = source_yl_theta

        for column in prange(destiny_width):

            longitude = (angle_of_a_column * column)
            destiny_array[row, column, :] = source.get_value_from_spherical(latitude, longitude)

    return destiny_array


def compute_best_width(source: SphereImage) -> int:
    return int(np.round(source.lens_image.dpf * 2 * np.pi))


