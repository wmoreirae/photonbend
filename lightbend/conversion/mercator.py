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

from lightbend.core import LensImage, SphereImage, ImageType
from lightbend.utils import degrees_to_radians, radians_to_degrees


@njit(parallel=True)
def make_panoramic(source: SphereImage, desired_width):
    radius = desired_width / (np.pi * 2)

    destiny_width = int(np.round(radius * np.pi * 2))
    destiny_height = int(np.round(radius * np.log(np.tan(np.pi / 4 + degrees_to_radians(89 / 2))))) * 2
    destiny_array = np.zeros((destiny_height, destiny_width, 3), np.core.uint8)

    angle_of_a_column = (np.pi * 2) / destiny_width
    destiny_center_row = (destiny_height / 2) - 0.5
    for row in prange(destiny_height):

        row_delta = destiny_center_row - row
        source_yl_theta = _get_latitude(row_delta, radius)
        latitude = source_yl_theta

        for column in prange(destiny_width):
            longitude = (angle_of_a_column * column)
            destiny_array[row, column, :] = source.get_value_from_spherical(latitude, longitude)

    return destiny_array


@njit
def _get_latitude(delta_y, radius):
    return 2 * np.arctan(np.exp(delta_y / radius)) - np.pi / 2


@njit(parallel=False)
def make_sphere_image(source: np.array, lens):
    # This implementation is a dirty dirty hack and should be modified to use the inverse of the mercator instead of
    # what it does here.
    max_magnitude = lens(np.pi / 2)
    source_height, source_width = source.shape[:2]
    radius = source_width / (2 * np.pi)
    source_center_row = source_height / 2 - 0.5

    destiny_height = int(np.round(source_width / np.pi))
    destiny_width = 2 * destiny_height
    angle_of_a_column = (2 * np.pi) / source_width

    d_sphere = SphereImage(np.zeros((destiny_height, destiny_width, 3), np.core.uint8), ImageType.DOUBLE_INSCRIBED,
                           np.pi * 2, lens)

    for column in range(source_width):
        for row in range(source_height):
            row_delta = source_center_row - row
            latitude = _get_latitude(row_delta, radius)
            longitude = (angle_of_a_column * column)
            value = source[row, column, :]
            d_sphere.set_value_to_spherical(latitude, longitude, value)

            if np.abs(latitude) < np.pi / 6:
                # This is a dirty hack to eliminate some dark spots that show because sometimes the projection
                # doesn't have detail enough to fill all the pixels on the sphere
                latitude2 = _get_latitude(row_delta - 0.5, radius)
                d_sphere.set_value_to_spherical(latitude2, longitude, value)
            # print('latitude: ', latitude)
            # print('longitude: ', longitude)
    return d_sphere


def compute_best_width(source: SphereImage) -> int:
    factor = source.lens_image.lens(np.pi / 2)
    dpf = source.lens_image.dpf
    return int(np.round(factor * dpf * 2 * np.pi))
