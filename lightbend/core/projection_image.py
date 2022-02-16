#  Copyright (c) 2022. Edson Moreira
#
#  Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
#  documentation files (the "Software"), to deal in the Software without restriction, including without limitation
#  the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and
#  to permit persons to whom the Software is furnished to do so, subject to the following conditions:
#
#  The above copyright notice and this permission notice shall be included in all copies or substantial portions of
#  the Software.
#
#  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING
#  BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
#  NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
#  DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
#  OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.


from typing import Tuple, Callable

import numpy as np

from numba import uint8, float64, njit, typeof, complex128, cfunc, int64, bool_
from numba.experimental import jitclass

FULL_CIRCLE = (np.pi * 2)


@cfunc(complex128(float64, complex128, bool_))
def _projection_mapping(_b, _c):
    """A SIMPLE IDENTITY FUNCTION THAT IS NOT MEANT TO BE USED!

    It is present only to make it easier to categorize functions marked with its decorator
    <@cfunc(float64(float64, bool_))> so we can have those as class members on a numba jitclass
    """
    return _b


spec = [
    ('radius', float),
    ('width', complex128),
    ('height', complex128),
    ('center', complex),
    ('virtual_width', float64),
    ('virtual_height', float64),
    ('cardinal_window_start', complex),
    ('cardinal_window_end', complex),
    ('spherical_window_start', complex),
    ('spherical_window_end', complex),
    ('image', uint8[:, :, :]),
    ('mapping', typeof(_projection_mapping)),
]


@jitclass(spec)
class MapProjection:
    """A class that represents a map projection allowing us to get geodesic coordinates for a given pixel and backwards
    This class allows us to get or set pixel values that are mapped to specific coordinates of a sphere. This
    enables us to convert images between different projections or lens images from it.
    """

    def __init__(self, radius, i_type, fov, mapping: Callable[[float, complex, bool], complex]):
        self.radius = radius
        self.mapping = mapping

        self._set_poles()
        self._set_dpf()

    def _set_poles(self):
        pass

    @property
    def shape(self):
        return self.image.shape

    def is_position(self, x, y):
        """
        Checks whether the passed parameters are within the limits of the underlying image
        :param x: absolute x position of the image you want to check
        :param y: absolute y position of the image you want to check
        :return: True if withing the image, False otherwise
        """
        height, width = self.image.shape[:2]
        if (0 > x or x >= width) or (0 > y or y >= height):
            return False
        return True

    def is_valid_position(self, x, y) -> bool:
        pass

    def get_from_coordinates(self, latitude: float, longitude: float) -> uint8[:]:
        pass

    def set_to_coordinates(self, latitude, longitude, data):
        pass

    def translate_to_cartesian(self, latitude, longitude):
        pass

    def translate_to_spherical(self, x, y):
        pass

    def relative_to_absolute(self, relative_position: complex) -> complex:
        return (relative_position.real + self.center.real) + 1j * (
                self.center.imag - relative_position.imag)

    def absolute_to_relative(self, absolute_position: complex) -> Tuple[complex, complex]:
        return (absolute_position.real - self.center.real) + 1j * (self.center.imag - absolute_position.imag)


    def _calculate_width(self, radius):
        max_longitude = np.pi
        x = np.abs(self.mapping(radius, complex(max_longitude, 0), False))
        return 2 * x

    def _calculate_height(self, radius):
        max_latitude = np.pi/2
        y = np.abs(self.mapping(radius, complex(max_longitude, 0), False))

    def _calculate_dimensions(self, radius):
        x = 2 * self._calculate_width(radius)
        y = 2 * self._calculate_height(radius)

@njit(parallel=True)
def make_panoramic(source: SphereImage, desired_width):
    radius = desired_width / (np.pi * 2)

    destiny_width = int(np.round(radius * np.pi * 2))
    destiny_height = int(np.round(radius * np.log(np.tan(np.pi / 4 + degrees_to_radians(89 / 2))))) * 2
    destiny_array = np.zeros((destiny_height, destiny_width, 3), np.core.uint8)

    for row in prange(destiny_height):
        latitude = _mercator_y_latitude(radius, row, True)
        for column in prange(destiny_width):
            longitude = _mercator_x_longitude(radius, column, True)
            destiny_array[row, column, :] = source.get_value_from_spherical(latitude, longitude)

    return destiny_array