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


# TODO Improve on the use of exceptions
from enum import IntEnum, auto, Enum
from typing import Tuple

import numpy as np

from numba import uint8, float64, njit, typeof, complex128, cfunc, int64, bool_
from numba.experimental import jitclass

from lightbend.utils import vector_magnitude
from lightbend.core.image_type import LensImageType

FULL_CIRCLE = (np.pi * 2)


@njit
def decompose(a_complex_number):
    """Decomposes a complex number into it's real and imaginary parts and returns them as integers.
    To turn the parts into integers, it rounds them first and the proceed s to cast them.

    :param a_complex_number: any complex number
    :return: a tuple of integers representing the real and imaginary parts of the original number
    """
    x = int(np.round(a_complex_number.real))
    y = int(np.round(a_complex_number.imag))
    return x, y


@cfunc(float64(float64, bool_))
def _a(_b, _c):
    """A SIMPLE IDENTITY FUNCTION THAT IS NOT MEANT TO BE USED!

    It is present only to make it easier to categorize functions marked with its decorator <@cfunc(float64(float64, bool_))>
    so we can have those as class members on a numba jitclass
    """
    return _b


spec = [
    ('north_pole', complex128),
    ('south_pole', complex128),
    ('fov', float64),
    ('dpf', float64),
    ('image', uint8[:, :, :]),
    ('image_type', int64),
    ('lens', typeof(_a)),
]


@jitclass(spec)
class LensImage:
    """A class that maps images to a sphere
    This class allows us to get or set pixel values that are mapped to specific coordinates of that sphere. This
    enables us to convert images between different kinds of lenses or to create projections from it.

    Not only that, this class also allows us to rotate the sphere so we can get different angles, which is essential
    when dealing with 360 degree images or can greatly simplify creating traverse projection.
    """

    def __init__(self, image_arr, i_type, fov, lens):
        self.image = image_arr
        self.image_type = i_type
        self.lens = lens
        self.fov = fov
        self._set_poles()
        self._set_dpf()

    def _set_poles(self):
        height, width = self.image.shape[:2]

        if self.image_type == LensImageType.DOUBLE_INSCRIBED:  # double image
            if width > height:  # horizontal
                real_width = width / 2
                self.north_pole = np.complex(real_width, height) / 2
                self.north_pole -= complex(0.5, 0.5)
                self.south_pole = self.north_pole + real_width
            else:
                raise ValueError("The LensImage class doesn't support vertical double inscribed images")
        else:  # Simple image
            self.north_pole = np.complex(width, height) / 2
            self.north_pole -= complex(0.5, 0.5)
            self.south_pole = complex(0, 0)

    def _set_dpf(self) -> None:
        """
        This function sets the dpf (dots per focal distance)

        Only the self.init method should call this.

        It computes the maximum distance the maximum angle this lens is set to produce in focal distances.
        To simplify the calculations, we always use a focal distance of one, and make the dots per focal distance (dpf)
        variable.
        So, in order to calculate the dpf, we measure the maximum distance the lens produce if focal distances,
        we calculate the longest vector of this image (from the center of the image to one of it's sides) and we
        divide the second by the first to arrive at the dpf.  Then we set this to the current object.

        :return: None
        """
        maximum_lens_angle = self.fov / 2

        maximum_image_magnitude = self.maximum_magnitude
        lens_max_angle_magnitude = self.lens(maximum_lens_angle, False)
        self.dpf = maximum_image_magnitude / lens_max_angle_magnitude

    @property
    def maximum_magnitude(self):
        half = 0.5
        half_c = complex(half, half)
        if self.image_type == LensImageType.FULL_FRAME:
            return vector_magnitude(self.north_pole + half_c)
        elif self.image_type == LensImageType.CROPPED_CIRCLE:
            return self.north_pole.imag + half
        elif self.image_type == LensImageType.INSCRIBED:
            return self.north_pole.imag + half
        elif self.image_type == LensImageType.DOUBLE_INSCRIBED:
            return self.north_pole.imag + half
        return self.north_pole.imag + half

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
        if not self.is_position(x, y):
            return False

        relative_position, _ = self.absolute_to_relative(complex(x, y))
        if np.round(vector_magnitude(relative_position)) <= self.maximum_magnitude:
            return False

        return True

    def get_from_spherical(self, latitude: float, longitude: float) -> uint8[:]:
        x, y = self.translate_to_cartesian(latitude, longitude)
        if self.is_position(x, y):
            return self.image[y, x, :]
        return np.zeros(3, np.core.uint8)

    def set_to_spherical(self, latitude, longitude, data):
        x, y = self.translate_to_cartesian(latitude, longitude)
        if self.is_position(x, y):
            self.image[y, x, :] = data

    def translate_to_cartesian(self, latitude, longitude):
        """
        :param latitude:
        :param longitude:
        :return:
        """
        if self.image_type == LensImageType.DOUBLE_INSCRIBED and latitude < 0:
            polar_distance = self.lens(np.pi / 2 + latitude, False) * self.dpf
            factors = np.exp(longitude * 1j)
            factors = complex(-factors.real, factors.imag)  # double X inscribed inversion
            relative_position = factors * polar_distance
            position = self.relative_to_absolute(relative_position, self.south_pole)
        else:
            polar_distance = self.lens(np.pi / 2 - latitude, False) * self.dpf
            factors = np.exp(longitude * 1j)
            relative_position = factors * polar_distance
            position = self.relative_to_absolute(relative_position, self.north_pole)
        x, y = decompose(position)
        return x, y

    def translate_to_spherical(self, x, y):
        max_latitude = np.pi / 2
        min_latitude = max_latitude - self.fov / 2
        if self.image_type == LensImageType.DOUBLE_INSCRIBED:
            min_latitude = -max_latitude

        absolute_position = complex(x, y)
        relative_position, reference_point = self.absolute_to_relative(absolute_position)

        magnitude = vector_magnitude(relative_position)
        if magnitude > self.maximum_magnitude:
            raise Exception('not a valid position')

        if reference_point == self.north_pole:
            latitude = max_latitude - self.lens(magnitude / self.dpf, True)
        else:  # reference_point == self.south_pole
            latitude = min_latitude + self.lens(magnitude / self.dpf, True)

        if self.image_type == LensImageType.DOUBLE_INSCRIBED and reference_point == self.south_pole:
            relative_position = complex(-relative_position.real, relative_position.imag)  # double inscribed inversion
        euler_relative_position = relative_position / magnitude
        longitude = np.log(euler_relative_position).imag

        return latitude, longitude

    def relative_to_absolute(self, relative_position: complex, reference_point: complex) -> complex:
        if reference_point == self.north_pole or reference_point == self.south_pole:
            return (relative_position.real + reference_point.real) + 1j * (
                    reference_point.imag - relative_position.imag)
        raise Exception("Only the image's north and south poles may be used as a reference point")

    def absolute_to_relative(self, absolute_position: complex) -> Tuple[complex, complex]:
        x = absolute_position.real

        reference_point = self.north_pole
        if self.image_type == LensImageType.DOUBLE_INSCRIBED:
            halfway_x = self.image.shape[1] / 2
            if x >= halfway_x:
                reference_point = self.south_pole

        return ((absolute_position.real - reference_point.real) + 1j * (reference_point.imag - absolute_position.imag),
                reference_point)
