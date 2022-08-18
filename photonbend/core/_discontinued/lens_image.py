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
from typing import Tuple

import numpy as np

from numba import uint8, float64, typeof, complex128, cfunc, int64, bool_
from numba.experimental import jitclass

from photonbend.utils import vector_magnitude, decompose, weighted_sum
from photonbend.core.lens_image_type import LensImageType
from photonbend.utils.utils import _2ints

DoubleCardinal = Tuple[Tuple[int, int], Tuple[int, int]]
FULL_CIRCLE = (np.pi * 2)
INVALID_POSITION = (-1, -1)


@cfunc(float64(float64, bool_))
def _a_numba_function_definition(_b, _c):
    """A SIMPLE IDENTITY FUNCTION THAT IS NOT MEANT TO BE USED!

    It is present only to make it easier to categorize functions marked with its decorator
    <@cfunc(float64(float64, bool_))> so we can have those as class members on a numba jitclass
    """
    return _b


spec = [
    ('north_pole', complex128),
    ('south_pole', complex128),
    ('fov', float64),
    ('dpf', float64),
    ('image', uint8[:, :, :]),
    ('image_type', int64),
    ('lens', typeof(_a_numba_function_definition)),
]


@jitclass(spec)
class LensImage:
    """A class that maps images to a sphere
    This class allows us to get or set pixel values that are mapped to specific coordinates of that sphere. This
    enables us to convert images between different kinds of lenses or to create projections from it.
    """

    def __init__(self, image_arr, i_type, fov, lens):
        if i_type == LensImageType.DOUBLE_INSCRIBED:
            if fov < np.pi:
                raise ValueError("The FOV of a DOUBLE_INSCRIBED image should be of at minimum pi radians")

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
                self.north_pole = complex(real_width, height) / 2
                self.north_pole -= complex(0.5, 0.5)
                self.south_pole = self.north_pole + real_width
            else:
                raise ValueError("The LensImage class doesn't support vertical double inscribed images")
        else:  # Simple image
            self.north_pole = complex(width, height) / 2
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

    def get_image_array(self):
        """ Returns a copy of the underlying image matrix
        :return: A copy of the image array
        """
        return np.copy(self.image)

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
        """
        Checks whether the passed parameters are within the limits of the underlying image and the lens specification
        :param x: absolute x position of the image you want to check
        :param y: absolute y position of the image you want to check
        :return: True if withing the image, False otherwise
        """
        if not self.is_position(x, y):
            return False

        relative_position, _ = self.absolute_to_relative(complex(x, y))
        if round(vector_magnitude(relative_position)) <= self.maximum_magnitude:
            return False

        return True

    def get_from_cartesian(self, x: int, y: int) -> uint8[:]:
        """ Get the value represented on the cartesian position x, y of this image

        :param x: The horizontal position of the desired value
        :param y: The vertical position of the desired value
        :return: a triplet of values of type uint8
        """
        if self.is_position(x, y):
            return self.image[y, x, :]
        else:
            raise Exception("The requested cartesian coordinates are outside the bounds of the image!")

    def set_to_cartesian(self, x: int, y: int, value: uint8[:]) -> None:
        """ Get the value represented on the cartesian position x, y of this image

        :param value:
        :param x: The horizontal position of the desired value
        :param y: The vertical position of the desired value
        :return: a triplet of values of type uint8
        """
        if self.is_position(x, y):
            self.image[y, x, :] = value
        else:
            raise Exception("The target cartesian coordinates are outside the bounds of the image!")

    def get_from_spherical(self, latitude: float, longitude: float) -> uint8[:]:
        pos1, pos2 = self.translate_spherical_to_cartesian(latitude, longitude)
        try:
            if self.image_type != LensImageType.DOUBLE_INSCRIBED:
                return self.get_from_cartesian(*_2ints(*pos1))
            else:  # DOUBLE_INSCRIBED
                r_value = self._ds_get_from_spherical(latitude, pos1, pos2)
                return r_value
        except:
            r = np.zeros(3, np.core.uint8)
            r[:] = 0, 0, 0
            return r

    def _ds_get_from_spherical(self, latitude, pos1, pos2):
        r_value = np.zeros(3, np.core.uint8)
        double_image_latitude = (self.fov - np.pi) / 2
        if (latitude > double_image_latitude) or (latitude < (-double_image_latitude)):
            if latitude >= 0:
                r_value = self.get_from_cartesian(*_2ints(*pos1))
            else:
                r_value = self.get_from_cartesian(*_2ints(*pos2))
        else:
            cross_range = 2 * double_image_latitude
            factor_positive = cross_range - (double_image_latitude - latitude)
            factor_negative = cross_range + (-double_image_latitude - latitude)

            v_pos = self.get_from_cartesian(*_2ints(*pos1))
            v_neg = self.get_from_cartesian(*_2ints(*pos2))

            f_value = weighted_sum(v_pos, v_neg, factor_positive, factor_negative)
            r_value = f_value
        return r_value

    def set_to_spherical(self, latitude, longitude, data):
        """Sets data to the position in the image that is represented by latitude and longitude.
        If the given latitude and longitude points are not represented in the image, it raises an exception.

        :param latitude: The target latitude
        :param longitude: The target longitude
        :param data: The data that is to be put
        :return: None
        """
        try:
            if not self.image_type == LensImageType.DOUBLE_INSCRIBED:
                pos1, _ = self.translate_spherical_to_cartesian(latitude, longitude)
                if pos1 != INVALID_POSITION:
                    self.set_to_cartesian(*_2ints(*pos1), data)
            else:  # DOUBLE_INSCRIBED
                pos1, pos2 = self.translate_spherical_to_cartesian(latitude, longitude)
                if pos1 != INVALID_POSITION:
                    self.set_to_cartesian(*_2ints(*pos1), data)
                if pos2 != INVALID_POSITION:
                    self.set_to_cartesian(*_2ints(*pos2), data)
        except Exception:
            raise Exception("The given spherical coordinates are not represented on the cartesian image")

    def translate_spherical_to_cartesian(self, latitude: float, longitude: float) -> DoubleCardinal:
        """
        :param latitude:
        :param longitude:
        :return:
        """

        if self.image_type != LensImageType.DOUBLE_INSCRIBED:
            polar_distance = self.lens(np.pi / 2 - latitude, False) * self.dpf
            factors = np.exp(longitude * 1j)
            relative_position = factors * polar_distance
            position = self.relative_to_absolute(relative_position, self.north_pole)
            x, y = decompose(position)
            return (x, y), INVALID_POSITION

        else:  # DOUBLE_INSCRIBED
            pos1, pos2 = self._ds_translate_spherical_to_cartesian(latitude, longitude)
            return pos1, pos2

    def _ds_translate_spherical_to_cartesian(self, latitude, longitude):
        double_image_latitude = self.fov - np.pi
        # latitude > 0:
        if latitude > (0 - double_image_latitude):
            polar_distance = self.lens(np.pi / 2 - latitude, False) * self.dpf
            factors = np.exp(longitude * 1j)
            relative_position = factors * polar_distance
            position = self.relative_to_absolute(relative_position, self.north_pole)
            pos1 = decompose(position)
        else:
            pos1 = INVALID_POSITION
        # latitude < 0
        if latitude < (0 + double_image_latitude):
            polar_distance = self.lens(np.pi / 2 + latitude, False) * self.dpf
            factors = np.exp(longitude * 1j)
            factors = complex(-factors.real, factors.imag)  # double_inscribed X inversion
            relative_position = factors * polar_distance
            position = self.relative_to_absolute(relative_position, self.south_pole)
            pos2 = decompose(position)
        else:
            pos2 = INVALID_POSITION
        return pos1, pos2

    def translate_cartesian_to_spherical(self, x: float, y: float) -> Tuple[float, float]:
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
