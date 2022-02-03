#  Copyright (c) 2022. Edson Moreira
#
#  Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
#  documentation files (the "Software"), to deal in the Software without restriction, including without limitation
#   the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and
#   to permit persons to whom the Software is furnished to do so, subject to the following conditions:
#
#  The above copyright notice and this permission notice shall be included in all copies or substantial portions of
#  the Software.
#
#  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING
#  BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
#  NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
#  DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
#  OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

from enum import IntEnum, auto

import numpy as np

from numba import uint8, float64, njit, prange, typeof, complex128, cfunc, int64, bool_
from numba.experimental import jitclass
from lightbend.utils import degrees_to_radians, radians_to_degrees
from lightbend.exceptions.coordinates_out_of_image import CoordinatesOutOfImage

FULL_CIRCLE = (np.pi * 2)


# TODO Add super-sampling (possibly adapt code from the original camera.imaging module)
# TODO Extract rotation
# TODO Extract image management to another class because this one is already doing too much


@njit
def vector_magnitude(vector):
    return np.sqrt(vector.real ** 2 + vector.imag ** 2)


@njit
def decompose(a_complex_number):
    x = int(np.round(a_complex_number.real))
    y = int(np.round(a_complex_number.imag))
    return x, y


@njit
def _calculate_rotation_matrix(pitch, yaw, roll):
    cos_pitch = np.cos(pitch)
    sin_pitch = np.sin(pitch)
    pitch_matrix = np.array(((1, 0, 0),
                             (0, cos_pitch, -sin_pitch),
                             (0, sin_pitch, cos_pitch))).T

    cos_yaw = np.cos(yaw)
    sin_yaw = np.sin(yaw)
    yaw_matrix = np.array(((cos_yaw, 0, sin_yaw),
                           (0, 1, 0),
                           (-sin_yaw, 0, cos_yaw))).T

    cos_roll = np.cos(roll)
    sin_roll = np.sin(roll)
    roll_matrix = np.array(((cos_roll, -sin_roll, 0),
                            (sin_roll, cos_roll, 0),
                            (0, 0, 1))).T

    rotation_matrix = pitch_matrix @ yaw_matrix @ roll_matrix

    return rotation_matrix


@njit(parallel=True)
def _helper_map_from_sphere_image(this_image, that_image):
    height, width = this_image.image.shape[:2]
    h_fov = this_image.fov / 2

    for x in prange(width):
        for y in prange(height):
            try:
                lat, lon = this_image.get_coordinates_from_image_position(x, y)
            except Exception:  # njit can only catch the basic exception and can't store so just using exception
                continue

            # if lat < (np.pi / 2 - h_fov):
            #     continue
            # if lat > (np.pi / 2):
            #     continue
            pixel_values = np.zeros((1, 1, 3), np.core.uint8)
            pixel_values[0, 0, :] = that_image.get_from_coordinates(lat, lon)
            this_image.image[y, x, :] = pixel_values[0, 0, :]
    return


class ImageType(IntEnum):
    FULL_FRAME = auto()
    CROPPED_CIRCLE = auto()
    INSCRIBED = auto()
    DOUBLE_INSCRIBED = auto()


@cfunc(float64(float64))
def _a(_b):
    return _b


spec = [
    ('center', complex128),
    ('image_type', int64),
    ('fov', float64),
    ('dpf', float64),
    ('image', uint8[:, :, :]),
    ('rotated', bool_),
    ('rotation_matrix', float64[:, :]),
    ('i_rotation_matrix', float64[:, :]),
    ('lens', typeof(_a)),
    ('i_lens', typeof(_a)),
]


@njit
def _get_360_longitude(longitude):
    new_longitude = longitude % FULL_CIRCLE
    return new_longitude


@njit
def _get_180_longitude(longitude):
    new_longitude = longitude % FULL_CIRCLE
    if np.pi < new_longitude:
        new_longitude -= (2 * np.pi)
    return new_longitude


@jitclass(spec)
class SphereImageInscribed:

    def __init__(self, image_arr, image_type, fov, lens, i_lens):
        self.image = image_arr
        self.image_type = image_type
        self.lens = lens
        self.i_lens = i_lens
        self.fov = fov
        self.rotated = False
        self.rotation_matrix = np.zeros((3, 3), np.core.float64)
        self.i_rotation_matrix = np.zeros((3, 3), np.core.float64)
        self.set_rotation(0.0, 0.0, 0.0)
        self._set_center()
        self._set_dpf()

    def _set_center(self):
        height, width = self.image.shape[:2]
        self.center = np.complex(width, height)
        self.center = self.center / 2
        self.center = self.center - complex(0.5, 0.5)

    def _set_dpf(self):
        """
        This function sets the dpf (dots per focal distance)

        Only the self.init method should call this.

        It computes the maximum distance the maximum angle this lens is set to produce if focal distances.
        To simplify the calculations, we always use a focal distance of one, and make the dots per focal distance (dpf)
        variable.
        So, in order to calculate the dpf, we measure the maximum distance the lens produce if focal distances,
        we calculate the longest vector of this image (from the center of the image to one of it's sides) and we
        divide the second by the first to arrive at the dpf.  Then we set this to the current object.

        :return: None
        """
        maximum_lens_angle = self.fov / 2
        maximum_image_magnitude = self._get_maximum_image_magnitude()
        lens_max_angle_magnitude = self.lens(maximum_lens_angle)
        self.dpf = maximum_image_magnitude / lens_max_angle_magnitude

    def _translate_coordinates(self, latitude, longitude, inverse=False):
        if not self.rotated:
            return latitude, longitude

        y = np.sin(latitude)
        xz = np.exp(longitude * 1j)
        x = xz.real * np.cos(latitude)
        z = xz.imag * np.cos(latitude)

        # print('x: ', x)
        # print('y: ', y)
        # print('z: ', z)

        position_vector = np.zeros(3, np.core.float64)
        position_vector[:] = x, y, z
        if not inverse:
            new_position_vector = self.rotation_matrix.dot(position_vector)
        else:
            new_position_vector = self.i_rotation_matrix.dot(position_vector)

        # r_x, r_y, r_z = new_position_vector
        # print('r_x: ', r_x)
        # print('r_y: ', r_y)
        # print('r_z: ', r_z)

        r_latitude = np.arcsin(new_position_vector[1])
        r_xz_mag = np.cos(r_latitude)
        r_xz = complex(new_position_vector[0] / r_xz_mag, new_position_vector[2] / r_xz_mag)
        r_longitude = np.log(r_xz).imag

        # print('latitude:', radians_to_degrees(r_latitude))
        # print('longitude:', radians_to_degrees(r_longitude))

        return r_latitude, r_longitude

    def set_rotation(self, pitch, yaw, roll):
        if pitch == yaw == roll == 0.0:
            self.rotated = False
        else:
            self.rotated = True

        rot = _calculate_rotation_matrix(pitch, yaw, roll)
        i_rot = _calculate_rotation_matrix(-pitch, -yaw, -roll)
        self.rotation_matrix[:] = rot[:]
        self.i_rotation_matrix[:] = i_rot[:]

    def add_rotation(self, pitch, yaw, roll):
        self.rotated = True
        rot = _calculate_rotation_matrix(pitch, yaw, roll)
        i_rot = _calculate_rotation_matrix(-pitch, -yaw, -roll)
        self.rotation_matrix[:] = self.rotation_matrix @ rot
        self.i_rotation_matrix[:] = self.i_rotation_matrix @ i_rot

    def _get_maximum_image_magnitude(self):
        if self.image_type == ImageType.FULL_FRAME:
            return vector_magnitude(self.center)
        elif self.image_type == ImageType.CROPPED_CIRCLE:
            return self.center.real
        elif self.image_type == ImageType.INSCRIBED:
            return self.center.real
        elif self.image_type == ImageType.DOUBLE_INSCRIBED:
            # TODO
            raise NotImplementedError("The functionality for double inscribed images has not been finished")

    def get_image_array(self):
        """
        :return: A copy of the image array
        """
        return np.copy(self.image)

    @property
    def shape(self):
        return self.image.shape

    def check_position(self, x, y):
        height, width = self.image.shape[:2]
        if (0 > x or x >= width) or (0 > y or y >= height):
            return False
        return True

    def get_flat_position(self, x, y):
        if self.check_position(x, y):
            return self.image[y, x, :]
        else:
            return 0, 0, 0

    def get_from_coordinates(self, latitude, longitude) -> uint8[:]:
        _3_longitude = _get_360_longitude(longitude)
        x, y = self._get_image_position_from_coordinates(latitude, _3_longitude)
        # print('get_from: ', x, y)
        if self.check_position(x, y):
            return self.image[y, x, :]
        return np.zeros(3, np.core.uint8)

    def set_to_coordinates(self, latitude, longitude, data):
        _3_longitude = _get_360_longitude(longitude)
        x, y = self._get_image_position_from_coordinates(latitude, _3_longitude)
        if self.check_position(x, y):
            self.image[y, x, :] = data

    def _get_image_position_from_coordinates(self, latitude, longitude):
        assert (-np.pi / 2) <= latitude <= (np.pi / 2), "latitude should be between pi/2 and -pi/2"

        t_latitude, t_longitude = self._translate_coordinates(latitude, longitude)
        x, y = self._get_image_position_from_image_polar_coordinates(t_latitude, t_longitude)

        return x, y

    def _get_image_position_from_image_polar_coordinates(self, t_latitude, t_longitude):
        """
        No translation happens here. The coordinates are used as polar coordinates on the image!!!

        :param t_latitude:
        :param t_longitude:
        :return:
        """
        center_distance = self.lens(np.pi / 2 - t_latitude) * self.dpf
        factors = np.exp(t_longitude * 1j)
        relative_position = factors * center_distance
        position = self.relative_to_absolute(relative_position)
        x, y = decompose(position)
        return x, y

    def get_coordinates_from_image_position(self, x, y):
        # TODO fix this function so it can handle many kinds of images
        max_latitude = np.pi / 2
        h_fov = self.fov / 2
        min_latitude = max_latitude - h_fov

        absolute_position = complex(x, y)
        relative_position = self.absolute_to_relative(absolute_position)
        magnitude = vector_magnitude(relative_position)
        if magnitude > self._get_maximum_image_magnitude():
            raise Exception('not a valid position')

        latitude = max_latitude - self.i_lens(magnitude / self.dpf)
        normalized_position = relative_position / magnitude
        longitude = np.log(normalized_position).imag

        longitude = _get_180_longitude(longitude)
        return self._translate_coordinates(latitude, longitude, True)

    def map_from_sphere_image(self, sphere_image):
        _helper_map_from_sphere_image(self, sphere_image)

    def relative_to_absolute(self, relative_position):
        return (relative_position.real + self.center.real) + 1j * (self.center.imag - relative_position.imag)

    def absolute_to_relative(self, absolute_position):
        return (absolute_position.real - self.center.real) + 1j * (self.center.imag - absolute_position.imag)
