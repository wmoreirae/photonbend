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


# A lot of the code in here doesn't make the use of the standards and best tools available in the python language.
# Since we are dealing with a lot of arrays and calculations, it was thought that Numba (https://numba.pydata.org/)
# should be used so that we can have a reasonable execution time. That goal was achieved here at the cost of some
# python functionality.

# TODO Add super-sampling (possibly adapt code from the original camera.imaging module)
# TODO Extract rotation
# TODO Extract image management to another class because this one is already doing too much

from enum import IntEnum, auto

import numpy as np

from numba import uint8, float64, njit, prange, typeof, complex128, cfunc, int64, bool_
from numba.experimental import jitclass

FULL_CIRCLE = (np.pi * 2)


@njit
def vector_magnitude(vector):
    """Computes the magnitude of a complex number vector

    :param vector: A complex number
    :return: A floating point scalar of the vector magnitude
    """
    return np.sqrt(vector.real ** 2 + vector.imag ** 2)


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


@njit
def _calculate_rotation_matrix(pitch: float, yaw: float, roll: float):
    """Computes a rotation matrix from the three primary rotation axis
    For more info see:  https://en.wikipedia.org/wiki/Rotation_matrix
    or: https://mathworld.wolfram.com/RotationMatrix.html

    :param pitch: represents a rotation in the x axis in radians
    :param yaw: represents a rotation in the y axis in radians
    :param roll: represents a rotation in the z axis in radians
    :return: a rotation matrix
    """
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


class ImageType(IntEnum):
    FULL_FRAME = auto()
    CROPPED_CIRCLE = auto()
    INSCRIBED = auto()
    DOUBLE_INSCRIBED = auto()


@cfunc(float64(float64))
def _a(_b):
    """A SIMPLE IDENTITY FUNCTION THAT IS NOT MEANT TO BE USED!

    It is present only to make it easier to categorize functions marked with its decorator <@cfunc(float64(float64))>
    so we can have those as class members on a numba jitclass
    """
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
class SphereImage:
    """A class that maps images to a sphere
    This class allows us to get or set pixel values that are mapped to specific coordinates of that sphere. This
    enables us to convert images between different kinds of lenses or to create projections from it.

    Not only that, this class also allows us to rotate the sphere so we can get different angles, which is essential
    when dealing with 360 degree images or can greatly simplify creating traverse projection.
    """

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

        It computes the maximum distance the maximum angle this lens is set to produce in focal distances.
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

    def _translate_coordinates(self, latitude: float, longitude: float, inverse: bool = False):
        """ Translate coordinates using the instance rotation matrix

        :param latitude: The original latitude in radians as a float
        :param longitude: The original longitude in radians as a float
        :param inverse: Flag that sets the inverse transformation. This is used when converting coordinates generated
                        from the underlying image quasi-polar coordinates to sphere coordinates.
        :return: translated coordinates in radians as a 2-tuple of floats
        """
        # TODO refactor to make use of complex numbers
        if not self.rotated:
            return latitude, longitude

        y = np.sin(latitude)
        xz = np.exp(longitude * 1j)
        x = xz.real * np.cos(latitude)
        z = xz.imag * np.cos(latitude)

        position_vector = np.zeros(3, np.core.float64)
        position_vector[:] = x, y, z
        if not inverse:
            new_position_vector = self.rotation_matrix.dot(position_vector)
        else:
            new_position_vector = self.i_rotation_matrix.dot(position_vector)

        translated_latitude = np.arcsin(new_position_vector[1])
        translated_xz_magnitude = np.cos(translated_latitude)
        translated_xz = complex(new_position_vector[0] / translated_xz_magnitude,
                                new_position_vector[2] / translated_xz_magnitude)
        translated_longitude = np.log(translated_xz).imag

        return translated_latitude, translated_longitude

    def set_rotation(self, pitch: float, yaw: float, roll: float) -> None:
        """
        Sets the instance rotation matrix to the pitch, yaw, roll values passed

        If used with all values as 0 (zero), it resets the sphere to it's initial rotation.
        It also sets the inverse rotation matrix.
        :param pitch: The amount of rotation in radians along the x axis as a float
        :param yaw: The amount of rotation in radians along the y axis as a float
        :param roll: The amount of rotation in radians along the z axis as a float
        :return: None
        """
        if pitch == yaw == roll == 0.0:
            self.rotated = False
        else:
            self.rotated = True

        rot = _calculate_rotation_matrix(pitch, yaw, roll)
        i_rot = _calculate_rotation_matrix(-pitch, -yaw, -roll)
        self.rotation_matrix[:] = rot[:]
        self.i_rotation_matrix[:] = i_rot[:]

    def add_rotation(self, pitch, yaw, roll):
        """Add a new rotation to the instance's current rotation matrix.
        With this method a sphere can be rotated arbitrarily many times along its axis.
        This method does not reset the rotation. It actually updates it, adding this new rotation to the previously
        existing one.

        If used with all values as 0 (zero), it doesn't do anything
        It also updates the inverse rotation matrix.
        :param pitch: The amount of rotation in radians along the x axis as a float
        :param yaw: The amount of rotation in radians along the y axis as a float
        :param roll: The amount of rotation in radians along the z axis as a float
        :return: None
        """

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
        """ Returns a copy of the underlying image matrix
        :return: A copy of the image array
        """
        return np.copy(self.image)

    @property
    def shape(self):
        return self.image.shape

    def check_position(self, x, y):
        """
        Checks whether the passed parameters are withing the limits of the underlying image
        :param x: absolute x position of the image you want to check
        :param y: absolute y position of the image you want to check
        :return: True if withing the image, False otherwise
        """
        height, width = self.image.shape[:2]
        if (0 > x or x >= width) or (0 > y or y >= height):
            return False
        return True

    def get_flat_position(self, x, y):
        if self.check_position(x, y):
            return self.image[y, x, :]
        else:
            return 0, 0, 0

    def get_from_coordinates(self, latitude: float, longitude: float) -> uint8[:]:
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
        x, y = self._get_image_position_from_image_quasipolar_coordinates(t_latitude, t_longitude)

        return x, y

    def _get_image_position_from_image_quasipolar_coordinates(self, t_latitude, t_longitude):
        """
        No translation happens here. The coordinates are used almost as polar coordinates on the image!

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


@njit(parallel=True)
def _helper_map_from_sphere_image(this_image: SphereImage, that_image: SphereImage):
    height, width = this_image.image.shape[:2]
    h_fov = this_image.fov / 2

    for x in prange(width):
        for y in prange(height):
            try:
                lat, lon = this_image.get_coordinates_from_image_position(x, y)
            except Exception:  # Because of Numba's njit Exception limitation, that's all we currently use
                continue

            # if lat < (np.pi / 2 - h_fov):
            #     continue
            # if lat > (np.pi / 2):
            #     continue
            pixel_values = np.zeros((1, 1, 3), np.core.uint8)
            pixel_values[0, 0, :] = that_image.get_from_coordinates(lat, lon)
            this_image.image[y, x, :] = pixel_values[0, 0, :]
    return
