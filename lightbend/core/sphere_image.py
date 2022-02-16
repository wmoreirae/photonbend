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


# A lot of the code in here doesn't make the use of the standards and best tools available in the Python language.
# Since we are dealing with a lot of arrays and calculations, it was thought that Numba (https://numba.pydata.org/)
# should be used so that we can have a reasonable execution time. This goal has benn achieved here at the cost of some
# python functionality.

# TODO Add super-sampling (possibly adapt code from the original camera.imaging module)
# TODO Improve on the use of exceptions
from typing import Tuple

import numpy as np

from numba import uint8, float64, njit, prange,  complex128, bool_, typeof
from numba.experimental import jitclass
from numba.experimental.jitclass.base import JitClassType

from lightbend.core import LensImage

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


spec = [
    ('north_pole', complex128),
    ('south_pole', complex128),
    ('fov', float64),
    ('dpf', float64),
    ('rotated', bool_),
    ('rotation_matrix', float64[:, :]),
    ('i_rotation_matrix', float64[:, :]),
]


@jitclass(spec)
class SphereImage:
    lens_image: LensImage
    """A class that maps images to a sphere
    This class allows us to get or set pixel values that are mapped to specific coordinates of that sphere. This
    enables us to convert images between different kinds of lenses or to create projections from it.

    Not only that, this class also allows us to rotate the sphere so we can get different angles, which is essential
    when dealing with 360 degree images or can greatly simplify creating traverse projection.
    """

    def __init__(self, image_arr, image_type, fov, lens):
        """Create a new SphereImage

        :param image_arr: The array representing the image.
        :param image_type: An enum of <lightbend.core.ImageType>
        :param fov: The maximum Field of View of this image
        :param lens: The lens function of this sphere image
        """
        self.lens_image = LensImage(image_arr, image_type, fov, lens)
        self.rotated = False
        self.rotation_matrix = np.zeros((3, 3), np.core.float64)
        self.i_rotation_matrix = np.zeros((3, 3), np.core.float64)
        self.set_rotation(0.0, 0.0, 0.0)

    def _get_rotated_coordinates(self, latitude: float, longitude: float, inverse: bool = False):
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
        # TODO change method to chain rotations instead of simply adding them together.
        """Add a new rotation to the instance's current rotation matrix.
        With this method a sphere can be rotated arbitrarily many times along its axis.
        This method does not reset the rotation. It actually updates it, adding this new rotation to the previously
        existing one.
        The problem is it won't add consecutively.

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

    def get_image_array(self):
        """ Returns a copy of the underlying image matrix
        :return: A copy of the image array
        """
        return np.copy(self.lens_image.image)

    @property
    def shape(self):
        return self.lens_image.shape

    def check_position(self, x, y):
        """
        Checks whether the passed parameters are withing the limits of the underlying image
        :param x: absolute x position of the image you want to check
        :param y: absolute y position of the image you want to check
        :return: True if withing the image, False otherwise
        """
        height, width = self.lens_image.shape[:2]
        if (0 > x or x >= width) or (0 > y or y >= height):
            return False
        return True

    def get_value_from_cartesian(self, x, y):
        """ Get the value represented on the cartesian position x, y of this image

        :param x: The horizontal position of the desired value
        :param y: The vertical position of the desired value
        :return: a triplet of values of type uint8
        """
        if self.check_position(x, y):
            return self.lens_image.image[y, x, :]
        else:
            return 0, 0, 0

    def get_value_from_spherical(self, latitude: float, longitude: float) -> uint8[:]:
        """Returns the value at the given latitude and longitude.

        If the request point is invalid, it returns the triplet (0, 0 ,0)
        :param latitude: The latitude of the requested value
        :param longitude: The longitude of the requested value
        :return: a triplet of values of type uint8
        """

        x, y = self.translate_spherical_to_cartesian(latitude, longitude)

        if self.check_position(x, y):
            return self.lens_image.image[y, x, :]
        return np.zeros(3, np.core.uint8)

    def set_value_to_spherical(self, latitude: float, longitude: float, data: uint8[:]) -> None:
        """Sets the value at the given latitude and longitude to the value present on data

        :param latitude: The latitude of the value to set
        :param longitude: The longitude of the value to set
        :param data: The value that should be set
        :return: None
        """

        x, y = self.translate_spherical_to_cartesian(latitude, longitude)
        if self.check_position(x, y):
            self.lens_image.image[y, x, :] = data

    def translate_spherical_to_cartesian(self, latitude:float, longitude: float) -> Tuple[int, int]:
        """Translate spherical coordinates to cartesian coordinates of the image

        :param latitude: The latitude of the location to be translated
        :param longitude: The longitude of the location to be translated
        :return: The cartesian coordinates of the image represented by the spherical coordinates passed
        """
        assert (-np.pi / 2) <= latitude <= (np.pi / 2), "latitude should be between pi/2 and -pi/2"

        r_latitude, r_longitude = self._get_rotated_coordinates(latitude, longitude, True)
        x, y = self.lens_image.translate_to_cartesian(r_latitude, r_longitude)

        return x, y

    def translate_cartesian_to_spherical(self, x, y):
        """Translate cartesian coordinates to spherical coordinates of the image

        :param x: The horizontal coordinate
        :param y: The vertical coordinate
        :return: The spherical coordinates representing that location
        """
        latitude, longitude = self.lens_image.translate_to_spherical(x, y)
        r_latitude, r_longitude = self._get_rotated_coordinates(latitude, longitude)

        return r_latitude, r_longitude

    def map_from_sphere_image(self, target_image, super_sampler: int) -> None:
        """ Maps one spherical image to another, enabling conversion between different lenses, FoVs and target.

        The current sphere image will receive the data from the target sphere image. However it won't just copy it.
        The information will be translated to this sphere using their angular data and other specs, like its lens and
        FoV, producing a new image.

        :param target_image: The sphere image that will serve as the source of the information.
        :param super_sampler: The super sampling factor, which enables improving the image at the cost of processing.
            WARNING: The super sampling factor is actually the square of this parameter.
            e.g.: When using a factor of 3, it will actually compute 9 times more. For 4 it is 16 and 5 makes it 25.
            The super sampler is implemented in a way that uses little memory space, so it doesn't hinder it's use
            on low-spec machines. It won't increase much the memory footprint.
        :return: None
        """
        _helper_map_from_sphere_image(self, target_image, super_sampler)


@njit(parallel=True)
def _helper_map_from_sphere_image(this_image: SphereImage, that_image: SphereImage, super_sampler: int):
    height, width = this_image.lens_image.shape[:2]

    # SUPER SAMPLING DATA
    ss_matrix_center_pixel = complex((super_sampler - 1) / 2, (super_sampler - 1) / 2)
    ss_subpixel_distance = 1 / super_sampler

    for column in prange(width):
        for row in prange(height):
            if super_sampler == 1:
                # No super sampling case
                try:
                    lat, lon = this_image.translate_cartesian_to_spherical(column, row)
                except Exception:  # Because of Numba's njit Exception limitation, that's all we currently use
                    continue
                this_image.lens_image.image[row, column, :] = that_image.get_value_from_spherical(lat, lon)

            else:
                # super sampling
                super_sample_matrix = np.zeros((super_sampler, super_sampler, 3), np.core.uint8)
                for ss_row in prange(super_sampler):
                    for ss_column in prange(super_sampler):
                        ss_position = (complex(ss_column, ss_row) - ss_matrix_center_pixel) * ss_subpixel_distance

                        position_destiny = complex(column, row) + ss_position
                        a_column = position_destiny.real
                        a_row = position_destiny.imag
                        try:
                            lat, lon = this_image.translate_cartesian_to_spherical(a_column, a_row)
                        except Exception:  # Because of Numba's njit Exception limitation, that's all we currently use
                            continue

                        super_sample_matrix[ss_row, ss_column, :] = that_image.get_value_from_spherical(lat, lon)

                # Calculate the pixel mean and assign it!
                ss_mean_1st = np.mean(super_sample_matrix[:, :, 0])
                ss_mean_2nd = np.mean(super_sample_matrix[:, :, 1])
                ss_mean_3rd = np.mean(super_sample_matrix[:, :, 2])
                this_image.lens_image.image[row, column, 0] = int(np.round(ss_mean_1st))
                this_image.lens_image.image[row, column, 1] = int(np.round(ss_mean_2nd))
                this_image.lens_image.image[row, column, 2] = int(np.round(ss_mean_3rd))

    return
