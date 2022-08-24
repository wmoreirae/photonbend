#   Copyright (c) 2022. Edson Moreira
#  #
#   Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
#   documentation files (the "Software"), to deal in the Software without restriction, including without limitation
#    the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and
#    to permit persons to whom the Software is furnished to do so, subject to the following conditions:
#  #
#   The above copyright notice and this permission notice shall be included in all copies or substantial portions of
#   the Software.
#  #
#   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING
#   BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
#   NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
#   DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
#   OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

import numpy as np
import numpy.typing as npt

from photonbend.core.utils import make_complex


def _calculate_rotation_matrix(pitch: float, yaw: float, roll: float):
    """Computes a rotation matrix from the three primary rotation axis
    For more info see:  https://en.wikipedia.org/wiki/Rotation_matrix
    or: https://mathworld.wolfram.com/RotationMatrix.html

    :param pitch: represents a rotation in the x axis in radians
    :param yaw: represents a rotation in the y axis in radians
    :param roll: represents a rotation in the z axis in radians
    :return: a rotation matrix
    """

    # TODO unwind the matrices and wind them up with array methods so they are contiguous
    cos_pitch = np.cos(pitch)
    sin_pitch = np.sin(pitch)
    pitch_matrix = np.array((1, 0, 0, 0, cos_pitch, sin_pitch, 0, -sin_pitch, cos_pitch)).reshape((3, 3))

    cos_yaw = np.cos(yaw)
    sin_yaw = np.sin(yaw)
    yaw_matrix = np.array((cos_yaw, 0, -sin_yaw, 0, 1, 0, sin_yaw, 0, cos_yaw)).reshape((3, 3))

    cos_roll = np.cos(roll)
    sin_roll = np.sin(roll)
    roll_matrix = np.array((cos_roll, sin_roll, 0, -sin_roll, cos_roll, 0, 0, 0, 1)).reshape((3, 3))

    rotation_matrix = pitch_matrix @ yaw_matrix @ roll_matrix

    return rotation_matrix


class Rotation:
    def __init__(self, pitch: float, yaw: float, roll: float):
        self.rotation_matrix = _calculate_rotation_matrix(pitch, yaw, roll)

    def process_coordinate_map(self, coordinate_map: npt.NDArray[np.core.float64]) -> npt.NDArray[np.core.int8]:
        invalid_map = coordinate_map[:, :, 2] != 0.0
        polar_map = coordinate_map[:, :, :2]
        polar_map[invalid_map] = 0

        latitude = polar_map[:, :, 0]
        longitude = polar_map[:, :, 1]

        y = np.sin(latitude)
        xz = np.exp(longitude * 1j)
        x = xz.real * np.cos(latitude)
        z = xz.imag * np.cos(latitude)

        x = np.expand_dims(x, axis=2)
        y = np.expand_dims(y, axis=2)
        z = np.expand_dims(z, axis=2)

        position_vector: npt.NDArray[np.core.float64] = np.concatenate([x, y, z], axis=2)
        print(position_vector.shape)
        new_position_vector = np.apply_along_axis(self.rotation_matrix.dot, axis=2, arr=position_vector)
        print(new_position_vector.shape)

        translated_latitude = np.arcsin(new_position_vector[:, :, 1])
        translated_xz_magnitude = np.cos(translated_latitude)
        translated_xz = make_complex(new_position_vector[:, :, 0] / translated_xz_magnitude,
                                     new_position_vector[:, :, 2] / translated_xz_magnitude)
        translated_longitude = np.log(translated_xz).imag

        translated_latitude = np.expand_dims(translated_latitude, axis=2)
        translated_longitude = np.expand_dims(translated_longitude, axis=2)
        new_invalid_map = np.expand_dims(invalid_map, axis=2)

        ans = np.concatenate([translated_latitude, translated_longitude, new_invalid_map], axis=2)
        ans[invalid_map] = 0
        return ans
