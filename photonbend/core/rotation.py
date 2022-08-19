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
    def __init__(self):
        self.rotation_matrix = np.zeros((3, 3), np.core.float64)
        self.inverse_matrix = np.zeros((3, 3), np.core.float64)

    def set_rotation(self, pitch: float, yaw: float, roll: float):
        self.rotation_matrix = _calculate_rotation_matrix(pitch, yaw, roll)
        self.inverse_matrix = _calculate_rotation_matrix(-pitch, -yaw, -roll)

    def add_rotation(self, pitch: float, yaw: float, roll: float):
        rot = _calculate_rotation_matrix(pitch, yaw, roll)
        i_rot = _calculate_rotation_matrix(-pitch, -yaw, -roll)
        self.rotation_matrix[:] = self.rotation_matrix @ rot
        self.inverse_matrix[:] = self.inverse_matrix @ i_rot

    def rotate_point(self, x: float, y: float, z: float):
        # TODO
        pass
