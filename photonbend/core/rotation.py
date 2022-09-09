#   Copyright (c) 2022. Edson Moreira
#
#   Permission is hereby granted, free of charge, to any person obtaining a copy
#   of this software and associated documentation files (the "Software"), to deal
#   in the Software without restriction, including without limitation the rights
#   to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
#   copies of the Software, and to permit persons to whom the Software is
#   furnished to do so, subject to the following conditions:
#
#   The above copyright notice and this permission notice shall be included in
#   all copies or substantial portions of the Software.
#
#   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#   IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#   FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
#   AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#   LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
#   OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
#   SOFTWARE.

import numpy as np
import numpy.typing as npt

from photonbend.core._shared import make_complex


def _calculate_rotation_matrix(
    pitch: float, yaw: float, roll: float
) -> npt.NDArray[np.float64]:
    """Computes a rotation matrix from the three primary rotation axis
    For more info see:  https://en.wikipedia.org/wiki/Rotation_matrix
    or: https://mathworld.wolfram.com/RotationMatrix.html

    Args:
        pitch (float): represents a rotation in the x axis in radians
        yaw (float): represents a rotation in the y axis in radians
        roll (float): represents a rotation in the z axis in radians
    Return:
        A rotation matrix.
    """

    cos_pitch = np.cos(pitch)
    sin_pitch = np.sin(pitch)
    pitch_matrix = np.array(
        (1, 0, 0, 0, cos_pitch, sin_pitch, 0, -sin_pitch, cos_pitch)
    ).reshape((3, 3))

    cos_yaw = np.cos(yaw)
    sin_yaw = np.sin(yaw)
    yaw_matrix = np.array((cos_yaw, 0, -sin_yaw, 0, 1, 0, sin_yaw, 0, cos_yaw)).reshape(
        (3, 3)
    )

    cos_roll = np.cos(roll)
    sin_roll = np.sin(roll)
    roll_matrix = np.array(
        (cos_roll, sin_roll, 0, -sin_roll, cos_roll, 0, 0, 0, 1)
    ).reshape((3, 3))

    rotation_matrix = pitch_matrix @ yaw_matrix @ roll_matrix

    return rotation_matrix


class Rotation:
    """Represents a rotation, allowing it to be applied to coordinate maps.

    The intended usage of this class is to being created with rotation
    parameters, and use the method rotate_coordinate_map to rotate the maps
    generated by classes that follow the ProjectionImage protocol
    (photonbend.core.projection.ProjectionImage).

    Attributes:
        pitch (float): rotation measured in radians in the pitch axis.
        yaw (float): rotation measured in radians in the yaw axis.
        roll (float): rotation measured in radians in the roll axis.

    Example:
        Rotate an image using a rotation and a coordinate map:

            # Get a coordinate map
            coordinate_map = a_projection_image.get_coordinate_map()
            # Creates a rotation and uses it to rotate a coordinate map
            rotation = Rotation(np.pi/2, 0, 0)
            rotated_coordinate_map = rotation.rotate_coordinate_map(coordinate_map)
            # Get the rotated image array
            rotated_image = a_projection_image
                .process_coordinate_map(rotated_coordinate_map)

    """

    def __init__(self, pitch: float, yaw: float, roll: float) -> None:
        """Initializes instance attributes

        Args:
            pitch (float): rotation measured in radians in the pitch axis.
            yaw (float): rotation measured in radians in the yaw axis.
            roll (float): rotation measured in radians in the roll axis.
        """
        self.rotation_matrix = _calculate_rotation_matrix(-pitch, -yaw, -roll)

    def rotate_coordinate_map(
        self, coordinate_map: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]:
        """Rotates a coordinate map in the pitch, yaw, and roll axis.

        Rotates a coordinate map, producing a new coordinate map.

        *For more info about coordinate maps check the documentation for
        the photonbend.core module.*

        Args:
            coordinate_map (np.ndarray): A numpy array of float64.
        Returns:
            The rotated coordinate map with the same shape as the input.
        """

        # Create views the various elements of the coordinate map into components
        polar_map = coordinate_map[:, :, :2]
        latitude = polar_map[:, :, 0]
        longitude = polar_map[:, :, 1]

        # Create an invalid selector so we can do some clean up
        invalid_map = coordinate_map[:, :, 2] != 0.0
        polar_map[invalid_map] = 0

        # Convert the polar coordinate map into 3 maps representing 3D
        # coordinates (x, y, z)
        y = np.cos(latitude)
        xz = np.exp(longitude * 1j) * np.sin(latitude)
        x = xz.real
        z = xz.imag

        # Concatenate the elements to produce a single map
        x = np.expand_dims(x, axis=2)
        y = np.expand_dims(y, axis=2)
        z = np.expand_dims(z, axis=2)
        position_vector: npt.NDArray[np.float64] = np.concatenate([x, y, z], axis=2)
        # Expands the dimensions to use fast matrix multiplication
        position_vector = np.expand_dims(position_vector, axis=3)

        # Does the rotation using matrix multiplication in the last two axis of
        # both matrices, which works really fast
        new_position_vector = np.matmul(
            self.rotation_matrix,
            position_vector,
            axes=[
                (-2, -1),
                (-2, -1),
                (-2, -1),
            ],
        )
        new_position_vector = new_position_vector.reshape(
            new_position_vector.shape[:-1]
        )

        # Turn the 3D map back into a polar coordinate map
        translated_latitude = np.arccos(new_position_vector[:, :, 1])
        translated_xz = make_complex(
            new_position_vector[:, :, 0], new_position_vector[:, :, 2]
        )
        translated_longitude = np.log(translated_xz).imag

        translated_latitude = np.expand_dims(translated_latitude, axis=2)
        translated_longitude = np.expand_dims(translated_longitude, axis=2)
        new_invalid_map = np.expand_dims(invalid_map, axis=2)

        pre_ans = np.concatenate([translated_latitude, translated_longitude], axis=2)
        # cleans the data before returning to ensure all other functions will work
        pre_ans[invalid_map] = 0

        # concatenate the invalid map to the back of pre_ans to complete the data
        ans = np.concatenate([pre_ans, new_invalid_map], axis=2)
        return ans
