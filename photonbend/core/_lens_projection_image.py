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

from photonbend.core.protocols._projection_image import ProjectionImage
from .utils import make_complex
from typing import Tuple, Callable, Union

ForwardReverseLensFunction = Callable[[Union[float, npt.NDArray[float]]], Union[float, npt.NDArray[float]]]
LensFunction = Tuple[ForwardReverseLensFunction, ForwardReverseLensFunction]


class LensProjectionImage(ProjectionImage):
    def __init__(self, image_arr: npt.NDArray[np.core.int8], fov: float, lens: LensFunction,
                 magnitude: Union[None, float] = None):
        self.image = image_arr
        self.fov = fov

        forward_lens, reverse_lens = lens
        self.forward_lens = forward_lens
        self.reverse_lens = reverse_lens

        self.magnitude: float = (self.image.shape[0] / 2.0) if (magnitude is None) else magnitude
        self.dpf = self._compute_dpf()

    def _compute_dpf(self) -> float:
        """
        This method compute the dpf (dots per focal distance)

        Usually only the self.init method should call this.

        It computes the maximum distance the maximum angle this lens is set to produce in focal distances.
        To simplify the calculations, we always use a focal distance of one, and make the dots per focal distance (dpf)
        variable.
        So, in order to calculate the dpf, we measure the maximum distance the lens produce if focal distances,
        we calculate the longest vector of this image (from the center of the image to one of it's sides) and we
        divide the second by the first to arrive at the dpf.  Then we set this to the current object.

        :return: float
        """
        maximum_lens_angle = self.fov / 2
        maximum_image_magnitude = self.magnitude
        lens_max_angle_magnitude = self.forward_lens(maximum_lens_angle)
        return maximum_image_magnitude / lens_max_angle_magnitude

    # Protocol implementation
    def get_coordinate_map(self) -> npt.NDArray[np.core.float64]:
        o_height, o_width = self.image.shape[:2]

        # making of the mesh
        x_axis_range = np.linspace(-o_width / 2 + 0.5, o_width / 2 - 0.5, num=o_width)
        y_axis_range = np.linspace(o_height / 2 - 0.5, -o_height / 2 + 0.5, num=o_height)
        mesh_y, mesh_x = np.meshgrid(y_axis_range, x_axis_range, sparse=True, indexing='ij')

        distance_mesh = np.sqrt(mesh_x ** 2 + mesh_y ** 2) / self.dpf
        latitude: npt.NDArray[float] = self.reverse_lens(distance_mesh)
        longitude = np.log(make_complex(mesh_x, mesh_y)).imag

        latitude = latitude.reshape(*latitude.shape, 1)
        longitude = longitude.reshape(*longitude.shape, 1)
        invalid = distance_mesh > self.forward_lens(self.fov / 2)

        invalid_float = invalid.astype(np.core.float64)
        invalid_float = np.expand_dims(invalid_float, axis=2)

        polar_coordinates = np.concatenate([latitude, longitude, invalid_float], 2)
        return polar_coordinates

    # Protocol implementation
    def process_coordinate_map(self, coordinate_map: npt.NDArray[np.core.float64]) -> npt.NDArray[np.core.int8]:
        invalid_map = coordinate_map[:, :, 2] != 0.0
        polar_map = coordinate_map[:, :, :2]

        image_center = np.array(self.image.shape[:2]) / 2 - 0.5

        # Use valid values for the calculation. Must clean up later on!
        polar_map[invalid_map] = 0

        distance = self.forward_lens(polar_map[:, :, 0]) * self.dpf
        unbalanced_position = np.exp(polar_map[:, :, 1] * 1j) * distance

        new_image_array = self.image[
            ((unbalanced_position.imag * (-1)) + image_center[0]).astype(int),
            (unbalanced_position.real + image_center[1]).astype(int)]

        new_image_array[invalid_map] = 0

        return new_image_array
