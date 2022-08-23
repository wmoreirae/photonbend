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
from numpy import typing as npt

from .projection_image import ProjectionImage


class PanoramaProjectionImage(ProjectionImage):
    def __init__(self, image_arr: npt.NDArray[np.core.int8]):
        self.image = image_arr

    def get_coordinate_map(self) -> npt.NDArray[np.core.float64]:
        height, width = self.image.shape[:2]
        half_pi_element = np.pi / width / 2

        x_axis_range = np.linspace(-np.pi + half_pi_element, np.pi - half_pi_element, num=width)
        y_axis_range = np.linspace(0, np.pi, num=height)
        mesh_y, mesh_x = np.meshgrid(y_axis_range, x_axis_range, sparse=False, indexing='ij')
        mesh_y = mesh_y.reshape(*mesh_y.shape, 1)
        mesh_x = mesh_x.reshape(*mesh_x.shape, 1)
        invalid = np.zeros((*self.image.shape[:2], 1), np.core.float64)
        coordinate_map = np.concatenate((mesh_y, mesh_x, invalid), axis=2)
        return coordinate_map

    def process_coordinate_map(self, coordinate_map: npt.NDArray[np.core.int8]) -> npt.NDArray[np.core.int8]:
        invalid_map = coordinate_map[:, :, 2] != 0.0
        polar_map = coordinate_map[:, :, :2]

        polar_map[invalid_map] = 0

        height, width = self.image.shape[:2]
        width_pi_segment = np.pi / (width / 2)
        height_pi_segment = np.pi / height

        latitude = polar_map[:, :, 0] / height_pi_segment
        longitude = polar_map[:, :, 1] / width_pi_segment + (width / 2)

        image = self.image[latitude.astype(int), longitude.astype(int)]
        image[invalid_map] = 0
        return image
