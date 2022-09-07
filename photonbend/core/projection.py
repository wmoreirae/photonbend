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

# TODO try to use the old SphereImage supersampling code to improve the new base

from typing import Protocol, Union, TypeVar
from abc import abstractmethod
import numpy as np
import numpy.typing as npt

from photonbend.core._shared import make_complex
from photonbend.core.lens import Lens


UniFloat = TypeVar("UniFloat", float, npt.NDArray[np.float64])


class ProjectionImage(Protocol):
    image: np.ndarray

    @abstractmethod
    def get_coordinate_map(self) -> npt.NDArray[np.float64]:
        ...

    @abstractmethod
    def process_coordinate_map(
        self, coordinate_map: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.uint8]:
        ...


class CameraImage(ProjectionImage):
    """Store and process camera-based images and their coordinates.

    This class maps each pixel of an image to a polar coordinates of the form
    (latitude, longitude) based on its attributes.
    It can handle images that follow the principles of camera-based imagery.

    Attributes:
        image (np.ndarray[int8]): The image as a numpy array with the shape
            (height, width, 3).
        fov (float): The image Field of View in radians.
        forward_lens (Callable): The lens function of this image.
        reverse_lens (Callable): The inverse of this image's lens function.
        magnitude (float): The distance in pixels from the center of the image
            where the maximum FoV is reached.
        f_distance (float): The focal distance of this image in pixels.
    """

    def __init__(
        self,
        image_arr: npt.NDArray[np.uint8],
        fov: float,
        lens: Lens,
        magnitude: Union[None, float] = None,
    ):
        """Initializes instance attributes.
        Args:
            image_arr (np.ndarray): A numpy array of int8 representing an RGB
                image. The image follows the shape (height, width, 3).
            fov (float): Thehe Field of View in radians.
            lens (Lens): A lens with its forward and reverse functions.
            magnitude (float): The distance in pixels from the center of the
                image where the maximum FoV is reached.
                For the default case of an inscribed circle image, this value is
                calculated automatically, therefore it should only be used when
                passing an image that is not an inscribed circle.

                **Examples:**
                    * For the inscribed image, it is the image width or
                        height divided by 2.
                    * For the full canvas image, it is the distance in
                        pixels of the image center to one of its
                        corners.
        """
        self.image = image_arr
        self.fov = fov

        self.forward_lens = lens.forward_function
        self.reverse_lens = lens.reverse_function

        self.magnitude: float = (
            (self.image.shape[0] / 2.0) if (magnitude is None) else magnitude
        )
        self.f_distance = self._compute_f_distance()

    def _compute_f_distance(self) -> float:
        """
        This method compute the f_distance (focal distance in pixels)

        THIS IS NOT PART OF THE API - USE AT YOUR OWN RISK

        Usually only the self.init method should call this.

        It computes the maximum distance the maximum angle this lens is set to produce
        in focal distances. To simplify the calculations, we always use a focal distance
        of one, and make the dots per focal distance (dpf) variable.
        So, in order to calculate the dpf, we measure the maximum distance the lens
        produce if focal distances, we calculate the longest vector of this image (from
        the center of the image to one of it's sides) and we divide the second by the
        first to arrive at the dpf.  Then we set this to the current object.

        :return: float
        """
        maximum_lens_angle = self.fov / 2
        maximum_image_magnitude = self.magnitude
        lens_max_angle_magnitude = self.forward_lens(maximum_lens_angle)
        return maximum_image_magnitude / lens_max_angle_magnitude

    # Protocol implementation
    def get_coordinate_map(self) -> npt.NDArray[np.float64]:
        """Returns this image coordinate map.

        Returns a coordinate map for this image based on its size, fov, lens
        function and magnitude.

        *For more information on coordinate maps check the documentation
        for the photonbend.core module.*

        Returns:
            A numpy array of float64 as a coordinate map.
        """
        o_height, o_width = self.image.shape[:2]

        # making of the mesh
        x_axis_range = np.linspace(-o_width / 2 + 0.5, o_width / 2 - 0.5, num=o_width)
        y_axis_range = np.linspace(
            o_height / 2 - 0.5, -o_height / 2 + 0.5, num=o_height
        )
        mesh_y, mesh_x = np.meshgrid(
            y_axis_range, x_axis_range, sparse=True, indexing="ij"
        )

        distance_mesh = np.sqrt(mesh_x**2 + mesh_y**2) / self.f_distance
        latitude: npt.NDArray[np.float64] = self.reverse_lens(distance_mesh)
        longitude = np.log(make_complex(mesh_x, mesh_y)).imag

        latitude = latitude.reshape(*latitude.shape, 1)
        longitude = longitude.reshape(*longitude.shape, 1)
        invalid = distance_mesh > self.forward_lens(self.fov / 2)

        invalid_float = invalid.astype(np.float64)
        invalid_float = np.expand_dims(invalid_float, axis=2)

        polar_coordinates = np.concatenate([latitude, longitude, invalid_float], 2)
        return polar_coordinates

    # Protocol implementation
    def process_coordinate_map(
        self, coordinate_map: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.uint8]:
        """Produces a new image based on a coordinate map.

        Process a given coordinate map and maps each of its coordinates to a
        pixel on this instance image, producing a new image.

        *For more information on coordinate maps, check the documentation
        for the photonbend.core module.*

        Args:
            coordinate_map (np.ndarray[np.float64]): A coordinate map.
        Returns:
            A new image based on the pixel data of this instance and the given
                coordinate map.
        """

        invalid_map = coordinate_map[:, :, 2] != 0.0
        polar_map = coordinate_map[:, :, :2]

        image_center = np.array(self.image.shape[:2]) / 2 - 0.5

        # Use valid values for the calculation. Must clean up later on!
        polar_map[invalid_map] = 0

        distance = self.forward_lens(polar_map[:, :, 0]) * self.f_distance
        unbalanced_position = np.exp(polar_map[:, :, 1] * 1j) * distance

        new_image_array = self.image[
            ((unbalanced_position.imag * (-1)) + image_center[0]).astype(int),
            (unbalanced_position.real + image_center[1]).astype(int),
        ]

        new_image_array[invalid_map] = 0

        return new_image_array


class PanoramaImage(ProjectionImage):
    """Store and process equirectangular panorama images and their coordinates.

    This class maps each pixel of an image to a polar coordinates of the form
    (latitude, longitude) based on its size. It can handle equirectangular
    panoramas (images with a 2:1 width to height ration).

    Attributes:
        image (np.ndarray[int]): The image as an array of shape
            (height, width, 3).
    """

    def __init__(self, image_arr: npt.NDArray[np.uint8]) -> None:
        """Initializes instance attributes

        Args:
            image_arr (np.ndarray): A numpy ndarray of shape (height, width, 3),
                where height is equal to half the width.
        """

        self.image = image_arr

    def get_coordinate_map(self) -> npt.NDArray[np.float64]:
        """Returns this image coordinate map.

        Returns a coordinate map for this image based solely on its  dimensions.

        *For more information on coordinate maps check the documentation for the
        photonbend.core module.*

        Returns:
            A numpy ndarray of float64 as a coordinate map.
        """

        height, width = self.image.shape[:2]
        half_pi_element = np.pi / width / 2

        x_axis_range = np.linspace(
            -np.pi + half_pi_element, np.pi - half_pi_element, num=width
        )
        y_axis_range = np.linspace(0, np.pi, num=height)
        mesh_y, mesh_x = np.meshgrid(
            y_axis_range, x_axis_range, sparse=False, indexing="ij"
        )
        mesh_y = mesh_y.reshape(*mesh_y.shape, 1)
        mesh_x = mesh_x.reshape(*mesh_x.shape, 1)
        invalid = np.zeros((*self.image.shape[:2], 1), np.float64)
        coordinate_map = np.concatenate((mesh_y, mesh_x, invalid), axis=2)
        return coordinate_map

    def process_coordinate_map(
        self, coordinate_map: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.uint8]:
        """Produces a new image based on a coordinate maps.

        Process a given coordinate map and maps each of its coordinates  to a
        pixel on this instance image, producing a new image.

        *For more information on coordinate maps, check the documentation for
        the photonbend.core module.*

        Args:
            coordinate_map (np.ndarray): A numpy array of float64 as a
            coordinate map.
        Returns:
            A new image (ndarray) based on the pixel data of this instance and
            the given coordinate map.
        """
        invalid_map = coordinate_map[:, :, 2] != 0.0
        polar_map = coordinate_map[:, :, :2]

        polar_map[invalid_map] = 0

        height, width = self.image.shape[:2]
        width_pi_segment = np.pi / (width / 2)
        height_pi_segment = np.pi / height

        latitude = polar_map[:, :, 0] / height_pi_segment
        longitude = polar_map[:, :, 1] / width_pi_segment + (width / 2)

        image = self.image[latitude.astype(int) % height, longitude.astype(int) % width]
        image[invalid_map] = 0
        return image


def map_projection(
    coordinate_map: npt.NDArray[np.float64],
) -> npt.NDArray[np.uint8]:
    """Converts a coordinate map to a color map.

    Converts a coordinate to a RGB color map so that we can see the projection.
    """
    rgb_range = 255.0

    invalid_map = coordinate_map[:, :, 2] != 0.0
    valid_map = np.logical_not(invalid_map)
    polar_map = coordinate_map[:, :, :2]

    polar_map[invalid_map] = 0

    # Distance
    distance = polar_map[:, :, 0]
    min_distance = np.min(distance[valid_map])
    max_distance = np.max(distance[valid_map])
    min_max_distance = max_distance - min_distance
    mm_factor = rgb_range / min_max_distance
    new_distance = distance.copy()
    new_distance[valid_map] -= min_distance
    new_distance[valid_map] *= mm_factor

    distance_map_8bits = np.round(new_distance).astype(np.uint8)

    # Direction
    unbalanced_position = polar_map[:, :, 1]
    d_factor = rgb_range / (np.pi * 2)
    position_map = d_factor * unbalanced_position
    position_map_8bits = np.round(position_map).astype(np.uint8)

    # TODO check if some overflow is happening
    invalid_map_8bits = (invalid_map.astype(np.uint8) * 255).astype(np.uint8)

    mapping_image = np.concatenate(
        [
            np.expand_dims(distance_map_8bits, 2),
            np.expand_dims(position_map_8bits, 2),
            np.expand_dims(invalid_map_8bits, 2),
        ],
        axis=2,
    )

    return mapping_image
