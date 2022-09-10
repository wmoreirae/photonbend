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

from typing import Protocol, Union, TypeVar, Tuple
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
        height, width = self.image.shape[:2]

        invalid_map = coordinate_map[:, :, 2] != 0.0
        polar_map = coordinate_map[:, :, :2]

        image_center = np.array(self.image.shape[:2]) / 2 - 0.5

        # Use valid values for the calculation. Must clean up later on!
        polar_map[invalid_map] = 0

        distance = self.forward_lens(polar_map[:, :, 0]) * self.f_distance
        unbalanced_position = np.exp(polar_map[:, :, 1] * 1j) * distance

        # calculates the balanced positions
        balanced_position_y = (
            (unbalanced_position.imag * (-1)) + image_center[0]
        ).astype(int)
        balanced_position_x = (unbalanced_position.real + image_center[1]).astype(int)

        # removes bad positions in y
        problem_positions_y = np.logical_or(
            balanced_position_y >= height, balanced_position_y < 0
        )
        balanced_position_y[problem_positions_y] = 0

        # removes bad positions in x
        problem_positions_x = np.logical_or(
            balanced_position_x >= width, balanced_position_x < 0
        )
        balanced_position_x[problem_positions_x] = 0

        # makes a simplified map to set all bad positions to black
        problem_positions_yx = np.logical_or(problem_positions_y, problem_positions_x)

        # makes a new image
        new_image_array = self.image[
            balanced_position_y,
            balanced_position_x,
        ]

        # sets all pixels with detected bad positions to black
        new_image_array[problem_positions_yx] = 0

        # set the invalid areas as set in the coordinate map to black
        new_image_array[invalid_map] = 0

        return new_image_array


class DoubleCameraImage(ProjectionImage):
    def __init__(
        self,
        image_arr: npt.NDArray[np.uint8],
        sensor_fov: float,
        lens: Lens,
    ):
        """Initializes instance attributes.
        Args:
            image_arr (np.ndarray): A numpy array of int8 representing an RGB
                image. The image follows the shape (height, width, 3).
            sensor_fov (float): The Field of View in radians of a single
                sensor. Since 360 degrees cameras normally use 2 equal sensors
                in opposite directions, the software needs to know the FoV used
                by them.
            lens (Lens): A lens with its forward and reverse functions.
        """
        self.image = image_arr
        self.sensor_fov = sensor_fov

        self.forward_lens = lens.forward_function
        self.reverse_lens = lens.reverse_function
        self.magnitude = self.image.shape[0] / 2.0
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
        maximum_lens_angle = self.sensor_fov / 2
        maximum_image_magnitude = self.magnitude
        lens_max_angle_magnitude = self.forward_lens(maximum_lens_angle)
        return maximum_image_magnitude / lens_max_angle_magnitude

    def get_coordinate_map(self) -> npt.NDArray[np.float64]:
        """Returns this image coordinate map.

        Returns a coordinate map for this image based on its size, fov, lens
        function and magnitude.

        *For more information on coordinate maps check the documentation
        for the photonbend.core module.*

        Returns:
            A numpy array of float64 as a coordinate map.
        """

        half_width = self.image.shape[1] // 2

        # making of the meshes
        mesh_x, mesh_y = self._make_mesh()
        distance_mesh = np.sqrt(mesh_x**2 + mesh_y**2) / self.f_distance

        # Maps the invalid areas
        invalid_map = distance_mesh > self.forward_lens(self.sensor_fov / 2)

        # computes latitudes
        latitude: npt.NDArray[np.float64] = self.reverse_lens(distance_mesh)
        # image on the right has descending latitude (starts at Pi and reduces)
        latitude[:, half_width:] *= -1
        latitude[:, half_width:] += np.pi

        longitude = np.log(make_complex(mesh_x, mesh_y)).imag

        latitude = latitude.reshape(*latitude.shape, 1)
        longitude = longitude.reshape(*longitude.shape, 1)

        invalid_float = invalid_map.astype(np.float64)
        invalid_float = np.expand_dims(invalid_float, axis=2)

        polar_coordinates = np.concatenate([latitude, longitude, invalid_float], 2)
        return polar_coordinates

    def _make_mesh(self) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        original_height, original_width = self.image.shape[:2]

        half_width: int = original_width // 2
        half_x_axis_range = np.linspace(
            -half_width / 2 + 0.5, half_width / 2 - 0.5, num=half_width
        )
        # Double images have the X axis inverted on the right image
        half_x_axis_inverted = half_x_axis_range * (-1)

        # Join both images X axis
        x_axis_range = np.concatenate([half_x_axis_range, half_x_axis_inverted], 0)

        y_axis_range = np.linspace(
            original_height / 2 - 0.5, -original_height / 2 + 0.5, num=original_height
        )
        mesh_y, mesh_x = np.meshgrid(
            y_axis_range, x_axis_range, sparse=True, indexing="ij"
        )

        return mesh_x, mesh_y

    def process_coordinate_map(
        self, coordinate_map: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.uint8]:
        left_map = self._process_coordinate_map(coordinate_map, right_image=False)
        right_map = self._process_coordinate_map(coordinate_map, right_image=True)

        intermediate_map = left_map + right_map
        final_map = intermediate_map.astype(np.int8)
        return final_map

    def _process_coordinate_map(
        self, coordinate_map: npt.NDArray[np.float64], right_image: bool
    ) -> npt.NDArray[np.float64]:
        # Calculate the shape for half of the image horizontally
        real_height, real_width = self.image.shape[:2]
        height, width = real_height, real_width // 2
        fov_overdata_ref = (self.sensor_fov/2) - (np.pi / 2)
        fov_overdata_min = np.pi - fov_overdata_ref
        fov_overdata_max = np.pi + fov_overdata_ref

        # Get the data from the passed coordinate map
        invalid_map = coordinate_map[:, :, 2] != 0.0
        polar_map = coordinate_map[:, :, :2]  # latitude and longitude in radians

        image_center = np.array((height, width)) / 2 - 0.5

        # Use valid values for the calculation. Must clean up later on!
        polar_map[invalid_map] = 0

        # Separate latitude from longitude
        latitude = polar_map[:, :, 0]
        longitude = polar_map[:, :, 1]

        # rest of the method should work with local image
        local_image: npt.NDArray[np.int8]
        x_reflection: int  # Used to control whether the X axis should be inverted
        if not right_image:  # left image
            local_image = self.image[:, :width]
            x_reflection = 1
        else:  # right image
            local_image = self.image[:, width:]
            x_reflection = -1
            latitude = np.pi - latitude

        distance = self.forward_lens(latitude) * self.f_distance
        unbalanced_position = np.exp(longitude * 1j) * distance

        balanced_position_y = (
            (unbalanced_position.imag * (-1)) + image_center[0]
        ).astype(
            int
        )  # -1 to invert Y axis
        balanced_position_x = (
            unbalanced_position.real * x_reflection + image_center[1]
        ).astype(int)

        # removes bad positions in y
        problem_positions_y = np.logical_or(
            balanced_position_y >= height, balanced_position_y < 0
        )
        balanced_position_y[problem_positions_y] = 0

        # removes bad positions in x
        problem_positions_x = np.logical_or(
            balanced_position_x >= width, balanced_position_x < 0
        )
        balanced_position_x[problem_positions_x] = 0

        # makes a simplified map to set all bad positions to black
        problem_positions_yx = np.logical_or(problem_positions_y, problem_positions_x)

        # get the pixels in the overdata area and define a multiplication factor
        overdata_area = np.logical_and(latitude >= fov_overdata_min, latitude <= fov_overdata_max)
        overdata_factor = np.ones((height, width, 1), np.float64)


        # makes a new image
        new_image_array = local_image[
            balanced_position_y,
            balanced_position_x,
        ]

        # sets all pixels with detected bad positions to black
        new_image_array[problem_positions_yx] = 0

        # set the invalid areas as set in the coordinate map to black
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
