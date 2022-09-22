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

__doc__ = """
This module provides the classes and methods that allow you to map pixels to angles and
vice-versa. It allows you to convert between different sort of images through its
classes.

"""

from typing import Protocol, Union, TypeVar, Tuple
from abc import abstractmethod
import numpy as np
import numpy.typing as npt

from photonbend.core._shared import make_complex
from photonbend.core.lens import Lens
from photonbend.utils import to_radians

UniFloat = TypeVar("UniFloat", float, npt.NDArray[np.float64])


class ProjectionImage(Protocol):
    """Defines the protocol used by all projection images."""

    image: np.ndarray

    @abstractmethod
    def get_coordinate_map(self) -> npt.NDArray[np.float64]:
        """Should return this image coordinate map.

        Returns a coordinate map for this image based on its structure

        *For more information on coordinate maps check the documentation
        for the photonbend.core module.*"""
        ...

    @abstractmethod
    def process_coordinate_map(
        self, coordinate_map: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.uint8]:
        """Should map the image based on the received coordinate map.

        Returns a image by mapping this image's pixels to the received coordinate map
        based on its coordinates and its own mapping.

        *For more information on coordinate maps check the documentation
        for the photonbend.core module.*"""
        ...


class CameraImage(ProjectionImage):
    """Store and process camera-based images and their coordinates.

    This class maps each pixel of an image to a geodesic-like coordinate of the form
    (latitude, longitude) based on its attributes.
    It can handle images that follow the principles of camera-based imagery.

    Attributes:
        image (np.ndarray[int8]): The image as a numpy array with the shape
            (height, width, 3).
        fov (float): The image Field of View in radians.
        lens (Lens): This image's lens instance.
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
            image_arr (numpy.ndarray): A numpy array of int8 representing an RGB
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
        maximum_incidence_angle = self.fov / 2
        max_magnitude_in_pixels = self.magnitude
        max_projection_distance_in_f_units = self.forward_lens(maximum_incidence_angle)
        return max_magnitude_in_pixels / max_projection_distance_in_f_units

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
        latitude, longitude = self._compute_latitude_longitude()
        invalid = latitude > self.fov / 2

        latitude = latitude.reshape(*latitude.shape, 1)
        longitude = longitude.reshape(*longitude.shape, 1)

        invalid_float = invalid.astype(np.float64)
        invalid_float = np.expand_dims(invalid_float, axis=2)

        coordinate_map = np.concatenate([latitude, longitude, invalid_float], 2)
        return coordinate_map

    def _compute_latitude_longitude(
        self,
    ) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        o_height, o_width = self.image.shape[:2]

        # making a the mesh to represent the pixel coordinates
        x_axis_range = np.linspace(-o_width / 2 + 0.5, o_width / 2 - 0.5, num=o_width)
        y_axis_range = np.linspace(
            o_height / 2 - 0.5, -o_height / 2 + 0.5, num=o_height
        )
        mesh_y, mesh_x = np.meshgrid(
            y_axis_range, x_axis_range, sparse=True, indexing="ij"
        )

        # uses euclidean distance to compute pixel distances from the center
        distance_mesh = np.sqrt(mesh_x**2 + mesh_y**2) / self.f_distance

        # uses the reverse lens function to get an angle of incidence for each pixel
        latitude: npt.NDArray[np.float64] = self.reverse_lens(distance_mesh)

        # uses complex math to get the angle as used when using polar coordinates on the
        # cartesian plane
        longitude = np.log(make_complex(mesh_x, mesh_y)).imag
        return latitude, longitude

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
        latitude = coordinate_map[:, :, 0]
        longitude = coordinate_map[:, :, 1]

        positions_x, positions_y = self._make_cartesian_map(latitude, longitude)

        # removes bad positions in y
        problem_positions_y = np.logical_or(positions_y >= height, positions_y < 0)
        positions_y[problem_positions_y] = 0

        # removes bad positions in x
        problem_positions_x = np.logical_or(positions_x >= width, positions_x < 0)
        positions_x[problem_positions_x] = 0

        # makes a simplified map to set all bad positions to black
        problem_positions_yx = np.logical_or(problem_positions_y, problem_positions_x)

        # makes a new image
        new_image_array = self.image[
            positions_y,
            positions_x,
        ]

        # sets all pixels with detected bad positions to black
        new_image_array[problem_positions_yx] = 0

        # set the invalid areas as set in the coordinate map to black
        new_image_array[invalid_map] = 0

        return new_image_array

    def _make_cartesian_map(self, latitude, longitude):
        image_center = self._get_image_center()
        # Use valid values for the calculation. Must clean up later on!
        # lat_long_map[invalid_map] = 0
        distance = self.forward_lens(latitude) * self.f_distance
        unbalanced_cartesian_position = np.exp(longitude * 1j) * distance
        # calculates the balanced positions
        balanced_position_y = (
            (unbalanced_cartesian_position.imag * (-1)) + image_center[0]
        ).astype(int)
        balanced_position_x = (
            unbalanced_cartesian_position.real + image_center[1]
        ).astype(int)
        return balanced_position_x, balanced_position_y

    def _get_image_center(self):
        """Gets the image center

        Even though pixels have an area, when dealing with them on the computer, they
        seem to be a point-like entity, residing on a specific position, like [1,1] or
        [1,2].
        A [2,2] image for example, have points one distributed like [0,0], [0,1], [1,0]
        and [1,1]. Knowing that fact, it seems clear that the middle should be
        [0.5, 0.5]. This method does that so that you don't have to remember the
        details.
        """

        return np.array(self.image.shape[:2]) / 2 - 0.5


class DoubleCameraImage(ProjectionImage):
    """Store and process 360 degrees camera-based images and their coordinates.

    This class maps each pixel of an image to a geodesic-like coordinate of the form
    (latitude, longitude) based on its attributes.
    It can handle images that follow the principles of 360 degrees cameras, which
    store the two images captured by its 2 opposite sensors, side-by-side on a
    single image file.

    Attributes:
        image (np.ndarray[int8]): The image as a numpy array with the shape
            (height, width, 3).
        fov (float): The image Field of View in radians for each sensor.
        lens (Lens): This image's lens instance.
        magnitude (float): The distance in pixels from the center of the image
            where the maximum FoV is reached.
        f_distance (float): The focal distance of this image in pixels.
    """

    def __init__(
        self, image_arr: npt.NDArray[np.uint8], sensor_fov: float, lens: Lens, **kwargs
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

        self.lens = lens
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

        latitude, longitude = self._compute_latitude_longitude()
        half_width = self.image.shape[1] // 2
        # Maps the invalid areas
        invalid_map = latitude > self.sensor_fov / 2.0
        invalid_map[:, half_width:] = latitude[:, half_width:] < np.pi - (
            self.sensor_fov / 2.0
        )
        latitude = latitude.reshape(*latitude.shape, 1)
        longitude = longitude.reshape(*longitude.shape, 1)

        invalid_float = invalid_map.astype(np.float64)
        invalid_float = np.expand_dims(invalid_float, axis=2)

        polar_coordinates = np.concatenate([latitude, longitude, invalid_float], 2)
        return polar_coordinates

    def _compute_latitude_longitude(self):
        half_width = self.image.shape[1] // 2

        # making of 2 meshes
        mesh_x, mesh_y = self._make_mesh()
        distance_mesh = np.sqrt(mesh_x**2 + mesh_y**2) / self.f_distance

        # computes latitudes
        latitude: npt.NDArray[np.float64] = self.reverse_lens(distance_mesh)

        # image on the right has descending latitude (starts at Pi and reduces)
        latitude[:, half_width:] *= -1
        latitude[:, half_width:] += np.pi
        longitude = np.log(make_complex(mesh_x, mesh_y)).imag
        return latitude, longitude

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
        # Calculate the shape for half of the image horizontally
        real_height, real_width = self.image.shape[:2]
        width = real_width // 2
        fov_merger_ref = (self.sensor_fov / 2) - (np.pi / 2)
        fov_merger_min = np.pi / 2 - fov_merger_ref
        fov_merger_max = np.pi / 2 + fov_merger_ref
        fov_merger_range = 2.0 * fov_merger_ref
        fov_merger_safety = to_radians(0.5)  # a margin value on a fade gradient

        # Get the data from the passed coordinate map
        invalid_map = coordinate_map[:, :, 2] != 0.0

        left_coordinate_map = coordinate_map

        right_coordinate_map = np.copy(coordinate_map)
        right_coordinate_map[:, :, 0] *= -1
        right_coordinate_map[:, :, 0] += np.pi

        left_image_data = self.image[:, :width]
        right_image_data = np.copy(self.image[:, width:])
        right_image_data = right_image_data[:, ::-1]

        left_cam_image = CameraImage(left_image_data, self.sensor_fov, self.lens)
        right_cam_image = CameraImage(right_image_data, self.sensor_fov, self.lens)

        left_mapping = left_cam_image.process_coordinate_map(left_coordinate_map)
        right_mapping = right_cam_image.process_coordinate_map(right_coordinate_map)

        left_latitude = left_coordinate_map[:, :, 0]
        left_merger_map = np.logical_and(
            left_latitude >= fov_merger_min,
            left_latitude <= (fov_merger_max + fov_merger_safety),
        )
        left_factor_map = (left_latitude - fov_merger_max) / fov_merger_range * -1
        left_factor_map[np.logical_not(left_merger_map)] = 1.0
        left_factor_map = np.expand_dims(left_factor_map, 2)
        left_image = left_mapping.astype(np.float64) * (left_factor_map)

        right_latitude = right_coordinate_map[:, :, 0]
        right_merger_map = np.logical_and(
            right_latitude >= fov_merger_min,
            right_latitude <= (fov_merger_max + fov_merger_safety),
        )
        right_factor_map = (right_latitude - fov_merger_max) / fov_merger_range * -1
        right_factor_map[np.logical_not(right_merger_map)] = 1.0
        right_factor_map = np.expand_dims(right_factor_map, 2)
        right_image = right_mapping.astype(np.float64) * (right_factor_map)

        final_image = (left_image + right_image).astype(np.uint8)
        final_image[invalid_map] = 0

        return final_image


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

    A simple visualization method that allows one to converts a coordinate map to to a
    RGB color map so that we can see the projection.

    Latitude gets translated to **red**, Longitude to **green** and invalid area mapping
    gets translated to **blue**.
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
