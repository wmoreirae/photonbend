from enum import IntEnum, auto

import numpy as np

from numba import uint8, float64, njit, prange, typeof, complex128, cfunc, int64
from numba.experimental import jitclass
from lightbend.utils import degrees_to_radians, radians_to_degrees
from lightbend.exceptions.coordinates_out_of_image import CoordinatesOutOfImage


@cfunc(float64(float64))
def _a(_b):
    return _b

spec = [
    ('center', complex128),
    ('image_type', int64),
    ('fov', float64),
    ('dpf', float64),
    ('image', uint8[:, :, :]),
    ('lens', typeof(_a)),
    ('i_lens', typeof(_a)),
]


@njit
def vector_magnitude(vector):
    return np.sqrt(vector.real ** 2 + vector.imag ** 2)


@njit
def decompose(a_complex_number):
    x = int(np.round(a_complex_number.real))
    y = int(np.round(a_complex_number.imag))
    return x, y


class ImageType(IntEnum):
    FULL_FRAME = auto()
    CROPPED_CIRCLE = auto()
    INSCRIBED = auto()
    DOUBLE_INSCRIBED = auto()


@jitclass(spec)
class SphereImageInscribed:

    def __init__(self, image_arr, image_type, fov, lens, i_lens):
        self.image = image_arr
        self.image_type = image_type
        self.lens = lens
        self.i_lens = i_lens
        self.fov = fov
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

        It computes the maximum distance the maximum angle this lens is set to produce if focal distances.
        To simplify the calculations, we always use a focal distance of one, and make the dots per focal distance (dpf)
        variable.
        So, in order to calculate the dpf, we measure the maximum distance the lens produce if focal distances,
        we calculate the longest vector of this image (from the center of the image to one of it's sides) and we
        divide the second by the first to arrive at the dpf.  Then we set this to the current object.

        :return: None
        """
        maximum_lens_angle = self.fov / 2
        maximum_image_magnitude = self.center.real
        lens_max_angle_magnitude = self.lens(maximum_lens_angle)
        self.dpf = maximum_image_magnitude / lens_max_angle_magnitude

    def get_image_array(self):
        """

        :return: A copy of the image array
        """
        return np.copy(self.image)

    @property
    def shape(self):
        return self.image.shape

    def check_position(self, x, y):
        height, width = self.image.shape[:2]
        if (0 > x or x > width) or (0 > y or y > height):
            return False
        return True

    def get_flat_position(self, x, y):
        return self.image[y, x, :]

    def get_from_coordinates(self, latitude, longitude) -> uint8[:]:
        x, y = self._get_image_position_from_coordinates(latitude, longitude)
        if self.check_position(x, y):
            return self.image[y, x, :]
        return np.zeros(3, np.core.uint8)

    def set_to_coordinates(self, latitude, longitude, data):
        x, y = self._get_image_position_from_coordinates(latitude, longitude)
        if self.check_position(x, y):
            self.image[y, x, :] = data

    def get_from_coordinates_deg(self, latitude, longitude):
        lat = degrees_to_radians(latitude)
        lon = degrees_to_radians(longitude)
        return self.get_from_coordinates(lat, lon)

    def _get_image_position_from_coordinates(self, latitude, longitude):
        # assert -np.pi / 2 <= latitude <= np.pi / 2, f"latitude should be between {np.pi / 2} and -{np.pi / 2}"
        # assert np.pi <= longitude <= -np.pi, f"longitude should be between {np.pi} and -{np.pi}"
        half_fov = self.fov / 2
        center_distance = self.lens(half_fov - latitude) * self.dpf
        factors = np.exp(longitude * 1j)
        relative_position = factors * center_distance

        position = self.relative_to_absolute(relative_position)
        x, y = decompose(position)

        return x, y

    def _get_coordinates_from_image_position(self, x, y):
        height, width = self.image.shape[:2]
        # assert (0 <= x <= width) and (0 <= y <= height), "x and y must be a point inside the image"
        half_fov = self.fov / 2
        absolute_position = complex(x, y)
        relative_position = self.absolute_to_relative(absolute_position)
        magnitude = vector_magnitude(relative_position) - 0.5
        latitude = half_fov - self.i_lens(magnitude / self.dpf)  # TODO Change for different angles
        normalized_position = relative_position / magnitude
        longitude = np.log(normalized_position).imag

        return latitude, longitude

    def map_from_sphere_image(self, sphere_image):
        _map_from_sphere_image(self, sphere_image)

    def relative_to_absolute(self, relative_position):
        return (relative_position.real + self.center.real) + 1j * (self.center.imag - relative_position.imag)

    def absolute_to_relative(self, absolute_position):
        return (absolute_position.real - self.center.real) + 1j * (self.center.imag - absolute_position.imag)

@njit(parallel=True)
def _map_from_sphere_image(first_image, sphere_image):
    height, width = first_image.image.shape[:2]

    for x in prange(width):
        for y in prange(height):
            lat, lon = first_image._get_coordinates_from_image_position(x, y)
            pixel_values = np.zeros((1, 1, 3), np.core.uint8)
            pixel_values[0, 0, :] = sphere_image.get_from_coordinates(lat, lon)
            first_image.image[y, x, :] = pixel_values[0, 0, :]
    return
