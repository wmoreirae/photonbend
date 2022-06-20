#  Copyright (c) 2022. Edson Moreira
#
#  Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
#  documentation files (the "Software"), to deal in the Software without restriction, including without limitation
#  the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and
#  to permit persons to whom the Software is furnished to do so, subject to the following conditions:
#
#  The above copyright notice and this permission notice shall be included in all copies or substantial portions of
#  the Software.
#
#  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING
#  BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
#  NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
#  DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
#  OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#
#  Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
#  documentation files (the "Software"), to deal in the Software without restriction, including without limitation
#  the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and
#  to permit persons to whom the Software is furnished to do so, subject to the following conditions:
#
#
from typing import Tuple

import numpy as np
from numba import njit, prange

from photonbend.core import SphereImage, LensImageType
from photonbend.utils import degrees_to_radians


@njit
def _projection_function(radius, standard_parallel, longitude_or_x: float, latitude_or_y: float, cardinal: bool):
    """ Gives the equirectangular projection function or it's inverse

    :param radius: The globe radius to project
    :param longitude_or_x:
    :param latitude_or_y:
    :param cardinal:
    :return:
    """
    x_or_long = _projection_x_longitude(radius, standard_parallel, longitude_or_x, cardinal)
    y_or_lat = _projection_y_latitude(radius, standard_parallel, latitude_or_y, cardinal)
    if not cardinal:
        x = x_or_long
        y = y_or_lat
        return x, y

    latitude = y_or_lat
    longitude = x_or_long
    return latitude, longitude


@njit
def _projection_x_longitude(radius: float, standard_parallel: float, longitude_or_x: float, cardinal: bool) -> float:
    if not cardinal:
        x = (radius * (longitude_or_x + np.pi)) * np.cos(standard_parallel)
        return x
    else:
        longitude = (longitude_or_x / (np.cos(standard_parallel) * radius)) - np.pi
        return longitude


@njit
def _projection_y_latitude(radius, standard_parallel: float, latitude_or_y: float, cardinal: bool) -> float:
    destiny_center_height = radius * np.pi / 2
    if not cardinal:
        y = destiny_center_height - (radius * latitude_or_y)
        return y
    else:
        latitude = (-(latitude_or_y - destiny_center_height)) / radius
        return latitude


@njit(parallel=False)
def make_projection(source: SphereImage, standard_parallel, desired_width) -> np.ndarray:
    # TODO Add a super sampler

    radius = desired_width / (np.pi * 2)

    destiny_width = int(np.abs(round(_projection_x_longitude(radius, standard_parallel, np.pi, False))))
    destiny_height = int(np.abs(round(_projection_y_latitude(radius, standard_parallel, -np.pi / 2, False))))
    destiny_array = np.zeros((destiny_height, destiny_width, 3), np.core.uint8)

    for row in prange(destiny_height):
        latitude = _projection_y_latitude(radius, standard_parallel, row, True)
        for column in prange(destiny_width):
            longitude = _projection_x_longitude(radius, standard_parallel, column, True)
            try:
                destiny_array[row, column, :] = source.get_from_spherical(latitude, longitude)
            except Exception:
                continue

    return destiny_array


@njit(parallel=False)
def make_sphere_image(source: np.array, lens, image_type: LensImageType, fov: float,
                      rotation: Tuple[float, float, float]):
    source_height, source_width = source.shape[:2]
    standard_parallel = np.arccos(min(1.0, source_width / 2 / source_height))

    # -0.1 is a dirty trick to go around rounding problems and continuous nature of geometry
    radius = (source_height) / np.pi

    destiny_height = int(round(np.pi * radius))
    destiny_width = destiny_height

    if image_type == LensImageType.DOUBLE_INSCRIBED:
        destiny_width = destiny_width * 2

    d_sphere = SphereImage(np.zeros((destiny_height, destiny_width, 3), np.core.uint8), image_type,
                           degrees_to_radians(fov), lens)

    if rotation != (0, 0, 0):
        rotation_rad = list(map(degrees_to_radians, rotation))
        pitch, yaw, roll = rotation_rad
        d_sphere.set_rotation(pitch, roll, yaw)

    for row in range(destiny_height):
        for column in range(destiny_width):
            try:
                latitude, longitude = d_sphere.translate_cartesian_to_spherical(column, row)

                x, y = _projection_function(radius, standard_parallel, longitude, latitude, False)

                x = int(round(x))
                x = x % source_width
                y = int(round(y))
                y = y % source_height

                if x < 0 or x >= source_width:
                    continue
                if y < 0 or y >= source_height:
                    continue
                # x = x % destiny_width
                # y = y % destiny_height

                value = source[y, x]
                d_sphere.set_to_cartesian(column, row, value)
            except Exception:
                continue
    return d_sphere


def compute_best_width(source: SphereImage) -> int:
    result = 2 * (int(round(2 * source.lens_image.shape[0])) // 2)
    return result
