#  Copyright (c) 2022. Edson Moreira
#
#  Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
#  documentation files (the "Software"), to deal in the Software without restriction, including without limitation
#  the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and
#  to permit persons to whom the Software is furnished to do so, subject to the following conditions:
#
#  The above copyright notice and this permission notice shall be included in all copies or substantial portions
#  of the Software.
#
#  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING
#  BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
#  NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
#  DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
#  OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
from typing import Tuple, Callable

import numpy as np
from numba import njit


@njit
def decompose(a_complex_number):
    """Decomposes a complex number into it's real and imaginary parts and returns them as integers.
    To turn the parts into integers, it rounds them first and the proceed s to cast them.

    :param a_complex_number: any complex number
    :return: a tuple of integers representing the real and imaginary parts of the original number
    """
    x = int(round(a_complex_number.real))
    y = int(round(a_complex_number.imag))
    return x, y


@njit
def angle_from_vector(vector: complex):
    u_vector = unit_vector(vector)
    theta = np.log(vector).imag
    return theta


@njit
def vector_magnitude(vector: complex):
    """Computes the magnitude of a complex number vector

        :param vector: A complex number
        :return: A floating point scalar of the vector magnitude
        """
    return np.sqrt(vector.real ** 2 + vector.imag ** 2)


@njit
def degrees_to_radians(degrees: float):
    return degrees / 180 * np.pi


@njit
def radians_to_degrees(rad: float) -> float:
    return rad / np.pi * 180.0


@njit
def dpi_to_dpmm(dpi: int) -> float:
    mm_in_a_inch = 25.4
    return dpi / mm_in_a_inch


@njit
def vector_from_coordinates(x, y) -> complex:
    vector = complex(x, y)
    return vector


@njit
def unit_vector(vector: complex) -> complex:
    magnitude = vector_magnitude(vector)
    if 0 == magnitude:
        return complex(0, 0)
    u_vector = vector / magnitude
    return u_vector


@njit
def c_round(complex_number) -> Tuple[int, int]:
    return int(round(complex_number.real)), int(round(complex_number.imag))


@njit
def vector_to_focal_units(vector: complex, focal_distance: float, pixels_per_f_distance: float) -> float:
    """Calculates the projection of a pixel.

    With the projection of a pixel, you can use an inverse_mapping_function to get the lens angle of it
    The projection is 'normalized' using the focal distance as the factor, therefore, 'normalized' may be a
    misnomer in case of a  any mapping function that produces values bigger than 1.
    """
    opposite_side = vector_magnitude(vector)
    adjacent_side = focal_distance * pixels_per_f_distance

    normalized_projection_magnitude = np.absolute(opposite_side / adjacent_side)

    return normalized_projection_magnitude


@njit
def calculate_lens_angle(vector: complex, f_distance: float, dpi: float,
                         mapping_function: Callable[[float, bool], float]):
    normalized_magnitude = vector_to_focal_units(vector, f_distance, dpi)
    angle = mapping_function(normalized_magnitude, True)
    return angle * 2


@njit
def calculate_pixels_per_f_distance(vector: complex, angle: float, f_distance: float,
                                    mapping_function: Callable[[float, bool], float]):
    half_angle = angle / 2
    quasi_magnitude = mapping_function(half_angle, False) * f_distance
    v_magnitude = vector_magnitude(vector)
    pixels_pfd = v_magnitude / quasi_magnitude
    return pixels_pfd


# TODO MAKE A TRANSLATE TO GEODESIC

@njit
def _Z_get_360_longitude(longitude):
    new_longitude = longitude % FULL_CIRCLE
    return new_longitude


@njit
def _Z_get_180_longitude(longitude):
    new_longitude = longitude % FULL_CIRCLE
    if np.pi < new_longitude:
        new_longitude -= (2 * np.pi)
    return new_longitude


@njit
def decompose(a_complex_number):
    """Decomposes a complex number into it's real and imaginary parts and returns them as integers.
    To turn the parts into integers, it rounds them first and the proceed s to cast them.

    :param a_complex_number: any complex number
    :return: a tuple of integers representing the real and imaginary parts of the original number
    """
    x = int(round(a_complex_number.real))
    y = int(round(a_complex_number.imag))
    return x, y


@njit
def weighted_sum(v1, v2, w1, w2):
    sum_weight = w1 + w2
    v1_arr = np.zeros(3, np.core.uint64)
    v2_arr = np.zeros(3, np.core.uint64)

    v1_arr[:] = v1[:]
    v2_arr[:] = v2[:]

    v1 = v1 * w1
    v2 = v2 * w2
    v3 = v1 + v2
    v3 = v3 / sum_weight
    v3 = v3.astype(np.core.uint8)
    return v3


@njit
def _2ints(v1, v2):
    return int(v1), int(v2)
