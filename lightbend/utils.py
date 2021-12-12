import numpy as np
from numba import njit


@njit
def angle_from_vector(vector):
    u_vector = unit_vector(vector)
    theta = np.log(vector).imag
    return theta


@njit
def vector_magnitude(vector):
    return np.sqrt(vector.real ** 2 + vector.imag ** 2)


@njit
def degrees_to_radians(degrees):
    return degrees / 180 * np.pi


@njit
def radians_to_degrees(rad):
    return rad / np.pi * 180


@njit
def dpi_to_dpmm(dpi: int):
    mm_in_a_inch = 25.4
    return dpi / mm_in_a_inch


@njit
def vector_from_coordinates(x, y):
    vector = complex(x, y)
    return vector


@njit
def unit_vector(vector):
    magnitude = vector_magnitude(vector)
    u_vector = vector / magnitude
    return u_vector


@njit
def c_round(complex_number):
    return int(np.round(complex_number.real)), int(np.round(complex_number.imag))


@njit
def vector_to_focal_units(vector, focal_distance, pixels_per_f_distance):
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
def calculate_lens_angle(vector, f_distance, dpi, inverse_mapping_function):
    normalized_magnitude = vector_to_focal_units(vector, f_distance, dpi)
    angle = inverse_mapping_function(normalized_magnitude)
    return angle * 2


@njit
def calculate_pixels_per_f_distance(vector, angle, f_distance, mapping_function):
    half_angle = angle / 2
    quasi_magnitude = mapping_function(half_angle) * f_distance
    v_magnitude = vector_magnitude(vector)
    pixels_pfd = v_magnitude / quasi_magnitude
    return pixels_pfd

"""
@njit
def calculate_f_distance(vector, angle, dpi, mapping_function):
    half_angle = angle / 2
    dpmm = dpi_to_dpmm(dpi)
    quasi_magnitude = mapping_function(half_angle)
    v_magnitude = vector_magnitude(vector)
    f_distance = v_magnitude / (quasi_magnitude * dpmm)
    return f_distance
"""