import numpy as np
from numba import njit, prange

from lightbend.utils import degrees_to_radians

_source_focal_distance = 1.0


# TODO fix this function to integrate well with the rest of the library
@njit(parallel=True)
def make_panoramic(source, source_function, inverse_source_function, destiny_function, inverse_destiny_function):
    source_height, source_width = source.shape[:2]
    assert source_height % 2 == source_width % 2 == 0, "Source image dimensions sizes must be even"
    source_dpi = source_height / 2 / source_function(degrees_to_radians(90))
    source_center = complex(source_width, source_height) / 2 - 0.5

    quality_factor = compute_quality_factor(source_function, destiny_function)
    destiny_size = compute_destiny_size(complex(source_width, source_height), source_function, destiny_function)
    destiny_size = destiny_size * quality_factor
    destiny_width = int(np.round(destiny_size.real))
    destiny_height = int(np.round(destiny_size.imag))
    destiny_dpi = destiny_height / destiny_function(degrees_to_radians(90))

    destiny_center_row = destiny_height - 0.5

    angle_of_a_column = (np.pi * 2) / destiny_width

    # origin_limit_factor = 1 / (2 * np.sin(np.pi / 4))
    # origin_col_size = source_width // 2

    # Define the destiny array
    destiny_array = np.zeros((destiny_height, destiny_width, 3), np.core.uint8)

    for row in prange(destiny_height):

        destiny_row_distance = int(np.round(row - destiny_center_row))
        destiny_row_f_distance = destiny_row_distance / destiny_dpi
        if destiny_row_f_distance < -1:
            destiny_row_distance = -1
        destiny_row_theta = inverse_destiny_function(destiny_row_f_distance)

        source_center_distance = source_function(destiny_row_theta + degrees_to_radians(90)) * source_dpi

        for column in prange(destiny_width):

            rotation = (angle_of_a_column * column) + np.pi  # np.pi is the starting angle

            angle_factor_x = np.cos(rotation)
            angle_factor_y = np.sin(rotation)

            origin_x = int(np.round(source_center.real + angle_factor_x * source_center_distance))
            origin_y = int(np.round(source_center.real - angle_factor_y * source_center_distance))
            if origin_x < 0 or origin_x > (source_width - 1):
                continue
            if origin_y < 0 or origin_y > (source_height - 1):
                continue

            destiny_array[row, column, :] = source[origin_y, origin_x, :]

    return destiny_array


# TODO Use this function from the camera.imaging submodule
@njit
def compute_quality_factor(function_source, function_destiny):
    """ Compute scaling of the image needed not to loose quality over different mapping functions """
    pixels_degree_o = function_source(degrees_to_radians(1))
    pixels_degree_d = function_destiny(degrees_to_radians(1))
    return pixels_degree_o / pixels_degree_d


@njit
def compute_destiny_size(source_size: complex, function_source, function_destiny):
    focal_distance = 1.0
    effective_source_size = source_size.real / 2.0

    # The angles defined bellow are defined from the center point in the capture device, so they are half the lens angle
    source_theta = degrees_to_radians(90)
    destiny_theta = degrees_to_radians(90)

    # Compute the relative sizes of the source image
    f_size_source = function_source(source_theta)
    f_size_destiny = function_destiny(destiny_theta)
    delta_f_size = f_size_destiny / f_size_source

    # Compute the relative sizes for the horizontal size
    a_destiny_degree = function_destiny(degrees_to_radians(1))
    a_degree_factor = a_destiny_degree / f_size_destiny

    height = int(np.round(effective_source_size * delta_f_size))
    width = int(np.round(height * a_degree_factor * 360))
    #  width = int(np.round(source_size.real * np.pi))

    destiny_size = complex(width, height)

    return destiny_size


