import numpy as np
from PIL import Image
from numba import njit, prange

from lightbend.utils import vector_magnitude, radians_to_degrees, degrees_to_radians, calculate_pixels_per_f_distance, \
    dpi_to_dpmm, \
    unit_vector, vector_to_focal_units, c_round

# Some default values
_half_pixel_vector = complex(0.5, 0.5)
_source_focal_distance = 1.0


@njit
def change_lens(src_image_arr, lens_angle, source_function, source_inverse_function, destiny_function,
                destiny_inverse_function,
                fullframe):
    """
    Firstly, the program reads the position from the destination, calculates the theta angle of the destination
    and then it translates it to a position on the origin, reads such position and writes it on the position
    from the destination
    """

    # Prepare some data from the inputs
    h, w = src_image_arr.shape[:2]
    source_size = complex(w, h)
    source_center = source_size / 2 - _half_pixel_vector
    source_fd_pixels = calculate_pixels_per_f_distance(source_center, lens_angle, _source_focal_distance,
                                                       source_function)

    # Calculate the max angle of the lens
    if fullframe:
        max_vector_o = source_center
    else:
        max_vector_o = complex(source_center.real, 0)

    destiny_focal_distance = compute_destiny_focal_distance(destiny_function, lens_angle, max_vector_o,
                                                            source_fd_pixels)

    quality_factor = compute_quality_factor(destiny_focal_distance, source_function, destiny_function)
    destiny_fd_pixels = source_fd_pixels * quality_factor

    super_sampling_factor = 3

    # Create the destiny array
    destiny_size = np.round(source_size * quality_factor)
    destiny_array = light_adjustment(src_image_arr, source_fd_pixels, source_function, destiny_size, destiny_fd_pixels,
                                     destiny_focal_distance, destiny_inverse_function, super_sampling_factor)

    return destiny_array


@njit(parallel=True)
def light_adjustment(src_image_arr, source_fd_pixels, source_function, destiny_size, destiny_fd_pixels,
                     destiny_focal_distance, destiny_ifunction, super_sampling_factor):
    h, w = src_image_arr.shape[:2]
    source_size = complex(w, h)
    source_center = source_size / 2 - _half_pixel_vector
    source_width, source_height = c_round(source_size)

    destiny_width, destiny_height = c_round(destiny_size)
    destiny_center = destiny_size / 2 - _half_pixel_vector
    destiny_array = np.zeros((destiny_height, destiny_width, 3), np.core.uint8)

    # SUPER SAMPLING DATA
    ss_pixel_center = complex((super_sampling_factor - 1) / 2, (super_sampling_factor - 1) / 2)
    ss_subpixel_distance = 1 / super_sampling_factor

    for row in prange(destiny_height):
        for column in prange(destiny_width):

            # SUPER SAMPLING ROUNDS
            ss_matrix = np.zeros((super_sampling_factor, super_sampling_factor, 3), np.core.uint8)
            for ss_row in prange(super_sampling_factor):
                for ss_column in prange(super_sampling_factor):
                    ss_position = (complex(ss_column, ss_row) - ss_pixel_center) * ss_subpixel_distance

                    position_destiny = complex(column, row) + ss_position
                    position_vector = position_destiny - destiny_center
                    unit_position_vector = unit_vector(position_vector)
                    projection_magnitude = vector_to_focal_units(position_vector, destiny_focal_distance,
                                                                 destiny_fd_pixels)

                    theta = destiny_ifunction(projection_magnitude)

                    projection_magnitude_o = source_function(theta) * _source_focal_distance * source_fd_pixels
                    position_o = source_center + (unit_position_vector * projection_magnitude_o)

                    column_o, row_o = c_round(position_o)

                    if 0 <= column_o < source_width and 0 <= row_o < source_height:
                        ss_matrix[ss_row, ss_column, :] = src_image_arr[row_o, column_o, :]

            # if 0 <= column_o < source_width and 0 <= row_o < source_height:
            ss_mean_1st = np.mean(ss_matrix[:, :, 0])
            ss_mean_2nd = np.mean(ss_matrix[:, :, 1])
            ss_mean_3rd = np.mean(ss_matrix[:, :, 2])
            destiny_array[row, column, 0] = int(np.round(ss_mean_1st))
            destiny_array[row, column, 1] = int(np.round(ss_mean_2nd))
            destiny_array[row, column, 2] = int(np.round(ss_mean_3rd))
    return destiny_array


@njit
def compute_destiny_focal_distance(function_destiny, lens_angle, max_vector_source, pixels_fd_source):
    """Compute the focal distance of the destiny lens so it can have the same angle as the origin lens"""
    f_factor = vector_magnitude(max_vector_source) / (function_destiny(lens_angle / 2) * pixels_fd_source)
    f_distance_d = _source_focal_distance * f_factor
    return f_distance_d


@njit
def compute_quality_factor(focal_distance_destiny, function_source, function_destiny):
    """ Compute scaling of the image needed not to loose quality over different mapping functions """
    pixels_degree_o = function_source(degrees_to_radians(1)) * _source_focal_distance
    pixels_degree_d = function_destiny(degrees_to_radians(1)) * focal_distance_destiny
    quality_factor = pixels_degree_o / pixels_degree_d
    return quality_factor
