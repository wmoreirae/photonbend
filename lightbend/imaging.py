import numpy as np
from PIL import Image
from numba import njit, prange

from lightbend.utils import vector_magnitude, radians_to_degrees, degrees_to_radians, calculate_pixels_per_f_distance, \
    dpi_to_dpmm, \
    unit_vector, vector_to_focal_units, c_round

# TODO
"""
Split this function into a function that calculates the image sizes, focal distances and pixels por focal distances and
the one that process the pixel data
"""
@njit(parallel=True)
def process_image(src_image_arr, lens_angle, function_source, ifunction_source, function_destiny, ifunction_destiny,
                  fullframe):
    """
    Firstly, the program reads the position from the destination, calculates the theta angle of the destination
    and then it translates it to a position on the origin, reads such position and writes it on the position
    from the destination
    """

    # Some defaults
    focal_distance_src = 1.0
    height_source, width_source = src_image_arr.shape[:2]

    center_source = complex(width_source, height_source) / 2 - complex(0.5, 0.5)

    # compute the pixels per focal distance of the source image
    pixels_fd_source = calculate_pixels_per_f_distance(center_source, lens_angle, focal_distance_src, function_source)

    # Calculate the max angle of the lens
    if fullframe:
        max_vector_o = center_source
    else:
        max_vector_o = complex(center_source.real, 0)

    focal_distance_destiny = compute_f_distance_destiny(focal_distance_src, function_destiny, lens_angle, max_vector_o,
                                                     pixels_fd_source)

    quality_factor = compute_quality_factor(focal_distance_src, focal_distance_destiny, function_source, function_destiny)
    quality_factor = quality_factor
    pixels_fd_destiny = pixels_fd_source * quality_factor

    # Create the destiny array
    x_size_d = int(np.round(width_source * quality_factor))
    y_size_d = int(np.round(height_source * quality_factor))

    center_destiny = complex(x_size_d, y_size_d) / 2 - complex(0.5, 0.5)
    destiny_array = np.zeros((y_size_d, x_size_d, 3), 'uint8')

    ss_factor = 3
    center_ss_pixel = complex((ss_factor - 1) / 2, (ss_factor - 1) / 2)
    ss_subpixel_distance = 1 / ss_factor

    for row in prange(y_size_d):
        for column in prange(x_size_d):

            # SUPER SAMPLING ROUNDS
            ss_matrix = np.zeros((ss_factor, ss_factor, 3), 'uint8')
            for ss_row in prange(ss_factor):
                for ss_column in prange(ss_factor):
                    ss_position = (complex(ss_column, ss_row) - center_ss_pixel) * ss_subpixel_distance

                    position_destiny = complex(column, row) + ss_position
                    position_vector = position_destiny - center_destiny
                    unit_position_vector = unit_vector(position_vector)
                    projection_magnitude = vector_to_focal_units(position_vector, focal_distance_destiny, pixels_fd_destiny)

                    theta = ifunction_destiny(projection_magnitude)

                    projection_magnitude_o = function_source(theta) * focal_distance_src * pixels_fd_source
                    position_o = center_source + (unit_position_vector * projection_magnitude_o)

                    column_o, row_o = c_round(position_o)

                    if 0 <= column_o < width_source and 0 <= row_o < height_source:
                        ss_matrix[ss_row, ss_column, :] = src_image_arr[row_o, column_o, :]

            # if 0 <= column_o < width_source and 0 <= row_o < height_source:
            ss_mean_1st = np.mean(ss_matrix[:, :, 0])
            ss_mean_2nd = np.mean(ss_matrix[:, :, 1])
            ss_mean_3rd = np.mean(ss_matrix[:, :, 2])
            destiny_array[row, column, 0] = int(np.round(ss_mean_1st))
            destiny_array[row, column, 1] = int(np.round(ss_mean_2nd))
            destiny_array[row, column, 2] = int(np.round(ss_mean_3rd))

    return destiny_array


@njit
def compute_f_distance_destiny(focal_distance_source, function_destiny, lens_angle, max_vector_o, pixels_fd_source):
    """Compute the focal distance of the destiny lens so it can have the same angle as the origin lens"""
    f_factor = vector_magnitude(max_vector_o) / (function_destiny(lens_angle / 2) * pixels_fd_source)
    f_distance_d = focal_distance_source * f_factor
    return f_distance_d


@njit
def compute_quality_factor(focal_distance_source, focal_distance_destiny, function_origin, function_destiny):
    """ Compute scaling of the image needed not to loose quality over different mapping functions """
    pixels_degree_o = function_origin(degrees_to_radians(1)) * focal_distance_source
    pixels_degree_d = function_destiny(degrees_to_radians(1)) * focal_distance_destiny
    quality_factor = pixels_degree_o / pixels_degree_d
    return quality_factor
