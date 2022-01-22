import numpy as np
from PIL import Image
from numba import njit, prange

from lightbend.utils import vector_magnitude, degrees_to_radians, calculate_pixels_per_f_distance, \
    unit_vector, vector_to_focal_units, c_round

# Some default values
_half_pixel_vector = complex(0.5, 0.5)
_source_focal_distance = 1.0


@njit
def change_lens(src_image_arr, lens_angle, source_function, source_inverse_function, destiny_function,
                destiny_inverse_function,
                source_cropped, crop):
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

    max_vector_o = source_center
    destiny_focal_distance = compute_destiny_focal_distance(destiny_function, lens_angle, max_vector_o,
                                                            source_fd_pixels)

    quality_factor = compute_quality_factor(destiny_focal_distance, source_function, destiny_function)
    destiny_fd_pixels = source_fd_pixels * quality_factor

    super_sampling_factor = 3

    frame_size = compute_frame_size(source_size, source_fd_pixels, source_inverse_function, destiny_function,
                                    crop=crop)

    # Create the destiny array
    destiny_size = np.round(frame_size * quality_factor)
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

            # Calculate the pixel mean and assign it!
            ss_mean_1st = np.mean(ss_matrix[:, :, 0])
            ss_mean_2nd = np.mean(ss_matrix[:, :, 1])
            ss_mean_3rd = np.mean(ss_matrix[:, :, 2])
            destiny_array[row, column, 0] = int(np.round(ss_mean_1st))
            destiny_array[row, column, 1] = int(np.round(ss_mean_2nd))
            destiny_array[row, column, 2] = int(np.round(ss_mean_3rd))
    return destiny_array


@njit
def compute_frame_size(source_size, source_fd_pixels, source_inverse_function, destiny_function, crop: bool):
    """Used to compute the correct frame size taking into consideration the destiny function and the crop parameter

    This function takes the source size and measures the destination of takes 3 special points:
    - A vertex
    - The bottom side midpoint
    - The right side midpoint

    Using those points, it calculates their destiny coordinates on the new image and either crops or enlarge the
    are of the image to accommodate them, according to the crop parameter.

    Obs.: The focal distance for this function is assumed to be 1 (One). As such it can be ignored for all intents
    and purposes.
    """

    # Acquires each position on the source image.
    vertex = source_size / 2 / source_fd_pixels
    mid_h = complex(0, vertex.imag)
    mid_v = complex(vertex.real, 0)

    # Produce vectors that indicate the angles of each pair of coordinates.
    vertex_unit = unit_vector(vertex)
    mid_h_unit = unit_vector(mid_h)
    mid_v_unit = unit_vector(mid_v)

    # Produce new coordinates (based on focal distance, not pixels).
    p_vertex = vertex_unit * destiny_function(source_inverse_function(vector_magnitude(vertex)))
    p_mid_h = mid_h_unit * destiny_function(source_inverse_function(vector_magnitude(mid_h)))
    p_mid_v = mid_v_unit * destiny_function(source_inverse_function(vector_magnitude(mid_v)))

    # Takes max and min to allow comparisons and cropping.
    factor_h_min = min([p_vertex.real,
                        p_mid_v.real])

    factor_h_max = max([p_vertex.real,
                        p_mid_v.real])

    factor_v_min = min([p_vertex.imag,
                        p_mid_h.imag])

    factor_v_max = max([p_vertex.imag,
                        p_mid_h.imag])

    # Creates binary vectors to ease the calculations.
    factor_max = np.array([factor_h_max, factor_v_max])
    factor_min = np.array([factor_h_min, factor_v_min])
    source_size_arr = np.array([source_size.real, source_size.imag])

    # Sets a default value.
    c_factor_arr = np.array([1.0, 1.0])

    if crop:
        c_factor_arr = factor_min / factor_max
    elif factor_max[0] > p_vertex.real:
        c_factor_arr = factor_max / factor_min

    frame_size_arr = source_size_arr * c_factor_arr
    frame_size = complex(frame_size_arr[0], frame_size_arr[1])
    return frame_size


@njit
def compute_destiny_focal_distance(function_destiny, lens_angle, max_vector_source, pixels_fd_source):
    """Compute the focal distance of the destiny lens so it can have the same angle as the origin lens"""
    # TODO change this function to accept images with barrel distortion
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
