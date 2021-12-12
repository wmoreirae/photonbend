import numpy as np
from PIL import Image
from numba import njit, prange


from lightbend.utils import vector_magnitude, radians_to_degrees, degrees_to_radians, calculate_pixels_per_f_distance, \
    dpi_to_dpmm, \
    unit_vector, vector_to_focal_units, c_round


@njit(parallel=True)
def process_image(o_image_arr, lens_angle, function_o, ifunction_o, function_d, ifunction_d, fullframe):
    """
    Firstly, the program reads the position from the destination, calculates the theta angle of the destination
    and then it translates it to a position on the origin, reads such position and writes it on the position
    from the destination
    """

    # Some defaults
    f_distance = 1.0
    y_size_o, x_size_o = o_image_arr.shape[:2]

    center_o = complex(x_size_o, y_size_o) / 2 - complex(0.5, 0.5)

    # compute the pixels per focal distance
    pixels_pfd_origin = calculate_pixels_per_f_distance(center_o, lens_angle, f_distance, function_o)

    # Calculate the max angle of the lens
    if fullframe:
        max_vector_o = center_o
    else:
        max_vector_o = complex(center_o.real, 0)

    # Compute the focal distance of the destiny lens so it can have the same angle as the origin lens
    f_factor = vector_magnitude(max_vector_o) / (function_d(lens_angle / 2) * pixels_pfd_origin)
    f_distance_d = f_distance * f_factor

    # Compute the amount of pixels needed to not loose quality over different projections
    pixels_degree_o = function_o(degrees_to_radians(1)) * f_distance
    pixels_degree_d = function_d(degrees_to_radians(1)) * f_distance_d
    quality_factor = pixels_degree_o / pixels_degree_d
    ss_factor = 1  # TODO remove the hard coding of the super-sampling
    quality_factor *= ss_factor
    pixels_pfd_destiny = pixels_pfd_origin * quality_factor

    # Create the destiny array
    x_size_d = int(np.round(x_size_o * quality_factor))
    y_size_d = int(np.round(y_size_o * quality_factor))

    center_d = complex(x_size_d, y_size_d) / 2 - complex(0.5, 0.5)
    destiny_array = np.zeros((y_size_d, x_size_d, 3), 'uint8')

    for row in prange(y_size_d):
        for column in range(x_size_d):

            position_d = complex(column, row)
            position_vector = position_d - center_d
            unit_position_vector = unit_vector(position_vector)
            projection_magnitude = vector_to_focal_units(position_vector, f_distance_d, pixels_pfd_destiny)

            theta = ifunction_d(projection_magnitude)

            projection_magnitude_o = function_o(theta) * f_distance * pixels_pfd_origin
            position_o = center_o + (unit_position_vector * projection_magnitude_o)

            column_o, row_o = c_round(position_o)

            if 0 <= column_o < x_size_o and 0 <= row_o < y_size_o:
                destiny_array[row, column, :] = o_image_arr[row_o, column_o, :]

    return destiny_array

