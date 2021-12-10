import numpy as np
from numba import njit, prange

from utils import vector_magnitude, radians_to_degrees, degrees_to_radians, calculate_dpi, dpi_to_dpmm, \
    unit_vector, vector_to_focal_units, c_round


@njit
def process_image(o_image_arr, lens_angle, function_o, ifunction_o, function_d, ifunction_d, fullframe):
    # Some defaults
    f_distance = 50

    # calculate the correct destiny size
    y_size_o, x_size_o = o_image_arr.shape[:2]

    # define the origin center as complex number to make it easier to do the algebra later
    center_o = complex(x_size_o, y_size_o) / 2 - complex(0.5, 0.5)

    # compute the dpi
    dpi = calculate_dpi(center_o, lens_angle, f_distance, function_o)
    dpmm = dpi_to_dpmm(dpi)

    # Calculate the max angle of the lens
    if fullframe:
        max_vector_o = center_o
    else:
        max_vector_o = complex(center_o.real, 0)

    # Compute the focal distance of the destiny lens so it can have the same angle
    f_factor = vector_magnitude(max_vector_o) / (function_d(lens_angle / 2) * f_distance * dpmm)
    print(f_factor)
    f_distance_d = f_distance * f_factor

    # Create the destiny array
    x_size_d = x_size_o
    y_size_d = y_size_o
    center_d = complex(x_size_d, y_size_d) / 2 - complex(0.5, 0.5)
    destiny_array = np.zeros((y_size_d, x_size_d, 3), 'uint8')

    """
    Firstly, the program reads the position from the destination, calculates the theta angle of the destination
    and then it translates it to a position on the origin, reads such position and writes it on the position
    from the destination
    """

    for row in prange(y_size_d):
        for column in range(x_size_d):

            position_d = complex(column, row)
            position_vector = position_d - center_d
            unit_position_vector = unit_vector(position_vector)
            projection_magnitude = vector_to_focal_units(position_vector, f_distance_d, dpi)

            theta = ifunction_d(projection_magnitude)

            projection_magnitude_o = function_o(theta) * f_distance * dpmm
            position_o = center_o + (unit_position_vector * projection_magnitude_o)

            column_o, row_o = c_round(position_o)

            if 0 <= column_o < x_size_o and 0 <= row_o < y_size_o:
                destiny_array[row, column, :] = o_image_arr[row_o, column_o, :]

    return destiny_array