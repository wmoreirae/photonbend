import numpy as np
from numba import njit, prange

# TODO fix this function to integrate well with the rest of the library
@njit(parallel=True)
def make_pano(source):
    factor = np.sqrt(2)

    dest_y_size = int(np.round(factor * source.size[0] / 2))  # 90º
    dest_x_size = dest_y_size * 4  # 360º

    # make horizontal size always even
    if dest_x_size % 2 != 0:
        dest_x_size += 1

    full_radius = dest_x_size / 2 / np.pi
    origin = (source.size[0] - 1) / 2
    origin_col_size = source.size[0] / 2
    dest_center_row = (dest_y_size) // 2
    dest_center_col = (dest_x_size) // 2
    origin_limit_factor = 1 / (2 * np.sin(np.pi / 4))

    angle_per_row = np.pi / 2 / dest_y_size


@njit(parallel=True)
def adjust_elipsis():
    im_dest = np.zeros((dest_y_size, dest_x_size, 3), 'uint8')

    for row in prange(dest_y_size):
        # print(f'row: {row}')

        row_angle = (row * angle_per_row) + (angle_per_row / 2)

        # Calculate data necessary for the columns
        radius_destiny = np.sin(row_angle) * full_radius
        perimeter = int(np.round(radius_destiny * np.pi * 2))

        # makes perimeter always even
        if perimeter % 2 != 0:
            perimeter = perimeter + 1

        half_perimeter = perimeter // 2

        origin_distance_y = 2 * np.sin(row_angle / 2) * origin_limit_factor * (origin_col_size - 1)

        # determine upper and lower bound columns for the current row
        start_col = dest_center_col - half_perimeter
        end_col = dest_center_col + half_perimeter

        for col in prange(start_col, end_col):
            col_angle = ((col - dest_center_col) * (np.pi / half_perimeter))
            # print(col_angle)

            angle_factor_x = np.cos(col_angle)
            angle_factor_y = np.sin(col_angle)
            origin_x = int(np.round(origin + angle_factor_x * origin_distance_y))
            origin_y = int(np.round(origin - angle_factor_y * origin_distance_y))

            im_dest[row, col, :] = imr_arr[origin_y, origin_x, :]

    return im_dest


imr_elipsis = Image.fromarray(adjust_elipsis())