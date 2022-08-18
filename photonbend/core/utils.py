import numpy as np
from typing import Callable, Tuple


def _panorama_to_photo_size_horizontal(panorama_width: int, lens_function: Callable[[float], float]):
    half_pi_f_radius = lens_function(np.pi / 2)
    pi_f_radius = lens_function(np.pi)
    f_factor = pi_f_radius / half_pi_f_radius

    pano_half_pi_diameter = panorama_width / np.pi
    photo_diameter = int(np.ceil(pano_half_pi_diameter * f_factor))
    return (photo_diameter,) * 2


def _panorama_to_photo_size_vertical(panorama_height: int, lens_function: Callable[[float], float]):
    half_pi_f_radius = lens_function(np.pi / 2)
    pi_f_radius = lens_function(np.pi)
    f_factor = pi_f_radius / half_pi_f_radius

    small_side_factor = 1.0 / (1.0 - f_factor if f_factor > 0.5 else f_factor)
    photo_diameter = int(np.ceil(panorama_height * small_side_factor))
    return (photo_diameter,) * 2


def panorama_to_photo_size(panorama_size: Tuple[int, int], lens_function: Callable[[float], float],
                           preserve_vertical: bool = False):
    width, height = panorama_size
    assert width == 2 * height, "Equirectangular panoramas should have width and height in a 2:1 proportion"

    if preserve_vertical:
        return _panorama_to_photo_size_vertical(height, lens_function=lens_function)
    return _panorama_to_photo_size_horizontal(width, lens_function=lens_function)