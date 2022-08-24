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
    photo_diameter = abs(int(np.ceil(panorama_height * small_side_factor)))
    return (photo_diameter,) * 2


def panorama_to_photo_size(panorama_size: Tuple[int, int], lens_function: Callable[[float], float],
                           preserve_vertical: bool = False):
    width, height = panorama_size
    assert width == 2 * height, "Equirectangular panoramas should have width and height in a 2:1 proportion"

    photo_size = _panorama_to_photo_size_horizontal(width, lens_function=lens_function)

    if preserve_vertical:
        v_photo_size = _panorama_to_photo_size_vertical(height, lens_function=lens_function)
        if v_photo_size > photo_size:
            return v_photo_size
    return photo_size

# New non-numba version
def make_complex(x, y):
    zx = x * 0
    zy = y * 0
    fx = x + zy
    fy = y + zx
    ans = np.concatenate([fx.reshape(*fx.shape, 1), fy.reshape(*fy.shape, 1)], 2).view(dtype=np.core.complex128).reshape(fx.shape[:2])
    return ans