#   Copyright (c) 2022. Edson Moreira
#
#   Permission is hereby granted, free of charge, to any person obtaining a copy
#   of this software and associated documentation files (the "Software"), to deal
#   in the Software without restriction, including without limitation the rights
#   to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
#   copies of the Software, and to permit persons to whom the Software is
#   furnished to do so, subject to the following conditions:
#
#   The above copyright notice and this permission notice shall be included in
#   all copies or substantial portions of the Software.
#
#   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#   IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#   FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
#   AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#   LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
#   OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
#   SOFTWARE.

__doc__ = "Some simple utility functions are available here."

from typing import Tuple, Callable
import numpy as np


def to_radians(degrees: float) -> float:
    """Convert degrees to radians

    Args:
        degrees: A float representing an angle in degrees

    Returns:
        A float representing the same angle in radians.
    """

    return degrees / 180 * np.pi


def to_degrees(radians: float) -> float:
    """Convert radians to degrees

    Args:
        radians: A float representing an angle in radians

    Returns:
        A float representing the same angle in degrees.
    """

    return radians / np.pi * 180.0


def _panorama_to_photo_size_horizontal(
    panorama_width: int, lens_function: Callable[[float], float]
) -> Tuple[float, float]:
    """Helper functions - not stable"""

    half_pi_f_radius = lens_function(np.pi / 2)
    pi_f_radius = lens_function(np.pi)
    f_factor = pi_f_radius / half_pi_f_radius

    pano_half_pi_diameter = panorama_width / np.pi
    photo_diameter = int(np.ceil(pano_half_pi_diameter * f_factor))
    return (photo_diameter,) * 2


def _panorama_to_photo_size_vertical(
    panorama_height: int, lens_function: Callable[[float], float]
) -> Tuple[float, float]:
    """Helper function - not stable"""

    half_pi_f_radius = lens_function(np.pi / 2)
    pi_f_radius = lens_function(np.pi)
    f_factor = pi_f_radius / half_pi_f_radius

    small_side_factor = 1.0 / (1.0 - f_factor if f_factor > 0.5 else f_factor)
    photo_diameter = abs(int(np.ceil(panorama_height * small_side_factor)))
    return (photo_diameter,) * 2


def calculate_size_panorama_to_photo(
    panorama_size: Tuple[int, int],
    lens_function: Callable[[float], float],
    preserve_vertical_resolution: bool = False,
) -> Tuple[float, float]:
    """Computes the size for converting a panorama to a camera image.

    Uses the panorama dimensions and lens function to compute the necessary
    radius of of an inscribed photo in order to preserve the panorama
    pixel information.

    Args:
        panorama_size: A tuple(int, int) storing respectively the
            panorama width and height in pixels.
        lens_function: A function that acts as the lens of a camera.
        preserve_vertical_resolution: A boolean that states whether this
            function should worry about preserving vertical resolution.

    Returns:
        A tuple containing the appropriate size (width, height) in
        pixels for the conversion of the panorama to a photo-line image
        for maintaining similar level of image detail.
    """

    width, height = panorama_size
    assert (
        width == 2 * height
    ), "Equirectangular panoramas should have width and height in a 2:1 proportion"

    photo_size = _panorama_to_photo_size_horizontal(width, lens_function=lens_function)

    if preserve_vertical_resolution:
        v_photo_size = _panorama_to_photo_size_vertical(
            height, lens_function=lens_function
        )
        if v_photo_size > photo_size:
            return v_photo_size
    return photo_size


__all__ = ["to_radians", "to_degrees", "calculate_size_panorama_to_photo"]
