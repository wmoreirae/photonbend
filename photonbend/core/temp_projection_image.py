#   Copyright (c) 2022. Edson Moreira
#  #
#   Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
#   documentation files (the "Software"), to deal in the Software without restriction, including without limitation
#    the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and
#    to permit persons to whom the Software is furnished to do so, subject to the following conditions:
#  #
#   The above copyright notice and this permission notice shall be included in all copies or substantial portions of
#   the Software.
#  #
#   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING
#   BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
#   NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
#   DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
#   OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

import numpy as np
from PIL import Image
from typing import Callable, Tuple
from .utils import panorama_to_photo_size


def photo_best_size(panorama_width: int, lens_function: Callable[[float], float]):  # Use panorama_to_photo_size
    half_pi_f_radius = lens_function(np.pi / 2)
    pi_f_radius = lens_function(np.pi)
    f_factor = pi_f_radius / half_pi_f_radius

    pano_half_pi_diameter = panorama_width / np.pi
    photo_diameter = int(np.ceil(pano_half_pi_diameter * f_factor))
    return (photo_diameter,) * 2


def equisolid(theta: float) -> float:
    return 2 * np.sin(theta / 2)


def equisolid_inverse(length: float) -> float:
    return 2 * np.arcsin(length / 2)


def equidistant(theta):
    return theta


@np.vectorize
def mkcomplex(x, y):
    return complex(x, y)


# Begin the algorithm for lens projection (makes a lens projection out of a panorama
original_image = Image.open("View_From_The_Deck_6k.jpg")

o_array = np.asarray(original_image)

o_size = original_image.size
o_width, o_height = o_size

photo_size = photo_best_size(o_width, equidistant)

side_size = photo_size[0]

# making of the mesh
x_axis_range = np.linspace(-side_size / 2 + 0.5, side_size / 2 - 0.5, num=side_size)
y_axis_range = np.linspace(side_size / 2 - 0.5, -side_size / 2 + 0.5, num=side_size)
mesh_y, mesh_x = np.meshgrid(y_axis_range, x_axis_range, sparse=True, indexing='ij')

# using equidistant lens function
distance_mesh = np.sqrt(mesh_x ** 2 + mesh_y ** 2)
dpi_factor = side_size / 2 / equidistant(np.pi)
latitude = distance_mesh / dpi_factor
latitude[latitude > np.pi] = 0

longitude = np.log(mkcomplex(mesh_x, mesh_y)).imag

direct_coordinates = mkcomplex(latitude * (o_height/np.pi), (longitude * (o_width/ (2 * np.pi)))%o_width)

# make new array and image out of a coordinate map
new_image_array = o_array[
    direct_coordinates.real.astype(int),
    direct_coordinates.imag.astype(int),]
new_image_array[latitude == 0] = 0, 0 ,0
new_image = Image.fromarray(new_image_array)
new_image.save("Equidistant.jpg")