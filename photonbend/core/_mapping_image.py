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
import numpy.typing as npt

from .protocols import ProjectionImage
from ._shared import make_complex
from typing import Tuple, Callable, Union

ForwardReverseLensFunction = Callable[[Union[float, npt.NDArray[float]]], Union[float, npt.NDArray[float]]]
LensFunction = Tuple[ForwardReverseLensFunction, ForwardReverseLensFunction]


def make_mapping_map(coordinate_map: npt.NDArray[np.core.float64]) -> \
        npt.NDArray[np.core.int8]:
    """Converts a coordinate map to a color map

    Converts a coordinate to a RGB color map so that we can see the
    """
    rgb_range = 255.0

    invalid_map = coordinate_map[:, :, 2] != 0.0
    valid_map = np.logical_not(invalid_map)
    polar_map = coordinate_map[:, :, :2]

    polar_map[invalid_map] = 0

    # Distance
    distance = polar_map[:, :, 0]
    min_distance = np.min(distance[valid_map])
    max_distance = np.max(distance[valid_map])
    min_max_distance = max_distance - min_distance
    mm_factor = rgb_range / min_max_distance
    new_distance = distance.copy()
    new_distance[valid_map] -= min_distance
    new_distance[valid_map] *= mm_factor

    distance_map_8bits = np.round(new_distance).astype(np.core.uint8)

    # Direction
    unbalanced_position = polar_map[:, :, 1]
    d_factor = rgb_range / (np.pi * 2)
    position_map = d_factor * unbalanced_position
    position_map_8bits = np.round(position_map).astype(np.core.uint8)

    # TODO check if some overflow is happening
    invalid_map_8bits = (invalid_map.astype(np.core.uint8) * 255).astype(np.core.uint8)

    mapping_image = np.concatenate([np.expand_dims(distance_map_8bits, 2), np.expand_dims(position_map_8bits, 2),
                                    np.expand_dims(invalid_map_8bits, 2)], axis=2)

    return mapping_image
