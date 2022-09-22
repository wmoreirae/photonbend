#  Copyright (c) 2022. Edson Moreira
#
#  Permission is hereby granted, free of charge, to any person obtaining a copy
#  of this software and associated documentation files (the "Software"), to deal
#  in the Software without restriction, including without limitation the rights
#  to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
#  copies of the Software, and to permit persons to whom the Software is
#  furnished to do so, subject to the following conditions:
#
#  The above copyright notice and this permission notice shall be included in
#  all copies or substantial portions of the Software.
#
#  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
#  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
#  OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
#  SOFTWARE.


import sys
from pathlib import Path
from typing import Tuple, List, Optional

import click
import numpy as np
from PIL import Image

from photonbend.utils import to_radians
from . import (
    _verify_output_path,
    _calculate_magnitude,
    _process_image_type,
    _process_lens,
    _open_image,
    CamImgTypeStr,
    CamLensStr,
    lens_choices,
    type_choices,
    type_choices_help,
    double_type_fov_warning,
    rotation_help,
    _process_fov,
    _get_camera,
    _calculate_destiny_size,
)
from photonbend.core.projection import PanoramaImage
from photonbend.core.rotation import Rotation


@click.argument("input_image", type=click.Path(exists=True, path_type=Path))
@click.option(
    "--type",
    "otype",
    required=True,
    help="The type of the output image. " + type_choices_help,
    type=type_choices,
)
@click.option(
    "--lens",
    required=True,
    help="The lens type to be used on the output photo.",
    type=lens_choices,
)
@click.option(
    "--fov",
    required=True,
    type=click.FLOAT,
    help="The lens field of view of the output photo in degrees. "
    + double_type_fov_warning,
)
@click.option(
    "-r",
    "--rotation",
    required=False,
    type=click.FLOAT,
    nargs=3,
    default=[],
    help=rotation_help,
    multiple=True,
)
@click.option(
    "-s",
    "--size",
    required=False,
    type=click.INT,
    default=None,
    help="The vertical size of the destiny image",
)
@click.argument("output_image", type=click.Path(exists=False, path_type=Path))
def make_photo(
    input_image: Path,
    otype: CamImgTypeStr,
    lens: CamLensStr,
    fov: float,
    output_image: Path,
    rotation: List[Tuple[float, float, float]],
    size: Optional[int],
) -> None:
    """Make a photo out of a panorama.

    \b
    INPUT is the path to the source panorama.
    OUTPUT is the desired path of the destiny photo.
    """
    out = _verify_output_path(output_image)

    # Opens the image or finish the application if there is no image
    source_array = _open_image(input_image)
    source_image = PanoramaImage(source_array)
    s_height, s_width, _ = source_array.shape

    destiny_type = _process_image_type(otype)
    destiny_shape = _calculate_destiny_size(destiny_type, source_array, height=size)
    destiny_lens = _process_lens(lens)

    destiny_magnitude = _calculate_magnitude(destiny_type, destiny_shape)
    destiny_fov = _process_fov(fov, destiny_type)
    destiny_image = _get_camera(destiny_type)(
        np.zeros(destiny_shape, np.int8),
        destiny_fov,
        destiny_lens,
        magnitude=destiny_magnitude,
    )
    destiny_map = destiny_image.get_coordinate_map()

    for rot in rotation:
        rad_rotation = tuple(map(to_radians, rot))
        rotation_transform = Rotation(*rad_rotation)
        destiny_map = rotation_transform.rotate_coordinate_map(destiny_map)

    mapped_array = source_image.process_coordinate_map(destiny_map)
    mapped_image = Image.fromarray(np.copy(mapped_array))

    try:
        mapped_image.save(out)
    except IOError:
        print("Could not save to the specified location!")
        print("Exiting!")
        sys.exit(1)
