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
from typing import Tuple, List

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
)
from photonbend.core.projection import CameraImage, PanoramaImage
from photonbend.core.rotation import Rotation


@click.argument("input_image", type=click.Path(exists=True, path_type=Path))
@click.option(
    "--type",
    required=True,
    help="The type of the input image. " + type_choices_help,
    type=type_choices,
)
@click.option(
    "--lens",
    required=True,
    help="The lens type that was used on the input photo.",
    type=lens_choices,
)
@click.option(
    "--fov",
    required=True,
    type=click.FLOAT,
    help="The lens field of view of the input photo in degrees. "
    + double_type_fov_warning,
)
@click.option(
    "--ssample",
    required=False,
    type=click.INT,
    help="The ammount of supersampling applied",
    default=1,
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
@click.argument("output_image", type=click.Path(exists=False, path_type=Path))
def make_pano(
    input_image: Path,
    type: CamImgTypeStr,
    lens: CamLensStr,
    fov: float,
    output_image: Path,
    ssample: int,
    rotation: List[Tuple[float, float, float]],
) -> None:
    """Make a panorama out of a photo.

    \b
    INPUT is the path to the source photo.
    OUTPUT is the desired path of the destiny panorama.
    """
    out = _verify_output_path(output_image)

    # Opens the image or finish the application if there is no image
    source_array = _open_image(input_image)
    source_type = _process_image_type(type)
    source_lens = _process_lens(lens)
    source_magnitude = _calculate_magnitude(source_type, source_array.shape)
    source_fov = _process_fov(fov, source_type)
    source_image = CameraImage(
        source_array, source_fov, source_lens, magnitude=source_magnitude
    )
    s_height, s_width, _ = source_array.shape

    destiny_shape = (s_height, int(s_width * 2), 3)
    destiny_array = np.zeros(destiny_shape, np.uint8)
    destiny_image = PanoramaImage(destiny_array)
    destiny_map = destiny_image.get_coordinate_map()

    for rot in rotation:
        rad_rotation = tuple(map(to_radians, rot))
        rotation_transform = Rotation(*rad_rotation)
        destiny_map = rotation_transform.rotate_coordinate_map(destiny_map)

    mapped_array = source_image.process_coordinate_map(destiny_map)
    mapped_image = Image.fromarray(mapped_array)

    try:
        mapped_image.save(out)
    except IOError:
        print("Could not save to the specified location!")
        print("Exiting!")
        sys.exit(1)
