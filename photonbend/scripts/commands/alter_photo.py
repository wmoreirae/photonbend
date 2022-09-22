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
from photonbend.core.projection import ProjectionImage
from photonbend.core.rotation import Rotation


@click.argument("input_image", type=click.Path(exists=True, path_type=Path))
@click.option(
    "--itype",
    required=True,
    help="The type of the input image. " + type_choices_help,
    type=type_choices,
)
@click.option(
    "--ilens",
    required=True,
    help="The lens type that was used on the input photo.",
    type=lens_choices,
)
@click.option(
    "--ifov",
    required=True,
    type=click.FLOAT,
    help="The lens field of view of the input photo in degrees. "
    + double_type_fov_warning,
)
@click.option(
    "--otype",
    required=True,
    help="The type of the output image." + type_choices_help,
    type=type_choices,
)
@click.option(
    "--olens",
    required=True,
    help="The lens type that was used on the input photo. " + double_type_fov_warning,
    type=lens_choices,
)
@click.option(
    "--ofov",
    required=True,
    type=click.FLOAT,
    help="The lens field of view of the output photo in degrees.",
)
@click.argument("output_image", type=click.Path(exists=False, path_type=Path))
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
def alter_photo(
    input_image: Path,
    itype: CamImgTypeStr,
    ilens: CamLensStr,
    ifov: float,
    otype: CamImgTypeStr,
    olens: CamLensStr,
    ofov: float,
    output_image: Path,
    rotation: List[Tuple[float, float, float]],
    size: Optional[int],
) -> None:
    """Change the the lens and FoV of a photo.

    \b
    INPUT is the path to the source photo.
    OUTPUT is the desired path of the destiny photo.
    """
    out = _verify_output_path(output_image)

    # Opens the image or finish the application if there is no image
    source_array = _open_image(input_image)
    source_type = _process_image_type(itype)
    source_lens = _process_lens(ilens)
    source_magnitude = _calculate_magnitude(source_type, source_array.shape)
    source_fov = _process_fov(ifov, source_type)
    source_image: ProjectionImage = _get_camera(source_type)(
        source_array, source_fov, source_lens, magnitude=source_magnitude
    )

    destiny_type = _process_image_type(otype)
    destiny_shape = _calculate_destiny_size(destiny_type, source_array, height=size)
    destiny_array = np.zeros(destiny_shape, np.uint8)
    destiny_lens = _process_lens(olens)
    destiny_magnitude = _calculate_magnitude(destiny_type, source_array.shape)
    destiny_fov = _process_fov(ofov, destiny_type)
    destiny_image: ProjectionImage = _get_camera(cam_img_type=destiny_type)(
        destiny_array, destiny_fov, destiny_lens, magnitude=destiny_magnitude
    )
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
