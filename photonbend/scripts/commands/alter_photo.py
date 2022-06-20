#  Copyright (c) 2022. Edson Moreira
#
#  Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
#  documentation files (the "Software"), to deal in the Software without restriction, including without limitation
#  the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and
#  to permit persons to whom the Software is furnished to do so, subject to the following conditions:
#
#  The above copyright notice and this permission notice shall be included in all copies or substantial portions of
#  the Software.
#
#  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING
#  BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
#  NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
#  DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
#  OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#
#  Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
#  documentation files (the "Software"), to deal in the Software without restriction, including without limitation
#  the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and
#  to permit persons to whom the Software is furnished to do so, subject to the following conditions:
#
#
import sys
from pathlib import Path
from typing import Tuple

import click
import numpy as np
from PIL import Image

from photonbend.core.lens_image_type import LensImageType
from photonbend.core.sphere_image import SphereImage
from photonbend.lens import equisolid, rectilinear, equidistant, \
    orthographic, stereographic
from photonbend.utils import degrees_to_radians
from .shared import lens_choices, type_choices, type_choices_help, double_type_fov_warning, rotation_help


def _check_fov(fov: float, image_type: LensImageType):
    if image_type is LensImageType.DOUBLE_INSCRIBED and fov < 180:
        raise ValueError("The fov of a double image can't be smaller than 180 degrees.")
    if fov > 360:
        raise ValueError("The fov of an image can't be higher than 360 degrees.")
    r_fov = degrees_to_radians(fov)
    return r_fov


def check_output(output: Path):
    out = Path(output)
    if not (out.suffix.lower() in ['.jpg', '.jpeg', '.png']):
        print("The desired output image should be a JPG or PNG file.")
        print("Provide an output filename ending in either JPG, JPEG or PNG (case insensitive)")
        print("Exiting!")
        sys.exit(1)
    if out.exists():
        while True:
            ans = input("File already exists. Overwrite? (y/n) ")
            if ans in ['y', 'n']:
                break
        if ans == 'n':
            print('Exiting!')
            sys.exit(0)
    return out


@click.argument('input_image', type=click.Path(exists=True))
@click.option('--itype', required=True, help='The type of the input image. ' + type_choices_help,
              type=type_choices)
@click.option('--ilens', required=True, help='The lens type that was used on the input photo.',
              type=lens_choices)
@click.option('--ifov', required=True, type=click.FLOAT,
              help='The lens field of view of the input photo in degrees. ' + double_type_fov_warning)
@click.option('--otype', required=True, help='The type of the output image.' + type_choices_help,
              type=type_choices)
@click.option('--olens', required=True,
              help='The lens type that was used on the input photo. ' + double_type_fov_warning,
              type=lens_choices)
@click.option('--ofov', required=True, type=click.FLOAT, help='The lens field of view of the output photo in degrees.')
@click.option('--ssample', required=False, type=click.INT, help='The ammount of supersampling applied (ss²)', default=1)
@click.argument('output_image', type=click.Path(exists=False))
@click.option('-r', '--rotation', required=False, type=click.FLOAT, nargs=3, default=(0, 0, 0), help=rotation_help)
def alter_photo(input_image: click.Path, itype: str, ilens: str, ifov: float, otype: str, olens: str, ofov: float,
                output_image: click.Path, ssample: int, rotation: Tuple[float, float, float]) -> None:
    """Change the the lens and FoV of a photo.

    \b
    INPUT is the path to the source photo.
    OUTPUT is the desired path of the destiny photo.
    """
    out = check_output(output_image)

    types_dict = {'inscribed': LensImageType.INSCRIBED,
                  'double': LensImageType.DOUBLE_INSCRIBED,
                  'cropped': LensImageType.CROPPED_CIRCLE,
                  'full': LensImageType.FULL_FRAME}

    lens_types = {'equidistant': equidistant,
                  'equisolid': equisolid,
                  'orthographic': orthographic,
                  'rectilinear': rectilinear,
                  'stereographic': stereographic}

    source_lens = lens_types[ilens]
    source_type = types_dict[itype]
    source_fov = _check_fov(ifov, source_type)

    destiny_lens = lens_types[olens]
    destiny_type = types_dict[otype]
    destiny_fov = _check_fov(ofov, destiny_type)

    try:
        with Image.open(input_image) as image:
            source_array = np.asarray(image)
    except IOError:
        print('Error: Input image could not be opened!')
        print('Exiting!')
        sys.exit(1)

    source_sphere = SphereImage(source_array, source_type, source_fov, source_lens)

    if (source_type is not destiny_type) and destiny_type is LensImageType.DOUBLE_INSCRIBED:
        y, x, c = source_array.shape
        destiny_array = np.zeros((y, x * 2, c), np.core.uint8)
    elif (source_type is not destiny_type) and source_type is LensImageType.DOUBLE_INSCRIBED:
        y, x, c = source_array.shape
        destiny_array = np.zeros((y, x // 2, c), np.core.uint8)
    else:
        destiny_array = np.zeros(source_array.shape, np.core.uint8)

    destiny_sphere = SphereImage(destiny_array, destiny_type, destiny_fov, destiny_lens)

    if rotation != (0, 0, 0):
        rotation_rad = list(map(degrees_to_radians, rotation))
        pitch, yaw, roll = rotation_rad
        source_sphere.set_rotation(pitch, roll, yaw)

    destiny_sphere.map_from_sphere_image(source_sphere, ssample)
    destiny_arr = destiny_sphere.get_image_array()
    destiny_image = Image.fromarray(destiny_arr)

    try:
        destiny_image.save(output_image)
    except IOError:
        print('Could not save to the specified location!')
        print('Exiting!')
        sys.exit(1)
