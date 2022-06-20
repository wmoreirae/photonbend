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

import sys
from pathlib import Path
from typing import Tuple

import click
import numpy as np
from PIL import Image

from photonbend.core.lens_image_type import LensImageType
from photonbend.core.sphere_image import SphereImage
from photonbend.projections.equirectangular import make_projection, compute_best_width
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
@click.option('--type', required=True, help='The type of the input image. ' + type_choices_help,
              type=type_choices)
@click.option('--lens', required=True, help='The lens type that was used on the input photo.',
              type=lens_choices)
@click.option('--fov', required=True, type=click.FLOAT,
              help='The lens field of view of the input photo in degrees. ' + double_type_fov_warning)
@click.option('--ssample', required=False, type=click.INT, help='The ammount of supersampling applied', default=1)
@click.option('-r', '--rotation', required=False, type=click.FLOAT, nargs=3, default=(0, 0, 0), help=rotation_help)
@click.argument('output_image', type=click.Path(exists=False))
def make_pano(input_image: click.Path, type: str, lens: str, fov: float, output_image: click.Path,
              ssample: int, rotation: Tuple[float, float, float]) -> None:
    """Make a panorama out of a photo.

    \b
    INPUT is the path to the source photo.
    OUTPUT is the desired path of the destiny panorama.
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

    source_lens = lens_types[lens]
    source_type = types_dict[type]
    source_fov = _check_fov(fov, source_type)

    try:
        with Image.open(input_image) as image:
            source_array = np.asarray(image)
    except IOError:
        print('Error: Input image could not be opened!')
        print('Exiting!')
        sys.exit(1)

    source_sphere = SphereImage(source_array, source_type, source_fov, source_lens)

    if rotation != (0, 0, 0):
        rotation_rad = list(map(degrees_to_radians, rotation))
        pitch, yaw, roll = rotation_rad
        source_sphere.set_rotation(pitch, roll, yaw)

    width = compute_best_width(source_sphere)

    destiny_arr = make_projection(source_sphere, 0, width)
    destiny_image = Image.fromarray(destiny_arr)
    print('Finished!')

    try:
        destiny_image.save(output_image)
    except IOError:
        print('Could not save to the specified location!')
        print('Exiting!')
        print('Exiting!')
        sys.exit(1)
