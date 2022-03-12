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


import numba
import numpy as np
from PIL import Image

from lightbend.lens import equisolid, rectilinear, equidistant, \
    orthographic, stereographic
from lightbend.utils import degrees_to_radians

from lightbend.core.sphere_image import SphereImage, LensImage

import click





@click.command()
def make_pano():
    pass


@click.command()
def make_photo():
    pass


@click.command()
@click.option('-i', '--input', required=True, type=click.Path(exists=True))
@click.option('--ilens', required=True)
@click.option('--ifov', required=True, type=click.FLOAT)
@click.option('--olens', required=True)
@click.option('--ofov', required=True, type=click.FLOAT)
@click.argument('output')
def alter_photo(input, ifov, olens, ofov, output):
    pass

@click.group([alter_photo])
def main():
    pass