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
import click

lens_choices = click.Choice(['equidistant', 'equisolid', 'orthographic', 'rectilinear', 'stereographic'])
type_choices = click.Choice(['inscribed', 'double', 'cropped', 'full'])

type_choices_help = \
    """

    \b
    The choices are:
    - inscribed: The valid data is on a inscribed circle inscribed.
    - double: The valid data is on two inscribed side-by-side circles.
    - cropped: The valid data is on a inscribed circle, top-and-bottom cropped. 
    - full: The whole area of the image is valid data. 
    """

double_type_fov_warning = \
    """

    IMPORTANT: When dealing with double images, you should pass the FOV of a single circle and it must be larger than 180."""

rotation_help = \
    """
    The rotation that should be applied to the camera.
    This is a 3-valued parameter in the form <pitch yaw roll>
    """
