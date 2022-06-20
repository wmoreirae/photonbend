#  Copyright (c) 2022. Edson Moreira
#
#  Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
#  documentation files (the "Software"), to deal in the Software without restriction, including without limitation
#  the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and
#  to permit persons to whom the Software is furnished to do so, subject to the following conditions:
#
#  The above copyright notice and this permission notice shall be included in all copies or substantial portions
#  of the Software.
#
#  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING
#  BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
#  NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
#  DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
#  OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

import numpy as np
from numba import njit, float64, cfunc, bool_
from photonbend.utils import degrees_to_radians


@cfunc(float64(float64))
def _rectilinear_inverse(projection_in_focal_distance_units):
    theta = np.arctan(projection_in_focal_distance_units)
    return theta


@cfunc(float64(float64))
def _rectilinear(theta):
    """Mapping that uses the angle tangent

    As it uses the angle tangent, it should not be used with lens angles closing on 180 degrees.
    It is best to limit its use to lens angles of at most 165 degrees.
    As the angle presented is halved, it should not see a theta angle larger than 82.5 degrees.
    """
    if theta < 0:
        raise ValueError('The angle theta cannot be negative')
    if theta > degrees_to_radians(89):
        raise ValueError('The rectilinear function was not made to handle FOV larger than 179 degrees')
    return np.tan(theta)


@cfunc(float64(float64))
def _stereographic_inverse(projection_in_focal_distance_units):
    half_tan_theta = projection_in_focal_distance_units / 2
    half_theta = np.arctan(half_tan_theta)
    theta = 2 * half_theta
    return theta


@cfunc(float64(float64))
def _stereographic(theta):
    half_theta = theta / 2
    half_projection = np.tan(half_theta)
    projection = 2 * half_projection
    return projection


@cfunc(float64(float64))
def _equidistant_inverse(projection_in_focal_distance_units):
    return projection_in_focal_distance_units


@cfunc(float64(float64))
def _equidistant(theta):
    return theta


@cfunc(float64(float64))
def _equisolid_inverse(projection_in_focal_distance_units):
    half_sin_theta = projection_in_focal_distance_units / 2
    half_theta = np.arcsin(half_sin_theta)
    theta = 2 * half_theta
    return theta


@cfunc(float64(float64))
def _equisolid(theta):
    half_theta = theta / 2
    half_projection = np.sin(half_theta)
    projection = 2 * half_projection
    return projection


@cfunc(float64(float64))
def _orthographic_inverse(projection_in_focal_distance_units):
    theta = np.arcsin(projection_in_focal_distance_units)
    return theta


@cfunc(float64(float64))
def _orthographic(theta):
    projection = np.sin(theta)
    return projection


@cfunc(float64(float64))
def _thoby_inverse(projection_in_focal_distance_units):
    k1 = 1.47
    k2 = 0.713
    theta = np.arcsin(projection_in_focal_distance_units / k1) / k2

    return theta


@cfunc(float64(float64))
def _thoby(theta):
    k1 = 1.47
    k2 = 0.713
    projection = k1 * np.sin(k2 * theta)
    return projection


# Begin the exported functions

@cfunc(float64(float64, bool_))
def rectilinear(angle_or_projection, inverse=False):
    if not inverse:
        return _rectilinear(angle_or_projection)
    return _rectilinear_inverse(angle_or_projection)


@cfunc(float64(float64, bool_))
def equisolid(angle_or_projection, inverse=False):
    if not inverse:
        return _equisolid(angle_or_projection)
    return _equisolid_inverse(angle_or_projection)


@cfunc(float64(float64, bool_))
def equidistant(angle_or_projection, inverse=False):
    if not inverse:
        return _equidistant(angle_or_projection)
    return _equidistant_inverse(angle_or_projection)


@cfunc(float64(float64, bool_))
def orthographic(angle_or_projection, inverse=False):
    if not inverse:
        return _orthographic(angle_or_projection)
    return _orthographic_inverse(angle_or_projection)


@cfunc(float64(float64, bool_))
def stereographic(angle_or_projection, inverse=False):
    if not inverse:
        return _stereographic(angle_or_projection)
    return _stereographic_inverse(angle_or_projection)


@cfunc(float64(float64, bool_))
def thoby(angle_or_projection, inverse=False):
    if not inverse:
        return _thoby(angle_or_projection)
    return _thoby_inverse(angle_or_projection)

