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
import numpy.typing as npt
import numba as nb

from typing import Callable, Union, Tuple
from photonbend.utils import degrees_to_radians

LensArgument = Union[float, npt.NDArray[float]]
ForwardReverseLensFunction = Callable[[LensArgument], LensArgument]
LensFunction = Tuple[ForwardReverseLensFunction, ForwardReverseLensFunction]


@nb.vectorize
def _rectilinear_inverse(projection_in_focal_distance_units: LensArgument) -> ForwardReverseLensFunction:
    theta = np.arctan(projection_in_focal_distance_units)
    return theta


@nb.vectorize
def _rectilinear(theta: LensArgument) -> LensArgument:
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


@nb.vectorize
def _stereographic_inverse(projection_in_focal_distance_units: LensArgument) -> LensArgument:
    half_tan_theta = projection_in_focal_distance_units / 2
    half_theta = np.arctan(half_tan_theta)
    theta = 2 * half_theta
    return theta


@nb.vectorize
def _stereographic(theta: LensArgument) -> LensArgument:
    half_theta = theta / 2
    half_projection = np.tan(half_theta)
    projection = 2 * half_projection
    return projection


@nb.vectorize
def _equidistant_inverse(projection_in_focal_distance_units: LensArgument) -> LensArgument:
    return projection_in_focal_distance_units


@nb.vectorize
def _equidistant(theta: LensArgument) -> LensArgument:
    return theta


@nb.vectorize
def _equisolid_inverse(projection_in_focal_distance_units: LensArgument) -> LensArgument:
    half_sin_theta = projection_in_focal_distance_units / 2
    half_theta = np.arcsin(half_sin_theta)
    theta = 2 * half_theta
    if np.isnan(theta):
        return 0.0
    return theta


@nb.vectorize
def _equisolid(theta: LensArgument) -> LensArgument:
    half_theta = theta / 2
    half_projection = np.sin(half_theta)
    projection = 2 * half_projection
    return projection


@nb.vectorize
def _orthographic_inverse(projection_in_focal_distance_units: LensArgument) -> LensArgument:
    theta = np.arcsin(projection_in_focal_distance_units)
    return theta


@nb.vectorize
def _orthographic(theta: LensArgument) -> LensArgument:
    projection = np.sin(theta)
    return projection


@nb.vectorize
def _thoby_inverse(projection_in_focal_distance_units: LensArgument) -> LensArgument:
    k1 = 1.47
    k2 = 0.713
    theta = np.arcsin(projection_in_focal_distance_units / k1) / k2

    return theta


@nb.vectorize
def _thoby(theta: LensArgument) -> LensArgument:
    k1 = 1.47
    k2 = 0.713
    projection = k1 * np.sin(k2 * theta)
    return projection


# Begin the exported functions


def rectilinear() -> LensFunction:
    return _rectilinear, _rectilinear_inverse


def equisolid() -> LensFunction:
    return _equisolid, _equisolid_inverse


def equidistant() -> LensFunction:
    return _equidistant, _equidistant_inverse


def orthographic() -> LensFunction:
    return _orthographic, _orthographic_inverse


def stereographic() -> LensFunction:
    return _stereographic, _stereographic_inverse


def thoby() -> LensFunction:
    return _thoby, _thoby_inverse
