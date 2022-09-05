

#   Copyright (c) 2022. Edson Moreira
#
#   Permission is hereby granted, free of charge, to any person obtaining a copy
#   of this software and associated documentation files (the "Software"), to deal
#   in the Software without restriction, including without limitation the rights
#   to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
#   copies of the Software, and to permit persons to whom the Software is
#   furnished to do so, subject to the following conditions:
#
#   The above copyright notice and this permission notice shall be included in
#   all copies or substantial portions of the Software.
#
#   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#   IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#   FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
#   AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#   LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
#   OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
#   SOFTWARE.

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
import warnings

from typing import Callable, Union
from photonbend.utils import to_radians

LensArgument = Union[float, npt.NDArray[float]]
ForwardReverseLensFunction = Callable[[LensArgument], LensArgument]


@dataclass
class Lens:
    """Represents a lens with both forward and reverse functions

    Attributes:
        forward_function (ForwardReverseLensFunction): A function that given
            an incidence angle in radians gives back a distance from the
            projection center in focal distance units.
            It must handle either a single float or an array of floats.
        reverse_function (ForwardReverseLensFunction): A function that
            given an distance from the projection center in focal
            distance units back the incidence angle in radians.
            It must handle either a single float or an array of floats.
    """

    forward_function: ForwardReverseLensFunction
    reverse_function: ForwardReverseLensFunction


# @nb.vectorize
def _rectilinear_inverse(
    projection_in_focal_distance_units: LensArgument,
) -> LensArgument:
    theta = np.arctan(projection_in_focal_distance_units)
    return theta


# TODO fix this function so it works with numpy arrays
def _rectilinear(theta: LensArgument) -> LensArgument:
    """Mapping that uses the angle tangent

    As it uses the angle tangent, it should not be used with lens angles closing on 180
    degrees. It is best to limit its use to lens angles of at most 165 degrees.
    As the angle presented is halved, it should not see a theta angle larger than 82.5
    degrees.

    Args:
        theta:

    """
    if theta < 0:
        raise ValueError("The angle theta cannot be negative")
    if theta > to_radians(89):
        raise ValueError(
            "The rectilinear function was not made to handle FOV larger than 179 degrees"
        )
    return np.tan(theta)


def _stereographic_inverse(projection_in_f_units: LensArgument) -> LensArgument:
    """Inverse stereographic function

    As an inverse lens function, it takes a distance from the center of
    the image in focal units and returns the incidence angle in radians
    that would produce such distance.

    Args:
         projection_in_f_units: either a float or numpy array containing
            floats representing the projection in focal distance units.

    Returns:
        A float or a numpy array of floats representing the angle or
            angles in radians.
    """

    half_tan_theta = projection_in_f_units / 2
    half_theta = np.arctan(half_tan_theta)
    theta = 2 * half_theta
    return theta


def _stereographic(theta: LensArgument) -> LensArgument:
    """The stereographic function

    As a lens function, it takes an incidence angle and returns a
    distance from the center of the image such angle would be projected
    in the final image.

    Args:
         theta: either a float or numpy array containing
            floats representing the incidence angle in radians.

    Returns:
        A float or a numpy array of floats representing the distance in
            focal units the angles would be projected.
    """
    half_theta = theta / 2
    half_projection = np.tan(half_theta)
    projection = 2 * half_projection
    return projection


def _equidistant_inverse(projection_in_f_units: LensArgument) -> LensArgument:
    """The inverse equidistant function

    As an inverse lens function, it takes a distance from the center of
    the image in focal units and returns the incidence angle in radians
    that would produce such distance.

    As an interesting fact the equidistant function and its inverse are
    one and the same.

    Args:
         projection_in_f_units: either a float or numpy array containing
            floats representing the projection in focal distance units.

    Returns:
        A float or a numpy array of floats representing the angle or
            angles in radians."""
    return projection_in_f_units


# @nb.vectorize
def _equidistant(theta: LensArgument) -> LensArgument:
    """The equidistant function

    As a lens function, it takes an incidence angle and returns a
    distance from the center of the image such angle would be projected
    in the final image.

    As an interesting fact the equidistant function and its inverse are
    one and the same.

    Args:
         theta: either a float or numpy array containing
            floats representing the incidence angle in radians.

    Returns:
        A float or a numpy array of floats representing the distance in
            focal units the angles would be projected.
    """
    return theta


# @nb.vectorize
def _equisolid_inverse(projection_in_f_units: LensArgument) -> LensArgument:
    """The inverse equisolid function

    As an inverse lens function, it takes a distance from the center of
    the image in focal units and returns the incidence angle in radians
    that would produce such distance.

    Args:
         projection_in_f_units: either a float or numpy array containing
            floats representing the projection in focal distance units.

    Returns:
        A float or a numpy array of floats representing the angle or
            angles in radians."""

    half_sin_theta = projection_in_f_units / 2
    with warnings.catch_warnings():
        # ignore warnings that will be thrown because of NaNs
        warnings.simplefilter("ignore")
        half_theta = np.arcsin(half_sin_theta)
        theta = 2 * half_theta

    nan_values = np.isnan(theta)
    if isinstance(theta, float):
        if nan_values:
            return 0.0
        return theta

    theta[nan_values] = 0.0
    return theta


# @nb.vectorize
def _equisolid(theta: LensArgument) -> LensArgument:
    """The equisolid function

    As a lens function, it takes an incidence angle and returns a
    distance from the center of the image such angle would be projected
    in the final image.

    Args:
         theta: either a float or numpy array containing
            floats representing the incidence angle in radians.

    Returns:
        A float or a numpy array of floats representing the distance in
            focal units the angles would be projected.
    """

    half_theta = theta / 2
    half_projection = np.sin(half_theta)
    projection = 2 * half_projection
    return projection


# @nb.vectorize
def _orthographic_inverse(projection_in_f_units: LensArgument) -> LensArgument:
    """The inverse orthographic function

    As an inverse lens function, it takes a distance from the center of
    the image in focal units and returns the incidence angle in radians
    that would produce such distance.

    Args:
         projection_in_f_units: either a float or numpy array containing
            floats representing the projection in focal distance units.

    Returns:
        A float or a numpy array of floats representing the angle or
            angles in radians."""
    theta = np.arcsin(projection_in_f_units)
    return theta


# @nb.vectorize
def _orthographic(theta: LensArgument) -> LensArgument:
    """The orthgraphic function

    As a lens function, it takes an incidence angle and returns a
    distance from the center of the image such angle would be projected
    in the final image.

    As an interesting fact the equidistant function and its inverse are
    one and the same.

    Args:
         theta: either a float or numpy array containing
            floats representing the incidence angle in radians.

    Returns:
        A float or a numpy array of floats representing the distance in
            focal units the angles would be projected.
    """

    projection = np.sin(theta)
    return projection


# @nb.vectorize
def _thoby_inverse(projection_in_f_units: LensArgument) -> LensArgument:
    """The inverse thoby function

    As an inverse lens function, it takes a distance from the center of
    the image in focal units and returns the incidence angle in radians
    that would produce such distance.

    Args:
         projection_in_f_units: either a float or numpy array containing
            floats representing the projection in focal distance units.

    Returns:
        A float or a numpy array of floats representing the angle or
            angles in radians."""

    k1 = 1.47
    k2 = 0.713
    theta = np.arcsin(projection_in_f_units / k1) / k2

    return theta


# @nb.vectorize
def _thoby(theta: LensArgument) -> LensArgument:
    """The thoby function

    As a lens function, it takes an incidence angle and returns a
    distance from the center of the image such angle would be projected
    in the final image.

    As an interesting fact the equidistant function and its inverse are
    one and the same.

    Args:
         theta: either a float or numpy array containing
            floats representing the incidence angle in radians.

    Returns:
        A float or a numpy array of floats representing the distance in
            focal units the angles would be projected.
    """

    k1 = 1.47
    k2 = 0.713
    projection = k1 * np.sin(k2 * theta)
    return projection


# Begin the exported functions


def rectilinear() -> Lens:
    """Returns a rectilinear lens"""
    return Lens(_rectilinear, _rectilinear_inverse)


def equisolid() -> Lens:
    """Returns an equisolid lens"""
    return Lens(_equisolid, _equisolid_inverse)


def equidistant() -> Lens:
    """Returns an equidistant lens"""
    return Lens(_equidistant, _equidistant_inverse)


def orthographic() -> Lens:
    """Returns an orthographic lens"""
    return Lens(_orthographic, _orthographic_inverse)


def stereographic() -> Lens:
    """Returns a stereographic lens"""
    return Lens(_stereographic, _stereographic_inverse)


def thoby() -> Lens:
    """Returns a thoby lens"""
    return Lens(_thoby, _thoby_inverse)


__all__ = [
    Lens,
    equisolid,
    equidistant,
    rectilinear,
    stereographic,
    orthographic,
    thoby,
]