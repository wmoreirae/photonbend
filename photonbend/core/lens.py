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

__doc__ = """This module has the **Lens** class and the implementation for some of the
    lens functions and reverse lens functions.

    It provides many Lenses models which are usually used with
    photonbend.core.projection.CameraImage instances.

    For static analysis purposes, it defines the **UniFloat** type.
    For all intents and purposes, this type means either **float** or
    **npt.NDArray[np.float64]**, but only one of them per function call.

    That means all lens functions:
    * Return a *float* when given one.
    * Return a *npt.NDArray[float]* when given one.
    """

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
import warnings

from typing import Callable, TypeVar, cast
from photonbend.utils import to_radians

UniFloat = TypeVar("UniFloat", float, npt.NDArray[np.float64])


@dataclass
class Lens:
    """Represents a lens with both forward and reverse functions.

    Attributes:
        forward_function (Callable[[UniFloat], UniFloat]): A function that given
            an incidence angle in radians gives back a distance from the
            projection center in focal distance units.
            It must handle either a single float or an array of floats.
        reverse_function (Callable[[UniFloat], UniFloat]): A function that
            given an distance from the projection center in focal
            distance units back the incidence angle in radians.
            It must handle either a single float or an array of floats.
    """

    forward_function: Callable[[UniFloat], UniFloat]
    reverse_function: Callable[[UniFloat], UniFloat]


# @nb.vectorize
def _rectilinear_inverse(
    projection_in_focal_distance_units: UniFloat,
) -> UniFloat:
    theta = np.arctan(projection_in_focal_distance_units)
    return theta


# TODO fix this function so it works with numpy arrays
def _rectilinear(theta: UniFloat) -> UniFloat:
    """Mapping that uses the angle tangent.

    As it uses the angle tangent, it should not be used with lens angles closing on 180
    degrees. It is best to limit its use to lens angles of at most 165 degrees.
    As the angle presented is halved, it should not see a theta angle larger than 82.5
    degrees.

    Args:
        theta:

    """
    if isinstance(theta, float):
        if theta < 0:
            raise ValueError("The angle theta cannot be negative")
        if theta > to_radians(89):
            raise ValueError(
                "The Rectilinear lens can't handle FoV larger than 179 degrees"
            )
        return np.tan(theta)
    else:
        invalid_bellow = theta < 0 
        invalid_above = theta > to_radians(89)
        invalid = np.logical_or(invalid_bellow, invalid_above)

        results = np.tan(theta)
        results[invalid] = np.nan
        return results

def _stereographic_inverse(projection_in_f_units: UniFloat) -> UniFloat:
    """Inverse stereographic function.

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

    half_tan_theta = projection_in_f_units / 2.0
    half_theta = np.arctan(half_tan_theta)
    theta = 2.0 * half_theta
    return theta


def _stereographic(theta: UniFloat) -> UniFloat:
    """The stereographic function.

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
    half_theta = theta / 2.0
    half_projection = np.tan(half_theta)
    projection = 2.0 * half_projection
    return projection


def _equidistant_inverse(projection_in_f_units: UniFloat) -> UniFloat:
    """The inverse equidistant function.

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
def _equidistant(theta: UniFloat) -> UniFloat:
    """The equidistant function.

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
def _equisolid_inverse(projection_in_f_units: UniFloat) -> UniFloat:
    """The inverse equisolid function.

    As an inverse lens function, it takes a distance from the center of
    the image in focal units and returns the incidence angle in radians
    that would produce such distance.

    Args:
         projection_in_f_units: either a float or numpy array containing
            floats representing the projection in focal distance units.

    Returns:
        A float or a numpy array of floats representing the angle or
            angles in radians."""

    half_sin_theta = projection_in_f_units / 2.0
    with warnings.catch_warnings():
        # ignore warnings that will be thrown because of NaNs
        warnings.simplefilter("ignore")
        half_theta = np.arcsin(half_sin_theta)
        theta = 2.0 * half_theta

    nan_values = np.isnan(theta)
    if isinstance(theta, float):
        if nan_values:
            return 0.0
        return theta

    theta[nan_values] = 0.0
    return theta


# @nb.vectorize
def _equisolid(theta: UniFloat) -> UniFloat:
    """The equisolid function.

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

    half_theta = theta / 2.0
    half_projection = cast(UniFloat, np.sin(half_theta))
    projection = 2 * half_projection
    return projection


# @nb.vectorize
def _orthographic_inverse(projection_in_f_units: UniFloat) -> UniFloat:
    """The inverse orthographic function.

    As an inverse lens function, it takes a distance from the center of
    the image in focal units and returns the incidence angle in radians
    that would produce such distance.

    Args:
         projection_in_f_units (float|ndarray[float]): either a float or an
            array representing the projection in focal distance units.

    Returns:
        A float or a numpy array of floats representing the angle or
            angles in radians."""
    theta = cast(UniFloat, np.arcsin(projection_in_f_units))
    return theta


# @nb.vectorize
def _orthographic(theta: UniFloat) -> UniFloat:
    """The orthgraphic function.

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
def _thoby_inverse(projection_in_f_units: UniFloat) -> UniFloat:
    """The inverse thoby function.

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
def _thoby(theta: UniFloat) -> UniFloat:
    """The thoby function.

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
    r"""Returns a rectilinear lens.

    It's functions are:
    * $f(\theta) = \tan(\theta)$
    * $f(projection) = \arctan(projection)$
    """
    return Lens(_rectilinear, _rectilinear_inverse)


def equisolid() -> Lens:
    r"""Returns an equisolid lens.

    It's functions are:
    * $f(\theta) = 2 \times \sin(\frac{\theta}{2})$
    * $f(projection) = 2 \times \arcsin(\frac{projection}{2})$
    """
    return Lens(_equisolid, _equisolid_inverse)


def equidistant() -> Lens:
    r"""Returns an equidistant lens.

    It's functions are:
    * $f(\theta) = \theta$
    * $f(projection) = projection$

    Both are a simple identity function.
    """
    return Lens(_equidistant, _equidistant_inverse)


def orthographic() -> Lens:
    r"""Returns an orthographic lens.

    It's functions are:
    * $f(\theta) = \sin(\theta)$
    * $f(projection) = \arcsin(projection)$
    """
    return Lens(_orthographic, _orthographic_inverse)


def stereographic() -> Lens:
    r"""Returns a stereographic lens

    It's functions are:
    * $f(\theta) = 2 \times \tan(\frac{\theta}{2})$
    * $f(projection) = 2\times \arctan(\frac{projection}{2})$
    """
    return Lens(_stereographic, _stereographic_inverse)


def thoby() -> Lens:
    r"""Returns a thoby lens.
    It's functions are:

    * $f(\theta) = 1.47 \times \sin(0.713 \times \theta)$

    * $f(projection) = \frac{\arcsin(\frac{projection}{1.47})}{0.713}$
    """
    return Lens(_thoby, _thoby_inverse)


__all__ = [
    "Lens",
    "equisolid",
    "equidistant",
    "rectilinear",
    "stereographic",
    "orthographic",
    "thoby",
]
