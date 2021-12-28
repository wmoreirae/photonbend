import numpy as np
from numba import njit


@njit
def rectilinear_inverse(projection_in_focal_distance_units):
    theta = np.arctan(projection_in_focal_distance_units)
    return theta


@njit
def rectilinear(theta):
    """Mapping that uses the angle tangent

    As it uses the angle tangent, it should not be used with lens angles closing on 180 degrees.
    It is best to limit its use to lens angles of at most 165 degrees.
    As the angle presented is halved, it should not see a theta angle larger than 82.5 degrees.
    """
    return np.tan(theta)


@njit
def stereographic_inverse(projection_in_focal_distance_units):
    half_tan_theta = projection_in_focal_distance_units / 2
    half_theta = np.arctan(half_tan_theta)
    theta = 2 * half_theta
    return theta


@njit
def stereographic(theta):
    half_theta = theta / 2
    half_projection = np.tan(half_theta)
    projection = 2 * half_projection
    return projection


@njit
def equidistant_inverse(projection_in_focal_distance_units):
    return projection_in_focal_distance_units


@njit
def equidistant(theta):
    return theta


@njit
def equisolid_inverse(projection_in_focal_distance_units):
    half_sin_theta = projection_in_focal_distance_units / 2
    half_theta = np.arcsin(half_sin_theta)
    theta = 2 * half_theta
    return theta


@njit
def equisolid(theta):
    half_theta = theta / 2
    half_projection = np.sin(half_theta)
    projection = 2 * half_projection
    return projection


@njit
def orthographic_inverse(projection_in_focal_distance_units):
    theta = np.arcsin(projection_in_focal_distance_units)
    return theta


@njit
def orthographic(theta):
    angle = np.sin(theta)
    return angle
