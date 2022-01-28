import numpy as np
from numba import njit, float64, cfunc
from lightbend.utils import degrees_to_radians

@cfunc(float64(float64))
def rectilinear_inverse(projection_in_focal_distance_units):
    theta = np.arctan(projection_in_focal_distance_units)
    return theta


@cfunc(float64(float64))
def rectilinear(theta):
    """Mapping that uses the angle tangent

    As it uses the angle tangent, it should not be used with lens angles closing on 180 degrees.
    It is best to limit its use to lens angles of at most 165 degrees.
    As the angle presented is halved, it should not see a theta angle larger than 82.5 degrees.
    """
    assert theta > 0
    if theta > degrees_to_radians(165):
        raise ValueError('The rectilinear function was not made to handle angles larger than 165 degrees')
    return np.tan(theta)


@cfunc(float64(float64))
def stereographic_inverse(projection_in_focal_distance_units):
    half_tan_theta = projection_in_focal_distance_units / 2
    half_theta = np.arctan(half_tan_theta)
    theta = 2 * half_theta
    return theta


@cfunc(float64(float64))
def stereographic(theta):
    half_theta = theta / 2
    half_projection = np.tan(half_theta)
    projection = 2 * half_projection
    return projection


@cfunc(float64(float64))
def equidistant_inverse(projection_in_focal_distance_units):
    return projection_in_focal_distance_units


@cfunc(float64(float64))
def equidistant(theta):
    return theta


@cfunc(float64(float64))
def equisolid_inverse(projection_in_focal_distance_units):
    half_sin_theta = projection_in_focal_distance_units / 2
    half_theta = np.arcsin(half_sin_theta)
    theta = 2 * half_theta
    return theta


@cfunc(float64(float64))
def equisolid(theta):
    half_theta = theta / 2
    half_projection = np.sin(half_theta)
    projection = 2 * half_projection
    return projection


@cfunc(float64(float64))
def orthographic_inverse(projection_in_focal_distance_units):
    theta = np.arcsin(projection_in_focal_distance_units)
    return theta


@cfunc(float64(float64))
def orthographic(theta):
    projection = np.sin(theta)
    return projection
