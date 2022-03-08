import numba
import numpy as np
from PIL import Image

from lightbend.lens import equisolid, equisolid_inverse, rectilinear, rectilinear_inverse, equidistant, \
    equidistant_inverse, orthographic, orthographic_inverse, stereographic, stereographic_inverse

from lightbend.utils import degrees_to_radians

from lightbend.core.sphere_image import SphereImage, ImageType

if __name__ == '__main__':

