import numba
import numpy as np
from PIL import Image

from lightbend.lens import equisolid, equisolid_inverse, rectilinear, rectilinear_inverse, equidistant, \
    equidistant_inverse, orthographic, orthographic_inverse, stereographic, stereographic_inverse

from lightbend.utils import degrees_to_radians

from lightbend.core.sphere_image import SphereImageInscribed, ImageType

if __name__ == '__main__':
    # origin_image = Image.open('./images/fisheye_180.jpg')
    # origin_arr = np.asarray(origin_image)
    # s_origin = SphereImageInscribed(origin_arr, ImageType.INSCRIBED, np.pi, equisolid, equisolid_inverse)
    # s_destiny = SphereImageInscribed(np.zeros([4096, 4096, 3], np.core.uint8), ImageType.INSCRIBED, degrees_to_radians(110),
    #                                  rectilinear, rectilinear_inverse)
    # s_destiny.map_from_sphere_image(s_origin)
    # destiny_arr = s_destiny.get_image_array()
    # destiny_image = Image.fromarray(destiny_arr)
    # destiny_image.save('results/Rectilinear.jpg')

    origin_image = Image.open('./images/fisheye_180.jpg')
    origin_arr = np.asarray(origin_image)
    s_origin = SphereImageInscribed(origin_arr, ImageType.INSCRIBED, degrees_to_radians(180), equisolid,
                                    equisolid_inverse)
    s_destiny = SphereImageInscribed(np.zeros([1800, 2880, 3], np.core.uint8), ImageType.CROPPED_CIRCLE,
                                     degrees_to_radians(180), equidistant, equidistant_inverse)
    s_destiny.map_from_sphere_image(s_origin)
    destiny_arr = s_destiny.get_image_array()
    destiny_image = Image.fromarray(destiny_arr)
    destiny_image.save('results/Equidistant.jpg')
