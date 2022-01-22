import numpy as np
from PIL import Image

from lightbend.lens import equisolid, equisolid_inverse, rectilinear, rectilinear_inverse, equidistant, \
    equidistant_inverse, orthographic, orthographic_inverse, stereographic, stereographic_inverse

from lightbend.camera.imaging import change_lens
from lightbend.utils import degrees_to_radians
from lightbend.conversion.circular_180.mercator import make_panoramic

if __name__ == '__main__':
    origin_image = Image.open('./images/fisheye_180.jpg')
    origin_arr = np.asarray(origin_image)

    destiny_array = make_panoramic(source=origin_arr,
                                   source_function=equisolid,
                                   inverse_source_function=equisolid_inverse)
                                   # destiny_function=equidistant,
                                   # inverse_destiny_function=equidistant_inverse)

    del origin_arr
    destiny_image = Image.fromarray(destiny_array)
    del destiny_array

    """
    del origin_image

    destiny_array = change_lens(origin_arr, degrees_to_radians(160),
                                equisolid,
                                equisolid_inverse,
                                rectilinear,
                                rectilinear_inverse,
                                True)

    del origin_arr
    destiny_image = Image.fromarray(destiny_array)
    del destiny_array
    """

    destiny_image.save('results/Pano180.jpg')
