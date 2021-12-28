import numpy as np
from PIL import Image

from lightbend.lens import equisolid, equisolid_inverse, rectilinear, rectilinear_inverse

from lightbend.camera.imaging import change_lens
from lightbend.utils import degrees_to_radians

if __name__ == '__main__':
    origin_image = Image.open('./images/fisheye-lens-city.jpg')
    origin_arr = np.asarray(origin_image)

    del origin_image

    destiny_array = change_lens(origin_arr, degrees_to_radians(155),
                                equisolid,
                                equisolid_inverse,
                                rectilinear,
                                rectilinear_inverse,
                                True)

    del origin_arr
    destiny_image = Image.fromarray(destiny_array)
    del destiny_array


    #destiny_image = destiny_image.resize(map(lambda x: x//3 , destiny_image.size), Image.BICUBIC)

    destiny_image.save('results/rectilinear.jpg')
