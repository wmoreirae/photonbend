import numpy as np
from numba import uint8, float64, cfunc
from numba.experimental import jitclass

spec = [
    ('center', float64[2]),
    ('image', uint8[:, :, :]),
    ('lens', cfunc)
]

@njitclass
class sphere_image:

    def __init__(self, image_arr, lens):
        self.image = image_arr
        self.lens = lens
        self._set_center()

    def _set_center(self):
        height, width = self.image.shape[:2]
        self.center[0] = height
        self.center[1] = width
        self.center = self.center / 2
        if self.center