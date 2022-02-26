#  Copyright (c) 2022. Edson Moreira
#
#  Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
#  documentation files (the "Software"), to deal in the Software without restriction, including without limitation
#  the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and
#  to permit persons to whom the Software is furnished to do so, subject to the following conditions:
#
#  The above copyright notice and this permission notice shall be included in all copies or substantial portions of
#  the Software.
#
#  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING
#  BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
#  NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
#  DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
#  OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

import pytest
import numpy as np
from lightbend.core import LensImage, LensImageType
from lightbend.lens import equidistant


# 180 INSCRIBED FIXTURES


@pytest.fixture
def _180_square_array():
    return np.zeros((180, 180, 3), np.core.uint8)


@pytest.fixture
def _180_inscribed_black_image(_180_square_array):
    return LensImage(_180_square_array, LensImageType.INSCRIBED, np.pi, equidistant)


@pytest.fixture
def _180_array_red_at_45_45degrees(_180_square_array):
    s = _180_square_array
    p = np.sqrt(45 ** 2 / 2)
    py = int(np.round(89.5 - p))
    px = int(np.round(p + 89.5))
    s[py, px, :] = (255, 0, 0)
    return s


@pytest.fixture
def _180_image_red_at_45_45degrees(_180_array_red_at_45_45degrees):
    s = _180_array_red_at_45_45degrees
    return LensImage(s, LensImageType.INSCRIBED, np.pi, equidistant)


# 360 INSCRIBED FIXTURES
@pytest.fixture
def _360_square_array():
    return np.zeros((360, 360, 3), np.core.uint8)


@pytest.fixture
def _360_inscribed_black_image(_360_square_array):
    return LensImage(_360_square_array, LensImageType.INSCRIBED, np.pi, equidistant)


# DOUBLE INSCRIBED FIXTURES
@pytest.fixture
def _180_360_rectangle_array():
    return np.zeros((180, 360, 3), np.core.uint8)


@pytest.fixture
def _360_double_inscribed_black_image(_180_360_rectangle_array):
    s = _180_360_rectangle_array
    return LensImage(s, LensImageType.DOUBLE_INSCRIBED, np.pi, equidistant)


@pytest.fixture
def _180_360_array_red_at_45_45degrees(_180_360_rectangle_array):
    s = _180_360_rectangle_array
    p = np.sqrt(45 ** 2 / 2)
    py = int(np.round(89.5 - p))
    px = int(np.round(269.5 - p))
    s[py, px, :] = (255, 0, 0)
    return s


@pytest.fixture
def _360_double_red_at_45_45degrees(_180_360_array_red_at_45_45degrees):
    s = _180_360_array_red_at_45_45degrees
    return LensImage(s, LensImageType.DOUBLE_INSCRIBED, np.pi, equidistant)
