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

import numpy as np


# Tests for is_position()
# p is (0, 0)
def test_is_position_0_0_valid_180_inscribed(_180_inscribed_black_image):
    s = _180_inscribed_black_image
    p = (0, 0)
    assert s.is_position(*p) is True, f"Position {p} must be True on a {s.shape} image."


def test_is_position_0_0_valid_360_inscribed(_360_inscribed_black_image):
    s = _360_inscribed_black_image
    p = (0, 0)
    assert s.is_position(*p) is True, f"Position {p} must be True on a {s.shape} image."


def test_is_position_0_0_valid_360_double(_360_double_inscribed_black_image):
    s = _360_double_inscribed_black_image
    p = (0, 0)
    assert s.is_position(0, 0) is True, f"Position {p} must be True on a {s.shape} image."


# p is (length-1, length-1)
def test_is_position_length_minus_1_valid_180_inscribed(_180_inscribed_black_image):
    s = _180_inscribed_black_image
    y, x = s.shape[:2]
    r = y - 1
    c = x - 1
    p = (c, r)
    assert s.is_position(*p) is True, f"Position {p} must be True on a {s.shape} image."


def test_is_position_length_minus_1_valid_360_inscribed(_360_inscribed_black_image):
    s = _360_inscribed_black_image
    y, x = s.shape[:2]
    r = y - 1
    c = x - 1
    p = (c, r)
    assert s.is_position(*p) is True, f"Position {p} must be True on a {s.shape} image."


def test_is_position_length_minus_1_valid_360_double(_360_double_inscribed_black_image):
    s = _360_double_inscribed_black_image
    y, x = s.shape[:2]
    r = y - 1
    c = x - 1
    p = (c, r)
    assert s.is_position(*p) is True, f"Position {p} must be True on a {s.shape} image."


# p is (length, length)
def test_is_position_length_valid_180_inscribed(_180_inscribed_black_image):
    s = _180_inscribed_black_image
    y, x = s.shape[:2]
    p = (x, y)
    assert s.is_position(*p) is False, f"Position {p} must be False on a {s.shape} image."


def test_is_position_length_valid_360_inscribed(_360_inscribed_black_image):
    s = _360_inscribed_black_image
    y, x = s.shape[:2]
    p = (x, y)
    assert s.is_position(*p) is False, f"Position {p} must be False on a {s.shape} image."


def test_is_position_length_valid_360_double(_360_double_inscribed_black_image):
    s = _360_double_inscribed_black_image
    y, x = s.shape[:2]
    p = (x, y)
    assert s.is_position(*p) is False, f"Position {p} must be False on a {s.shape} image."


# p is (0, length):
def test_is_position_0_length_minus_1_valid_180_double(_180_inscribed_black_image):
    s = _180_inscribed_black_image
    y, x = s.shape[:2]
    r = y - 1
    c = x - 1
    p = (0, r)
    assert s.is_position(*p) is True, f"Position {p} must be True on a {s.shape} image."


def test_is_position_0_length_valid_360_inscribed(_360_inscribed_black_image):
    s = _360_inscribed_black_image
    y, x = s.shape[:2]
    p = (0, y)
    assert s.is_position(*p) is False, f"Position {p} must be False on a {s.shape} image."


def test_is_position_0_length_valid_360_double(_360_double_inscribed_black_image):
    s = _360_double_inscribed_black_image
    y, x = s.shape[:2]
    p = (0, y)
    assert s.is_position(*p) is False, f"Position {p} must be False on a {s.shape} image."


# p is (length, 0):
def test_is_position_length_0_minus_1_valid_180_double(_180_inscribed_black_image):
    s = _180_inscribed_black_image
    y, x = s.shape[:2]
    r = y - 1
    c = x - 1
    p = (c, 0)
    assert s.is_position(*p) is True, f"Position {p} must be True on a {s.shape} image."


def test_is_position_length_0_valid_360_inscribed(_360_inscribed_black_image):
    s = _360_inscribed_black_image
    y, x = s.shape[:2]
    p = (x, 0)
    assert s.is_position(*p) is False, f"Position {p} must be False on a {s.shape} image."


def test_is_position_length_0_valid_360_double(_360_double_inscribed_black_image):
    s = _360_double_inscribed_black_image
    y, x = s.shape[:2]
    p = (x, 0)
    assert s.is_position(*p) is False, f"Position {p} must be False on a {s.shape} image."


# ----------------------------------------------------------------------------------------------------------------------

# Tests for get_from_spherical()
# p = (np.pi/4, np.pi/4)

def test_get_from_spherical_45_45_inscribed(_180_image_red_at_45_45degrees):
    s = _180_image_red_at_45_45degrees
    p = (np.pi / 4, np.pi / 4)
    r, g, b = s.get_from_spherical(*p)
    red = 255, 0, 0

    assert (r, g, b) == red, f'Position {p} must be {red}'


def test_get_from_spherical_45_45_double(_360_double_red_at_45_45degrees):
    s = _360_double_red_at_45_45degrees
    from PIL import Image
    Image.fromarray(s.image).show()
    p = (-np.pi / 4, np.pi / 4)
    r, g, b = s.get_from_spherical(*p)
    red = 255, 0, 0

    assert (r, g, b) == red, f'Position {p} must be {red}'
