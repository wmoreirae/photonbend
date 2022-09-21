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

__doc__ = """
This module is the main part of photonbend. Within its submodules it gathers the tools
needed to process camera-based images and panoramas.

# Terminology
Some basic definitions needed to be able to use its functionality properly.

## Image
An image is, for the intends and purposes of this module, a numpy array of uint8 with
shape (height, width, channels). Channels is **always** equal to number 3, and it stands for the 
pixels channels, the colors **Red, Green and Blue**.

```
# Making a new black image of width 1000 and height 600 
import numpy as np
black_image = np.zeros((600, 1000, 3), np.uint8)
```

## Coordinate Map
A coordinate map is a numpy array of float64 with shape (height, width, sub-elements).
The sub-elements value **always** equals to 3, representing the Latitude, Longitude and
a Invalid pixel marker.
- Latitude is represented in radians from 0 to Pi radians.
- Longitude is represented in radians from 0 to 2*Pi radians.
- Invalid is a marker. Any pixel which has an invalid value not equal to 0 should be
    considered invalid. 

## Projection Protocol
The protocol used by photonbend to allow the interchange between image formats.
It is composed of only 2 methods:
 - get_coordinate_map(): Should return the coordinate map of it's object/image.
 - process_coordinate_map(coordinate_map): Should use each pixel of the received
    coordinate_map, translating its coordinate data to this object's image data as
    referenced by this object own coordinate mapping function, returning this mapped
    image.
"""

