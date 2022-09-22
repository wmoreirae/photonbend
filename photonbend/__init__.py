#  Copyright (c) 2022. Edson Moreira
#
#  Permission is hereby granted, free of charge, to any person obtaining a copy
#  of this software and associated documentation files (the "Software"), to deal
#  in the Software without restriction, including without limitation the rights
#  to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
#  copies of the Software, and to permit persons to whom the Software is
#  furnished to do so, subject to the following conditions:
#
#  The above copyright notice and this permission notice shall be included in
#  all copies or substantial portions of the Software.
#
#  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
#  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
#  OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
#  SOFTWARE.


__doc__ = """
    # Intro
    Photonbend is a python module to handle photos, especially photos taken with
    fisheye lenses, and convert them between different kinds of lenses, FoV, and
    types of photos like inscribed circles, cropped circles, or even
    side-by-side double inscribed circles. It also allows you to rotate those
    photos, convert them to equirectangular panoramas or convert panoramas to
    fisheye photos.

    It can be used as a library to handle images on your projects or it can be
    used as a standalone tool with its own set of commands to help you alter
    your photos taken with a fisheye lens, an omnidirectional camera such as the
    Samsung Gear 360 or an equirectangular panorama.

    If you just want to use the tools go to the Scripts. If you want to
    understand how it works just keep reading.

    # How it works
    This module uses the information you provide about the image format, lenses,
    and FoV, couples it with mathematical functions that describes the ways the
    lenses behave, and makes use of trigonometry to map the pixels of your planar
    photos or panoramas to angles producing a sphere-like mappings.

    Using a sphere as a base lets you rotate the image. It provides lens functions
    and and objects that let you re-take a picture from another one, using different
    lenses and FoV to produce new images. It also lets you map the camera images
    to an equirectangular panorama.
    
    # Install
    Installing is simple using pip
    
    ```
    pip install photonbend
    ```
    
    For usage read the documentation for the photobend.core module
    """
