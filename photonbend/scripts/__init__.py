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
    # Scripts
    When photonbend is installed, it sets up a script named photonbend with 3 different
    commands to help you deal with your images.
     - [make-photo](#make-photo)
     - [alter-photo](#alter-photo)
     - [make-pano](#make-pano)

    ## Parameters
    The commands have a common theme among them.
    Parameters may have an **"i"** prefix or an **"o"** prefix. The former refer to
    parameters related to the input image. The latter refer to parameters related to the
    output image. When a single version of the parameter is possible for the operation,
    the prefix is omitted.
    - --help: Provide some instructions on the command usage
    - --lens (ilens|olens): Describe the desired lens used for the image.
        - equidistant
        - equisolid
        - orthographic
        - rectilinear
        - stereographic
    - --type (itype|otype): Type of the image used or desired.
        - inscribed: The valid data is on a inscribed circle.
        - double: The valid data is on two inscribed side-by-side circles.
        - cropped: The valid data is on a inscribed circle, top-and-bottom cropped.
        - full: The whole area of the image is valid data.
    - --fov (ifov|ofov): The camera or sensor Field of View in degrees.
    - --rotation: The rotation you want to apply in degrees on 3 axis of freedom.
        - pitch
        - yaw
        - roll
    - --size: The vertical size of the destiny image. This is usually optional. The
        script will select the same vertical height as the input image if omitted.

    ## make-photo
    This tool allows you to make a photo out of an equirectangular panorama (2:1 aspect
    ration).

    #### Make a 360 degrees photo with an equidistant lens
    The example below creates a photo of type `inscribed`, with an `equidistant` lens,
    and an FoV of `360` degrees named `equidistant.jpg` from the panorama in the file
    named `panorama.jpg`

    ```
    photonbend make-photo --type inscribed --lens equidistant --fov 360 \\
    panorama.jpg
    ```

    ## alter-photo
    This tool allows you to change your photos by exchanging lenses, FoV, and types as
    well as rotate your images.

    #### Change of projection (Lens)
    The example below changes the photo lenses from `equidistant` projection to
    `equisolid` projection.

    ```
    photonbend alter-photo --itype inscribed --otype inscribed --ilens equidistant \\
    --olens equisolid --ifov 360 --ofov 360 equidistant.jpg equisolid.jpg
    ```

    #### Change of FoV
    The example below changes the photo `equidistant.jpg`. Its FoV is altered from `360`
    degrees to `180`, producing the image `equidistant-180.jpg`.

    ```
    photonbend alter-photo --itype inscribed --otype inscribed --ilens equidistant \\
    --olens equidistant --ifov 360 --ofov 180 equidistant.jpg equidistant-180.jpg
    ```

    **Notice this is a very lossy operation. The new image will lose about half of its
     view data**

    #### Change of type
    The example below changes the photo `equidistant.jpg`.
    Its type from `inscribed` to `double`, producing `equidistant-double.jpg`.

    **Note**: When producing a **double inscribed** image, we **nominally** also have to
    **change the FoV**. That happens because the double inscribed image uses two
    inscribed images side by side on a single image file. Since double inscribed images
    are only meant to be used with full 360 degrees images, the FoV changes its meaning
    to describe the FoV of each sensor instead of the FoV of the whole image.

    ```
    photonbend alter-photo --itype inscribed --otype double --ilens equidistant \\
     --olens equidistant --ifov 360 --ofov 195 equidistant.jpg equidistant-double.jpg
    ```

    #### Change of type, lens, and FoV
    The example below changes the photo `equidistant.jpg` from type `inscribed` to
    `full`, its lenses from `equidistant` to `rectilinear`, and its FoV from `360`
    degrees to `140`, producing the image `rectlinear-full.jpg`.

    ```
    photonbend alter-photo --itype inscribed --otype full --ilens equidistant \\
    --olens rectilinear --ifov 360 --ofov 140 equidistant.jpg rectlinear-full.jpg
    ```

    #### Rotation
    The example below changes the photo `equidistant.jpg`, rotating it `-90` degrees in
    pitch, `0` degrees in yaw, and `0` degrees in roll, producing
    `equidistant-rotated.jpg`.

    ```
    photonbend alter-photo --itype inscribed --otype inscribed --ilens equidistant \\
    --olens equidistant --ifov 360 --ofov 360 --rotation -90 0 0 equidistant.jpg \\
    equidistant-rotated.jpg
    ```

    #### Combining it all
    The example below changes the photo `equidistant.jpg` from type `inscribed` to
    `full`, its lenses from `equidistant` to `rectilinear`, and its FoV from `360`
    degrees to `140`. It is also rotated by `-90` degrees in pitch, `195` degrees in yaw
     and `0` degrees in roll producing the image `rectlinear-140-full-rotated.jpg`.

    ```
    photonbend alter-photo --itype inscribed --otype full --ilens equidistant \\
    --olens rectilinear --ifov 360 --ofov 140 --rotation -90 0 195 equidistant.jpg \\
    rectlinear-140-full-rotated.jpg
    ```

    ## make-pano
    This tool allows you to change create panoramas out of your photos

    #### Make a panorama
    Make a panorama out of an `inscribed`, `equidistant` lens, `360` degrees FoV photo
    named `equidistant.jpg`, producing `panorama.jpg`.

    ```
    photonbend make-pano --type inscribed --lens equidistant \\
    --fov 360 equidistant.jpg panorama.jpg
    ```

    #### Make a rotated panorama
    Make a panorama out of an `inscribed`, `equidistant` lens, `360` degrees FoV photo
    named `equidistant.jpg`, producing `panorama_rotated.jpg`.

    ```
    photonbend make-pano --type inscribed --lens equidistant --fov 360 \\
    --rotation -90 0 90 equidistant.jpg panorama.jpg
    ```
    """

from typing import List

__all__: List[str] = []
