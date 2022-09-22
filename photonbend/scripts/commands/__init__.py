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

import sys
from enum import IntEnum, auto
from pathlib import Path
from typing import Tuple, Literal, Optional, Final

import click
import numpy as np
from PIL import Image
from numpy import typing as npt

from photonbend.core.lens import (
    equidistant,
    equisolid,
    orthographic,
    rectilinear,
    stereographic,
    Lens,
)

from photonbend.core.projection import DoubleCameraImage, CameraImage

# Some literal types that will be used on many commands
from photonbend.utils import to_radians

Channels: Final[int] = 3

CamImgTypeStr = Literal["inscribed", "double", "cropped", "full"]
CamLensStr = Literal[
    "equidistant", "equisolid", "orthographic", "rectilinear", "stereographic"
]


def _verify_output_path(output: Path):
    out = Path(output)
    if not (out.suffix.lower() in [".jpg", ".jpeg", ".png"]):
        print("The desired output image should be a JPG or PNG file.")
        print(
            "Provide an output filename ending in either JPG, JPEG or PNG (case insensitive)"  # noqa E501
        )
        print("Exiting!")
        sys.exit(1)
    if out.exists():
        while True:
            ans = input("File already exists. Overwrite? (y/n) ")
            if ans in ["y", "n"]:
                break
        if ans == "n":
            print("Exiting!")
            sys.exit(0)
    return out


def _euclidean_distance(x, y):
    return np.sqrt(x**2 + y**2)


class CameraImageType(IntEnum):
    FULL_FRAME = auto()
    CROPPED_CIRCLE = auto()
    INSCRIBED = auto()
    DOUBLE_INSCRIBED = auto()


def _get_camera(cam_img_type: CameraImageType):
    if cam_img_type is CameraImageType.DOUBLE_INSCRIBED:
        return DoubleCameraImage
    else:
        return CameraImage


def _calculate_magnitude(image_type: CameraImageType, shape: Tuple[int, ...]) -> float:
    if len(shape) > 3:
        raise ValueError(
            "Can't calculate magnitude of images with more than 3 dimensions"
        )
    height, width, _ = shape
    magnitude: float = 0.0
    if image_type is CameraImageType.INSCRIBED:
        magnitude = width / 2 - 0.5
    elif image_type is CameraImageType.DOUBLE_INSCRIBED:
        magnitude = height / 2 - 0.5
    elif image_type is CameraImageType.FULL_FRAME:
        y = height / 2.0 - 0.5
        x = width / 2.0 - 0.5
        magnitude = _euclidean_distance(x, y)
    elif image_type is CameraImageType.CROPPED_CIRCLE:
        magnitude = width / 2 - 0.5

    return magnitude


def _process_image_type(type: CamImgTypeStr) -> CameraImageType:
    types_dict = {
        "inscribed": CameraImageType.INSCRIBED,
        "double": CameraImageType.DOUBLE_INSCRIBED,
        "cropped": CameraImageType.CROPPED_CIRCLE,
        "full": CameraImageType.FULL_FRAME,
    }

    return types_dict[type]


def _process_lens(lens: CamLensStr) -> Lens:
    lens_types = {
        "equidistant": equidistant,
        "equisolid": equisolid,
        "orthographic": orthographic,
        "rectilinear": rectilinear,
        "stereographic": stereographic,
    }

    return lens_types[lens]()


def _open_image(input_image) -> npt.NDArray[np.uint8]:
    try:
        with Image.open(input_image) as image:
            source_array: npt.NDArray[np.unit8] = np.asarray(image)  # type: ignore
    except IOError:
        print("Error: Input image could not be opened!")
        print("Exiting!")
        sys.exit(1)
    return source_array


# Help messages that are used on many commands

lens_choices = click.Choice(
    ["equidistant", "equisolid", "orthographic", "rectilinear", "stereographic"]
)
type_choices = click.Choice(["inscribed", "double", "cropped", "full"])

type_choices_help = """

    \b
    The choices are:
    - inscribed: The valid data is on a inscribed circle.
    - double: The valid data is on two inscribed side-by-side circles.
    - cropped: The valid data is on a inscribed circle, top-and-bottom cropped.
    - full: The whole area of the image is valid data.
    """
double_type_fov_warning = """

    IMPORTANT: FoV for double images are the value for one of the sensors and > 180ª."""
rotation_help = """
    The rotation that should be applied to the camera.
    This is a 3-valued parameter in the form <pitch yaw roll>
    """


def _process_fov(fov: float, image_type: CameraImageType):
    if image_type is CameraImageType.DOUBLE_INSCRIBED and fov < 180:
        raise ValueError("The fov of a double image can't be smaller than 180 degrees.")
    if fov > 360:
        raise ValueError("The fov of an image can't be higher than 360 degrees.")
    r_fov = to_radians(fov)
    return r_fov


def _calculate_destiny_size(
    image_type: CameraImageType, source_image: npt.NDArray, height: Optional[int]
) -> Tuple[int, int, int]:
    local_height = source_image.shape[0]
    if height is not None:
        local_height = height

    if image_type is CameraImageType.DOUBLE_INSCRIBED:
        ans = (local_height, 2 * local_height, Channels)
    else:
        ans = (local_height, local_height, Channels)
    return ans
