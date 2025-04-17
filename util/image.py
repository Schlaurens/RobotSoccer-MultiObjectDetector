from enum import Enum

import numpy as np
import turbojpeg
from PIL import Image

JPEG = turbojpeg.TurboJPEG()


class ImageFormat(Enum):
    GRAYSCALE = 1
    YUYV = 2
    YUV = 3
    RGB = 4


def convert_yuv_to_rgb(image):
    assert len(image.shape) == 3
    assert image.shape[2] == 3
    return np.asarray(Image.fromarray(image, "YCbCr").convert("RGB"))


def load_bhuman_jpeg_image(data, image_format=ImageFormat.GRAYSCALE):
    image = JPEG.decode(data, pixel_format=turbojpeg.TJPF_CMYK)
    if image_format == ImageFormat.GRAYSCALE:
        return np.reshape(image[:, :, [0, 2]], (image.shape[0], image.shape[1] * 2, 1))
    if image_format == ImageFormat.YUYV:
        return np.reshape(image, (image.shape[0], image.shape[1] * 2, 2))
    image = np.reshape(
        image[:, :, [0, 1, 3, 2, 1, 3]], (image.shape[0], image.shape[1] * 2, 3)
    )
    return convert_yuv_to_rgb(image) if image_format == ImageFormat.RGB else image
