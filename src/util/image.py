from enum import Enum

import numpy as np
import tensorflow as tf
import turbojpeg
from PIL import Image

JPEG = turbojpeg.TurboJPEG()


class ImageFormat(Enum):
    """Enum for different image formats.

    Args:
        Enum: Enum for different image formats.

    """

    GRAYSCALE = 1
    YUYV = 2
    YUV = 3
    RGB = 4


def convert_yuv_to_rgb(image):
    """Convert YUV to RGB.

    Args:
        image: the image in YUV format.

    Returns:
        image in RGB format.

    """
    if tf.is_tensor(image):
        image = image.numpy()

    assert len(image.shape) == 3  # has dimensions [H, W, C]
    assert image.shape[2] == 3  # has 3 channels

    return np.asarray(Image.fromarray(image, "YCbCr").convert("RGB"))


def convert_yuyv_to_yuv(image):
    """Convert an image from YUYV to YUV. The image in the YUYV format has the has the dimensions [H, W/2, 4] due to horizontal chroma subsampling (YUV 422). The converted image with the dimensions [H, W, 3] has more elements but can be illustrated better.

    Args:
        image: image in YUYV format. [..., H, W/2, 4]

    Returns:
        image in YUV format. [..., H, W, 3]
    """
    # Stack the the image along the channel dimensions in order to go from YUYV to Y1UVY2UV. Then reshape it to [..., H_in, W_in*2, 3]
    image_yuv_stack = tf.stack(
        [
            image[..., 0],
            image[..., 1],
            image[..., 3],
            image[..., 2],
            image[..., 1],
            image[..., 3],
        ],
        axis=-1,
    )

    shape = tf.shape(image)

    # Calculate the output shape dynamically
    # If the rank is 3, the output shape is [H, W*2, 3]
    # If the rank is 4, the output shape is [B, H, W*2, 3]
    output_shape = tf.concat(
        [
            shape[:-2],  # Keep all dimensions except the last two
            [
                # shape[-3],
                shape[-2] * 2,
                3,
            ],  # Reshape the last two dimensions to [H, W*2, 3] or [B, H, W*2, 3]
        ],
        axis=0,
    )

    image_yuv = tf.reshape(
        image_yuv_stack, output_shape
    )  # [B, H_in, W_in*2, 3] or [H_in, W_in*2, 3]

    return image_yuv


def convert_yuyv_to_rgb(image):
    """converts an image in YUYV format to RGB format. The image in the YUYV format has the has the dimensions [H, W/2, 4] due to horizontal chroma subsampling (YUV 422). The converted image with the dimensions [H, W, 3] has more elements but can be illustrated better.

    Args:
        image: image in YUYV format

    Returns:
        image in RGB format
    """
    # remove batch dimension
    if len(image.shape) == 4:
        image = tf.squeeze(image, axis=0)  # [H, W/2, 4]

    return convert_yuv_to_rgb(convert_yuyv_to_yuv(image))


def load_bhuman_jpeg_image(data, image_format=ImageFormat.GRAYSCALE):
    """Load a bhuman jpeg image from the given jpeg data.

    Args:
        data: jpeg data to load the image from.
        image_format: Format of the image. Defaults to ImageFormat.GRAYSCALE.

    Returns:
        image in the desired format.

    """
    image = JPEG.decode(data, pixel_format=turbojpeg.TJPF_CMYK)
    if image_format == ImageFormat.GRAYSCALE:
        return np.reshape(image[:, :, [0, 2]], (image.shape[0], image.shape[1] * 2, 1))
    if image_format == ImageFormat.YUYV:
        return np.reshape(image, (image.shape[0], image.shape[1] * 2, 2))
    image = np.reshape(image[:, :, [0, 1, 3, 2, 1, 3]], (image.shape[0], image.shape[1] * 2, 3))
    return convert_yuv_to_rgb(image) if image_format == ImageFormat.RGB else image
