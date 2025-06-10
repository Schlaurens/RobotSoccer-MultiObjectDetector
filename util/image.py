from enum import Enum

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import turbojpeg
from matplotlib import patches
from PIL import Image

from . import dataset as u_dataset

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
    assert len(image.shape) == 3
    assert image.shape[2] == 3
    return np.asarray(Image.fromarray(image, "YCbCr").convert("RGB"))


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


def show_cell_on_image(directory, label, object_name=None, grid_dims=(15, 20)):
    """Show the given image with an illustrated cell grid of given dimension.

    If an object_name is given and that object is present in the label. Its objectness
    mask and loss mask are drawn on the image.

    Args:
        directory: Directory of the image
        label: label of the image
        object_name: name of the object. E. g. "ball". Defaults to None
        grid_dims: The dimensions of the cell_grid. Defaults to (15,20).

    """
    image = u_dataset.load_image(directory, label, image_format=ImageFormat.GRAYSCALE)

    # The dimension of a single cell
    cell_dims = np.array(image.shape[1::-1])[::-1] // np.array(grid_dims)

    _, ax = plt.subplots()
    ax.imshow(image, cmap='gray')
    ax.set_title(f"grid_dims={grid_dims}, cell_size={cell_dims}")

    # Draw cell grid with the given grid dimensions
    for i in range(image.shape[1])[:: cell_dims[1]]:
        ax.axvline(x=i, color="black")
    for i in range(image.shape[0])[:: cell_dims[0]]:
        ax.axhline(y=i, color="black")

    # Draw the cell with the highest objectness score (if a valid object_name is given)
    if object_name in label:
        _, objectness_mask, loss_mask = u_dataset.get_masks(
            label, object_name, output_dims=grid_dims
        )

        objectness_mask = np.array(objectness_mask)

        # Get the index with the highest value.
        indices_objectness = np.unravel_index(objectness_mask.argmax(), objectness_mask.shape)
        indices_loss_mask = np.dstack(np.where(loss_mask == loss_mask.min()))[0]

        # scale the index to the size of the cell grid
        scaled_objectness_mask_indices = indices_objectness * np.array(cell_dims)
        scaled_loss_mask_indices = indices_loss_mask * np.array(cell_dims)

        # Make sure that the indices are 2D arrays to make iteration possible in the next step.
        if len(scaled_loss_mask_indices.shape) == 1:
            scaled_loss_mask_indices = np.expand_dims(scaled_loss_mask_indices, axis=0)
        if len(scaled_objectness_mask_indices.shape) == 1:
            scaled_objectness_mask_indices = np.expand_dims(scaled_objectness_mask_indices, axis=0)

        # Draw a rectangle on the cell that cover the object.
        # TODO: will need modification when there
        # are multiple cell that cover an object.
        for i in scaled_objectness_mask_indices:
            rect_pos = patches.Rectangle(
                i[::-1],  # Flip x and y coordinates for matplotlib
                cell_dims[0],
                cell_dims[1],
                linewidth=1,
                edgecolor="black",
                facecolor=(255 / 255, 0 / 255, 0 / 255, 140 / 255),
            )
            ax.add_patch(rect_pos)

        for i in scaled_loss_mask_indices:
            rect_loss_mask = patches.Rectangle(
                i[::-1],  # Flip x and y coordinates for matplotlib
                cell_dims[0],
                cell_dims[1],
                linewidth=1,
                edgecolor="black",
                facecolor=(255 / 255, 123 / 255, 0 / 255, 140 / 255),
            )
            ax.add_patch(rect_loss_mask)

    plt.show()


def show_patches_on_image(image, label, results):
    """Draw the given image with rectangles that indicate the position of the extracted patches. And the patches in separate plots

    Args:
        image: the image in YUYV format [480, 640, 2]
        label: the label of the object
        results: the results from the patch extractor. Contains for each detected object
            a number of patch candidates,
            masks that indicate for each patch whether the center could be projected to the plane,
            and the normalized corner coordinates of each patch (y1, x1, y2, x2)

    """
    image_res = image.shape[0:-1]
    num_candidates = results[label][0].shape[1]

    # Image colorspace conversion
    image_yuv = tf.reshape(tf.constant(image), (-1, image_res[0], int(image_res[1] / 2), 4))
    image_converted = tf.reshape((image_yuv), (-1, image_res[0], image_res[1], 2))

    # Draw image with patches on top of it
    _, axes = plt.subplots()
    axes.imshow(image_converted[0, ..., 0] / 255, cmap="gray")

    for i, box in enumerate(results[label][2]):
        # Coordinates for each box are y1, x1, y2, x2
        # Upscale the normalized coordinates
        coords = (box[1] * (image_res[1] - 1), box[0] * (image_res[0] - 1))
        width = (box[3] - box[1]) * (image_res[1] - 1)
        height = (box[2] - box[0]) * (image_res[0] - 1)

        rect = patches.Rectangle(
            coords,
            width,
            height,
            linewidth=1,
            edgecolor="red",
            facecolor=(255 / 255, 123 / 255, 0 / 255, 0 / 255),
        )
        # Each patch has a number to identify the ordering
        axes.text(x=(coords[0] + 4.0), y=coords[1] + 17.0, s=i + 1, color = 'red')
        axes.add_patch(rect)

    plt.title("Image")

    # Draw the patch candidates in separate plots
    _, axes = plt.subplots(num_candidates)
    axes[0].imshow(results[label][0][0, 0, ..., 0].numpy() / 255, cmap="gray")
    axes[1].imshow(results[label][0][0, 1, ..., 0].numpy() / 255, cmap="gray")
    axes[2].imshow(results[label][0][0, 2, ..., 0].numpy() / 255, cmap="gray")
    axes[3].imshow(results[label][0][0, 3, ..., 0].numpy() / 255, cmap="gray")
    axes[4].imshow(results[label][0][0, 4, ..., 0].numpy() / 255, cmap="gray")

    plt.show()
