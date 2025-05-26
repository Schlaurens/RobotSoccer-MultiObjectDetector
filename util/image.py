from enum import Enum

import matplotlib.pyplot as plt
import numpy as np
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
    ax.imshow(image)
    ax.set_title(f"grid_dims={grid_dims}, cell_size={cell_dims}")

    # Draw cell grid with the given grid dimensions
    for i in range(image.shape[1])[:: cell_dims[1]]:
        ax.axvline(x=i, color="black")
    for i in range(image.shape[0])[:: cell_dims[0]]:
        ax.axhline(y=i, color="black")

    # Draw the cell with the highest objectness score (if a valid object_name is given)
    if object_name in label:
        _, objectness_mask, loss_mask = u_dataset.get_masks(label, object_name, output_dims=grid_dims)

        objectness_mask = np.array(objectness_mask)
        
        # Get the index with the highest value.
        indices_objectness = np.unravel_index(objectness_mask.argmax(), objectness_mask.shape)
        indices_loss_mask = np.dstack(np.where(loss_mask == loss_mask.min()))[0]

        # scale the index to the size of the cell grid
        scaled_objectness_mask_indices = indices_objectness * np.array(cell_dims)
        scaled_loss_mask_indices = indices_loss_mask * np.array(cell_dims)

        # Make sure that the indices are 2D arrays to make iteration possible in the next step.
        if (len(scaled_loss_mask_indices.shape) == 1):
            scaled_loss_mask_indices = np.expand_dims(scaled_loss_mask_indices, axis=0)
        if (len(scaled_objectness_mask_indices.shape) == 1):
            scaled_objectness_mask_indices = np.expand_dims(scaled_objectness_mask_indices, axis=0)
        
        # Draw a rectangle on the cell that cover the object.
        # TODO: will need modification when there
        # are multiple cell that cover an object.
        for i in scaled_objectness_mask_indices:
            rect_pos = patches.Rectangle(
                i[::-1], # Flip x and y coordinates for matplotlib
                cell_dims[0],
                cell_dims[1],
                linewidth=1,
                edgecolor="black",
                facecolor=(255 / 255, 0 / 255, 0 / 255, 140 / 255),
            )
            ax.add_patch(rect_pos)
        
        for i in scaled_loss_mask_indices:
            rect_loss_mask = patches.Rectangle(
                i[::-1], # Flip x and y coordinates for matplotlib
                cell_dims[0],
                cell_dims[1],
                linewidth=1,
                edgecolor="black",
                facecolor=(255 / 255, 123 / 255, 0 / 255, 140 / 255),
            )
            ax.add_patch(rect_loss_mask)

    plt.show()
