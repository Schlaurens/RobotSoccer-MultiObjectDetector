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


def show_masks_on_image(directory, label, object_name=None, mask_name=None, grid_dims=(15, 20)):
    """Show the given image with an illustrated cell grid of given dimension.

    If an object_name is given and that object is present in the label. Its objectness
    mask and loss mask are drawn on the image.

    Args:
        directory: Directory of the image
        label: label of the image
        object_name: name of the object. E. g. "ball". Defaults to None
        mask_name: Name of the mask that should be drawn. None := no mask, 'objectness' := Objectness mask, 'loss' := Loss mask
        grid_dims: The dimensions of the cell_grid. Defaults to (15,20).

    """
    image = u_dataset.load_image(directory, label, image_format=ImageFormat.RGB)

    # The dimension of a single cell
    cell_dims = np.array(image.shape[1::-1])[::-1] // np.array(grid_dims)

    _, ax = plt.subplots()
    ax.imshow(image)
    ax.set_title(f"grid_dims={grid_dims}, cell_size={cell_dims}, mask={mask_name}")

    # Draw cell grid with the given grid dimensions
    for i in range(image.shape[1])[:: cell_dims[1]]:
        ax.axvline(x=i, color="black")
    for i in range(image.shape[0])[:: cell_dims[0]]:
        ax.axhline(y=i, color="black")

    if mask_name != None and object_name in label:
        offset_mask, objectness_mask, loss_mask = u_dataset.get_masks(
            label, object_name, output_dims=grid_dims
        )

        if mask_name == "objectness":
            coords = u_dataset.get_coords_from_offsets(offset_mask)
            if coords != None:
                ax.plot(coords[0], coords[1], "rx")

            objectness_mask = np.array(objectness_mask)

            # Get the indices with the highest/lowest values.
            indices_objectness_pos = np.dstack(np.where(objectness_mask == loss_mask.max()))[0]
            indices_objectness_neg = np.dstack(np.where(objectness_mask == loss_mask.min()))[0]

            # Scale the indices to the size of the cell grid
            scaled_objectness_mask_indices_pos = indices_objectness_pos * np.array(cell_dims)
            scaled_objectness_mask_indices_neg = indices_objectness_neg * np.array(cell_dims)

            # Make sure that the indices are 2D arrays to make iteration possible in the next step.
            if len(scaled_objectness_mask_indices_pos.shape) == 1:
                scaled_objectness_mask_indices_pos = np.expand_dims(
                    scaled_objectness_mask_indices_pos, axis=0
                )

            if len(scaled_objectness_mask_indices_neg.shape) == 1:
                scaled_objectness_mask_indices_neg = np.expand_dims(
                    scaled_objectness_mask_indices_neg, axis=0
                )

            # Draw a rectangle on the cells
            for i in scaled_objectness_mask_indices_pos:
                rect_pos = patches.Rectangle(
                    i[::-1],  # Flip x and y coordinates for matplotlib
                    cell_dims[0],
                    cell_dims[1],
                    linewidth=1,
                    edgecolor="black",
                    facecolor=(63 / 255, 255 / 255, 0 / 255, 75 / 255),  # lime color
                )
                ax.add_patch(rect_pos)

            for i in scaled_objectness_mask_indices_neg:
                rect_pos = patches.Rectangle(
                    i[::-1],  # Flip x and y coordinates for matplotlib
                    cell_dims[0],
                    cell_dims[1],
                    linewidth=1,
                    edgecolor="black",
                    facecolor=(255 / 255, 0 / 255, 0 / 255, 75 / 255),  # red color
                )
                ax.add_patch(rect_pos)

        if mask_name == "loss":
            indices_loss_mask_pos = np.dstack(np.where(loss_mask == loss_mask.max()))[0]
            indices_loss_mask_neg = np.dstack(np.where(loss_mask == loss_mask.min()))[0]

            scaled_loss_mask_indices_pos = indices_loss_mask_pos * np.array(cell_dims)
            scaled_loss_mask_indices_neg = indices_loss_mask_neg * np.array(cell_dims)

            if len(scaled_loss_mask_indices_pos.shape) == 1:
                scaled_loss_mask_indices_pos = np.expand_dims(scaled_loss_mask_indices_pos, axis=0)

            if len(scaled_loss_mask_indices_neg.shape) == 1:
                scaled_loss_mask_indices_neg = np.expand_dims(scaled_loss_mask_indices_neg, axis=0)

            for i in scaled_loss_mask_indices_pos:
                rect_loss_mask = patches.Rectangle(
                    i[::-1],  # Flip x and y coordinates for matplotlib
                    cell_dims[0],
                    cell_dims[1],
                    linewidth=1,
                    edgecolor="black",
                    facecolor=(63 / 255, 255 / 255, 0 / 255, 75 / 255),  # lime color
                )
                ax.add_patch(rect_loss_mask)

            for i in scaled_loss_mask_indices_neg:
                rect_loss_mask = patches.Rectangle(
                    i[::-1],  # Flip x and y coordinates for matplotlib
                    cell_dims[0],
                    cell_dims[1],
                    linewidth=1,
                    edgecolor="black",
                    facecolor=(255 / 255, 0 / 255, 0 / 255, 75 / 255),  # red color
                )
                ax.add_patch(rect_loss_mask)

    plt.show()


def show_patches_on_image(image, label, results):
    """Draw the given image with rectangles that indicate the position of the extracted patches. And the patches in separate plots

    Args:
        image: the image in RGB format [480, 640, 3]
        label: the label of the object
        results: the results from the patch extractor. Contains for each detected object
            a number of patch candidates,
            masks that indicate for each patch whether the center could be projected to the plane,
            and the normalized corner coordinates of each patch (y1, x1, y2, x2)

    """
    image_res = image.shape[0:-1]
    num_candidates = results[label]["patches"].shape[1]

    # Draw image with patches on top of it
    _, axes = plt.subplots()
    axes.imshow(image[..., 0] / 255, cmap="gray")

    for i, box in enumerate(results[label]["boxes"][0]):  # take index 0 to remove batch dimension
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
            edgecolor="lime",
            facecolor=(255 / 255, 123 / 255, 0 / 255, 0 / 255),
        )
        # Each patch has a number to identify the ordering
        axes.text(x=(coords[0] + 4.0), y=coords[1] + 17.0, s=i + 1, color="lime")
        axes.add_patch(rect)

    plt.title("Predicted Patches in Image.")

    # Draw the patch candidates in separate plots
    _, axes = plt.subplots(num_candidates)
    axes[0].imshow(results[label]["patches"][0, 0, ..., 0].numpy() / 255, cmap="gray")
    axes[1].imshow(results[label]["patches"][0, 1, ..., 0].numpy() / 255, cmap="gray")
    axes[2].imshow(results[label]["patches"][0, 2, ..., 0].numpy() / 255, cmap="gray")
    axes[3].imshow(results[label]["patches"][0, 3, ..., 0].numpy() / 255, cmap="gray")
    axes[4].imshow(results[label]["patches"][0, 4, ..., 0].numpy() / 255, cmap="gray")

    plt.show()
