import json
from pathlib import Path

import numpy as np
import tensorflow as tf

from . import image as u_image


def get_label_path(directory):
    """Just a helper function to get the label path.

    Args:
        directory: directory of the dataset

    Returns:
        path to the JSON file with the labels

    """
    return Path(directory) / "labels.json"


def get_image_path(directory, name):
    """Get the path to an image with a given name from a given directory.

    Args:
        directory: directory of the dataset
        name: name of the image

    Returns:
        path to the image

    """
    return Path(directory) / f"{name}.jpg"


def save_labels(directory, labels):
    """Save labels to a JSON file in a given directory.

    Args:
        directory: path to where to save the labels.json file.
        labels: dictionary with labels to be saved

    """
    with Path(get_label_path(directory)).open("w") as f:
        json.dump(labels, f, indent=0)


def load_labels(directory):
    """Load labels from a given directory.

    Args:
        directory: directory of the dataset

    Returns:
        json file with the labels of a log file

    """
    with Path(get_label_path(directory)).open() as f:
        return json.load(f)


def load_image(directory, label, **kwargs):
    """Load image from a given directory and label.

    Args:
        directory: directory of the dataset
        label: corresponding label of the image
        **kwargs: image format

    Returns:
        the image

    """
    with Path(get_image_path(directory, label["name"])).open("rb") as f:
        return u_image.load_bhuman_jpeg_image(f.read(), **kwargs)


def load_image_direct(path, **kwargs):
    """Return an image from a direct path.

    Args:
        path: path to the image file
        **kwargs: image format

    Returns:
        the image

    """
    with Path(path).open("rb") as f:
        return u_image.load_bhuman_jpeg_image(f.read(), **kwargs)


def camera_from_label(label):
    """Calculate the camera roll pitch and height from the camera pose in the data.

    Args:
        label: the label with the camera pose

    Returns:
        A tuple of roll, pitch and height.
    """
    alpha = np.arccos(label["cpose"]["z"][2])
    if np.abs(alpha) < 0.01:
        roll = pitch = 0
    else:
        sin_alpha = np.sqrt(1 - label["cpose"]["z"][2] * label["cpose"]["z"][2])
        roll = label["cpose"]["z"][1] / sin_alpha * alpha
        pitch = -label["cpose"]["z"][0] / sin_alpha * alpha
    height = label["cpose"]["h"] * 0.001
    return (roll, pitch, height)


def intrinsics_from_label(label):
    """
    Get the camera intrinsics from the label.

    Args:
        label: A label from the dataset

    Returns:
        The camera intrinsics as a tuple (cx, cy, fx, fy).
    """

    return (
        label["cintr"]["cx"],
        label["cintr"]["cy"],
        label["cintr"]["fx"],
        label["cintr"]["fy"],
    )


def get_masks(label, object_name, input_dims=(480, 640), output_dims=(15, 20)) -> tuple:
    """Return label masks that are used to train the encoder.

    Generate an offset mask that converts the image coordinates of the object into offsets relative
    to given cell dimensions. And and objectsness mask that marks the cell where the center of
    the object is in.

    Args:
        label: label of the image
        object_name: name of the object to generate masks for
        input_dims: the full dimensions of the camera image.
        output_dims: the number of cells. Should be the same dimensions of encoder output

    Returns:
        an array of shape [output_dims_x, output_dims_y, 2]. Where the offset for each cell is
        portrayed in x and y coordinates.

    """
    if object_name not in label:
        # If there are no objects of interest in the image all the offset get an arbitrary value.
        # In the loss function all offsets of cell without an object are ignored anyway.
        offsets = tf.cast(tf.fill((*output_dims, 2), -1), dtype=tf.float32)

        # All cell are marked as false, as there are no objects in the whole image.
        objectness_mask = tf.fill(output_dims, value=False)

        # All cells are marked as true, as there are no objects in the image and therefore no loss should be ignored.
        loss_mask = tf.fill(output_dims, value=True)

        return offsets, objectness_mask, loss_mask

    coordinates = list(label[object_name].values())[
        :-1
    ]  # Only take x and y coordinates (ignore radius)

    # Make sure that input_dims are divisible by output_dims
    cell_dims = np.array(input_dims) // np.array(output_dims)
    scale = np.array(output_dims) / np.array(input_dims)

    # Generate the cell grid in the full image scale
    # (values point to upper left corner of each cell)
    cells = tf.cast(
        tf.stack(
            tf.meshgrid(
                range(input_dims[1])[:: cell_dims[1]], range(input_dims[0])[:: cell_dims[0]]
            ),
            axis=-1,
        ),
        dtype=tf.float32,
    )

    offsets = coordinates - cells

    # Scale offsets to the output size
    offsets_scaled = offsets * scale

    # Mark all cells with true, where the value is between 0 and 1 (object is in that cell)
    objectness_mask = [[all(n >= 0 and n < 1 for n in x) for x in row] for row in offsets_scaled]

    loss_mask = _generate_loss_mask(objectness_mask)

    return offsets_scaled, objectness_mask, loss_mask


def _generate_loss_mask(objectness_mask):
    """Generate a binary mask that is 0 in each cell where the loss function should be ignored and 1 everywhere else

    The loss function should be ignored when the presence of an object inside a cell in ambiguous. Whether this
    is the case can be determined by the IoU value of the object and the cell. If the object is just a 1 dimensional point
    (e. g. a penalty mark) the cell that contains the object coordinates is marked as one and the 8 cells surrounding it are  marked as 0 (just in case).

    Args:
        objectness_mask: the objectness mask

    Returns:
        A binary mask like described above.

    """

    # Loss mask for 1 dimensional objects

    # invert objectness_mask
    inverted_obj_mask = np.logical_not(np.array(objectness_mask))

    # get index
    index = np.unravel_index(inverted_obj_mask.argmin(), inverted_obj_mask.shape)

    # turn the cells surrounding the index cell to 0
    inverted_obj_mask[index] = 1.0

    # TODO: use a more elegant way to set the surrounding cells to 0, like convolution or einsum
    for i in range(-1, 2):
        for j in range(-1, 2):
            if i == 0 and j == 0:
                continue
            # Check boundries
            if (0 <= index[0] + i < inverted_obj_mask.shape[0]) and (
                0 <= index[1] + j < inverted_obj_mask.shape[1]
            ):
                # Set the surrounding cells to 0
                inverted_obj_mask[index[0] + i, index[1] + j] = 0.0

    return inverted_obj_mask
