import glob
import json
from pathlib import Path

import keras
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
        :2
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


@tf.function
def get_coords_from_offsets(offset_mask, image_dims=(480, 640)) -> tuple:
    """Extract the image coordinates from the offset mask

    Args:
        mask: the offset mask [B, H, W, 2]
        image_dims: the dimensions of the input image

    Returns:
        The coordinates of the object (x, y). (-1.0, -1.0) if the object is not in the image
    """
    # TODO: implement solution for offset_masks with multiple objects

    # Take the offset from the first cell of each offset_mask
    offset_cell = offset_mask[..., 0, 0, :]  # [B, 2]

    # Generate mask that is [0, 0] at every index where the offsets are other than [-1.0, -1.0]. And [1, 1] everywhere else.
    mask = tf.where(
        tf.math.equal(offset_cell, -1.0),
        tf.zeros_like(offset_cell, dtype=tf.float32),
        tf.ones_like(offset_cell, dtype=tf.float32),
    )  # [B, 2]

    # Get the output dims from the offset_mask
    output_dims = tf.cast(tf.shape(offset_mask)[-3:-1], dtype=tf.int32)

    scale = tf.cast(keras.ops.array(output_dims) / keras.ops.array(image_dims), dtype=tf.float32)

    # Scale the first offset_cell up
    coords = offset_cell / scale * mask

    # Put [-1.0, -1.0] at every index where the coords were set to [0, 0] by the mask.
    coords_masked = tf.where(tf.math.equal(coords, [0, 0]), tf.fill([2], -1.0), coords)

    return coords_masked


def get_dataset(directory):
    """Read and parse dataset from a given directory

    Args:
        directory: the directory that contains the .tfrecords files

    Returns:
        A parsed dataset where each sample consists of 6 tensors.
    """
    raw_dataset = tf.data.TFRecordDataset(directory)

    feature_description = {
        "image": tf.io.FixedLenFeature([], tf.string),
        "camera": tf.io.FixedLenFeature([], tf.string),
        "intrinsics": tf.io.FixedLenFeature([], tf.string),
        "object_ball": tf.io.FixedLenFeature([], tf.string),
        "offsets_ball": tf.io.FixedLenFeature([], tf.string),
        "loss_mask_ball": tf.io.FixedLenFeature([], tf.string),
        "object_penaltyMark": tf.io.FixedLenFeature([], tf.string),
        "offsets_penaltyMark": tf.io.FixedLenFeature([], tf.string),
        "loss_mask_penaltyMark": tf.io.FixedLenFeature([], tf.string),
    }

    def _parse_tensor(serialized_tensor):
        """Parse the feature tensors using tf.io.parse_tensors

        tf.ensure_shape ensures that the shape of the tensors is not unknown at runtime.

        Args:
            serialized_tensor: a dict of 6 serialized tensors

        Returns:
            a dict of 6 parsed tensors with known shapes.
        """
        return {
            "image": tf.ensure_shape(
                tf.io.parse_tensor(serialized_tensor["image"], out_type=tf.uint8), [480, 320, 4]
            ),
            "camera": tf.ensure_shape(
                tf.io.parse_tensor(serialized_tensor["camera"], out_type=tf.float32), [3]
            ),
            "intrinsics": tf.ensure_shape(
                tf.io.parse_tensor(serialized_tensor["intrinsics"], out_type=tf.float32), [4]
            ),
            "ball": {
                "object_mask": tf.ensure_shape(
                    tf.io.parse_tensor(serialized_tensor["object_ball"], out_type=tf.float32),
                    [15, 20],
                ),
                "offset_mask": tf.ensure_shape(
                    tf.io.parse_tensor(serialized_tensor["offsets_ball"], out_type=tf.float32),
                    [15, 20, 2],
                ),
                "loss_mask": tf.ensure_shape(
                    tf.io.parse_tensor(serialized_tensor["loss_mask_ball"], out_type=tf.float32),
                    [15, 20],
                ),
            },
            "penaltyMark": {
                "object_mask": tf.ensure_shape(
                    tf.io.parse_tensor(
                        serialized_tensor["object_penaltyMark"], out_type=tf.float32
                    ),
                    [15, 20],
                ),
                "offset_mask": tf.ensure_shape(
                    tf.io.parse_tensor(
                        serialized_tensor["offsets_penaltyMark"], out_type=tf.float32
                    ),
                    [15, 20, 2],
                ),
                "loss_mask": tf.ensure_shape(
                    tf.io.parse_tensor(
                        serialized_tensor["loss_mask_penaltyMark"], out_type=tf.float32
                    ),
                    [15, 20],
                ),
            },
        }

    def _parse_function(example_proto):
        """Parse the given example using tf.io.parse_single_example

        Uses the feature_description from above that maps the feature keys to it's datatype.
        Also parses the tensors that are inside the given example.

        Args:
            example_proto: a single serialized example

        Returns:
            the parsed example
        """
        # Parse the input tf.train.Example proto using the dictionary above.
        return _parse_tensor(tf.io.parse_single_example(example_proto, feature_description))

    return raw_dataset.map(_parse_function)


def get_data_info(directory="data"):
    """Get all .tfrecords files and the number of all samples across all selected data files.

    Args:
        directory: The directory of the .tfrecords files used for training.

    Returns:
        A dict where "file_names" is a list of paths to .tfrecords files and "num_samples is the number of samples across all data files
    """
    file_names = []
    num_samples = 0

    for file in glob.glob(f"./{directory}/*.tfrecords"):
        file_names.append(file)

        # Count samples
        num_samples += len(glob.glob(f"{file.removesuffix('.tfrecords')}/*.jpg"))

    return {"file_names": file_names, "num_samples": num_samples}
