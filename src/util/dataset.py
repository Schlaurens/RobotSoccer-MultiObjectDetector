import glob
import json
from enum import Enum
from pathlib import Path

import keras
import numpy as np
import tensorflow as tf
from shapely.geometry import Point, Polygon

from . import image as u_image


class IntersectionType(Enum):
    NONE = 0
    L = 1
    T = 2
    X = 3


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


def get_masks(
    label=None, object_name=None, coordinates=None, input_dims=(480, 640), output_dims=(15, 20)
) -> tuple:
    """Return label masks that are used to train the encoder.

    Generate an offset mask that converts the image coordinates of the object into offsets relative
    to given cell dimensions
    An object mask that marks the cell where the center of
    the object is in.
    And a loss mask that indicted which cell should have an impact on the loss function.

    Input can be either (label and object_name) or coordinates.

    Args:
        label: label of the image
        object_name: name of the object to generate masks for
        coordinates: the image coordinates of the object [B, 2]
        input_dims: the full dimensions of the camera image.
        output_dims: the number of cells. Should be the same dimensions of encoder output

    Returns:
        a dictionary with all three masks.

    """

    def _empty_masks(ignore_sample: bool = False):
        """generate default empty masks for when there are no objects in the image.
        The offset_mask will contains only -1.0. This is an arbitrary value, that indicates that no object is in the image.
        The object_mask will only contain False values as there are no objects any of the cells.
        The loss_mask will only contain True values as no loss should be ignored.

        Returns:
            the masks in a dictionary
        """
        offsets = tf.cast(tf.fill((*output_dims, 2), -1), dtype=tf.float32)
        object_mask = tf.fill(output_dims, value=False)
        loss_mask = tf.fill(output_dims, value=not ignore_sample)
        # return offsets, object_mask, loss_mask
        return {"offsets": offsets, "object_mask": object_mask, "loss_mask": loss_mask}

    # Case 1: Direct coordinates provided
    # TODO: make possible with a list of cooordinates.
    if coordinates is not None:
        if tf.reduce_all(tf.math.equal(coordinates, -1.0)):
            return _empty_masks()
    # Case 2: Label and object_name provided
    elif label is not None and object_name is not None:
        if object_name not in label:
            return _empty_masks()
        # if object_name == "intersections":
        #     coordinates == label[object_name].
        # else:
        coordinates = list(label[object_name].values())[
            :2
        ]  # Only take x and y coordinates (ignore radius)
    else:
        raise ValueError("Either (label and object_name) or coordinates must be provided.")

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
    )  # (15, 20)

    # TODO: only (coordinates - cells) for coordiantes that are closest to the cell.
    # distance masks for each intersection type?
    #
    l_coords = coordinates["L"]  # (dict) coordinates for all L intersections from a coords dict
    t_coords = coordinates["T"]
    x_coords = coordinates["X"]

    l_distances = tf.sqrt(l_coords**2 + cells**2)
    t_distances = tf.sqrt(t_coords**2 + cells**2)
    x_distances = tf.sqrt(x_coords**2 + cells**2)
    
    
    offsets = coordinates - cells

    # Scale offsets to the output size
    offsets_scaled = offsets * scale

    # Mark all cells with true, where the value is between 0 and 1 (object is in that cell)
    object_mask = [[all(n >= 0 and n < 1 for n in x) for x in row] for row in offsets_scaled]

    classification_mask = _generate_classification_mask()

    loss_mask = _generate_loss_mask(object_mask)

    # return offsets_scaled, object_mask, loss_mask
    return {"offsets": offsets_scaled, "object_mask": object_mask, "loss_mask": loss_mask}


def _generate_object_mask(object_name, label, cells):
    """Generate the binary object_mask using the cell coverage values for each object_category.

    ===== Work in Progress =====

    If the IoU value of the object and the cell is greater than a specified threshold, that cell is marked with a 1.0. And
    0.0 otherwise.

    Args:
        object_name: _description_
        label: _description_
    """

    def _get_threshold(distance, min_threshold=0.1, max_threshold=0.75):
        # Do some linear interpolation for the threshold
        pass

    # Generate object_mask for ball
    if object_name == "ball":
        # Geometry object of the ball
        ball = Point([label[object_name]["x"], label[object_name]["y"]]).buffer(
            label[object_name]["radius"], 128
        )

        # All cells from the cell grid as shapely polygons
        cell_polygons = [
            Polygon(
                (
                    (coords[0], coords[1]),
                    (coords[0], coords[1] + 32),
                    (coords[0] + 32, coords[1] + 32),
                    (coords[0] + 32, coords[1]),
                )
            )
            for coords in cells.numpy().reshape(-1, 2)
        ]

        intersections = np.array([ball.intersection(p).area for p in cell_polygons]).reshape(15, 20)
        # unions = np.array([ball.union(p).area for p in polygons]).reshape(15, 20)
        cell_areas = np.array([p.area for p in cell_polygons]).reshape(15, 20)

        cell_coverage = np.divide(intersections, cell_areas)
        print(cell_coverage)

        # if the ball is inside any of the cells, then the object_mask is 1.0
        return cell_coverage > 0

    # Generate object_mask for penaltyMark
    if object_name == "penaltyMark":
        pass


def _generate_classification_mask():
    """Generate a mask that has a value in each cell that corresponds to the class type of the object category.
    Example:
    The classification mask for line intersections can have four values: NONE, L, T, X. The values 
    
    Returns:
        A mask like described above.
    """
    pass


def _generate_loss_mask(object_mask):
    """Generate a binary mask that is 0 in each cell where the loss function should be ignored and 1 everywhere else

    The loss function should be ignored when the presence of an object inside a cell in ambiguous. Whether this
    is the case can be determined by the IoU value of the object and the cell. If the object is just a 1 dimensional point
    (e. g. a penalty mark) the cell that contains the object coordinates is marked as one and the 8 cells surrounding it are  marked as 0 (just in case).

    Args:
        object_mask: the object mask

    Returns:
        A binary mask like described above.

    """

    # Loss mask for 1 dimensional objects

    # invert object_mask
    inverted_obj_mask = np.logical_not(np.array(object_mask))

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

    # Generate mask that is False if the offset is -1.0 and True else. The offset_cell is [-1.0, -1.0] if there are no objects in the image.
    mask = tf.cast(tf.math.not_equal(offset_cell, -1.0), dtype=tf.float32)

    # Get the output dims from the offset_mask
    output_dims = tf.cast(tf.shape(offset_mask)[-3:-1], dtype=tf.int32)
    scale = tf.cast(keras.ops.array(output_dims) / keras.ops.array(image_dims), dtype=tf.float32)

    # Scale the first offset_cell up
    coords = offset_cell / scale * mask

    # set coords to [-1.0, -1.0] if they were set to [0, 0] by the mask. This means the coords are [-1.0, -1.0] if there are no object in the image
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

    @tf.function
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

    @tf.function
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


def get_sample_at_index(batched_data, index, keep_batch=True):
    """Extracts the element at the given index from batched data, handling any number of object types. Leave the batch dimension intact

    Args:
        batched_data: one batched of the dataset
        index: the index of the sample in the batch that is to be returned
        keep_batch: Whether the batch dimension should be preserved. Defaults to True
    """

    def maybe_batch_dim(tensor):
        """If keep_batch=True take element at index and preserve batch dimension. If keep_batch=False do not add batch dimension"""
        element = tensor[index]
        return tf.expand_dims(element, axis=0) if keep_batch else element

    result = {
        "image": maybe_batch_dim(batched_data["image"]),
        "camera": maybe_batch_dim(batched_data["camera"]),
        "intrinsics": maybe_batch_dim(batched_data["intrinsics"]),
    }

    # Dynamically handle the object categories
    result.update(
        {
            category: {
                "object_mask": maybe_batch_dim(batched_data[category]["object_mask"]),
                "offset_mask": maybe_batch_dim(batched_data[category]["offset_mask"]),
                "loss_mask": maybe_batch_dim(batched_data[category]["loss_mask"]),
            }
            for category in batched_data
            if category not in ["image", "camera", "intrinsics"]  # Skip non-object fields
        }
    )

    return result


def make_example(directory, label):
    """Generate a Tensorflow example for a given data label. Tensorflow examples are used to serialize data into .tfrecords files.

    Args:
        directory: The directory of the data that is to be serialized.
        label: the labels that are to be serialized (dict)

    Returns:
        instance of tf.Example
    """

    masks_ball = get_masks(label, "ball")
    masks_penaltyMark = get_masks(label, "penaltyMark")
    image_feature = tf.train.Feature(
        bytes_list=tf.train.BytesList(
            value=[
                tf.io.serialize_tensor(
                    tf.reshape(
                        tf.constant(
                            load_image(directory, label, image_format=u_image.ImageFormat.YUYV)
                        ),
                        (480, 320, 4),
                    )
                ).numpy(),
            ]
        )
    )
    camera_feature = tf.train.Feature(
        bytes_list=tf.train.BytesList(
            value=[
                tf.io.serialize_tensor(
                    tf.constant(camera_from_label(label), dtype=tf.float32)
                ).numpy(),
            ]
        )
    )
    intrinsics_feature = tf.train.Feature(
        bytes_list=tf.train.BytesList(
            value=[
                tf.io.serialize_tensor(
                    tf.constant(intrinsics_from_label(label), dtype=tf.float32)
                ).numpy(),
            ]
        )
    )
    object_feature_ball = tf.train.Feature(
        bytes_list=tf.train.BytesList(
            value=[
                tf.io.serialize_tensor(
                    tf.reshape(tf.cast(masks_ball["object_mask"], dtype=tf.float32), (15, 20))
                ).numpy(),
            ]
        )
    )
    offset_feature_ball = tf.train.Feature(
        bytes_list=tf.train.BytesList(
            value=[
                tf.io.serialize_tensor(tf.reshape(masks_ball["offsets"], (15, 20, 2))).numpy(),
            ]
        )
    )
    loss_mask_feature_ball = tf.train.Feature(
        bytes_list=tf.train.BytesList(
            value=[
                tf.io.serialize_tensor(
                    tf.reshape(tf.cast(masks_ball["loss_mask"], dtype=tf.float32), (15, 20))
                ).numpy(),
            ]
        )
    )
    object_feature_penaltyMark = tf.train.Feature(
        bytes_list=tf.train.BytesList(
            value=[
                tf.io.serialize_tensor(
                    tf.reshape(
                        tf.cast(masks_penaltyMark["object_mask"], dtype=tf.float32), (15, 20)
                    )
                ).numpy(),
            ]
        )
    )
    offset_feature_penaltyMark = tf.train.Feature(
        bytes_list=tf.train.BytesList(
            value=[
                tf.io.serialize_tensor(
                    tf.reshape(masks_penaltyMark["offsets"], (15, 20, 2))
                ).numpy(),
            ]
        )
    )
    loss_mask_feature_penaltyMark = tf.train.Feature(
        bytes_list=tf.train.BytesList(
            value=[
                tf.io.serialize_tensor(
                    tf.reshape(tf.cast(masks_penaltyMark["loss_mask"], dtype=tf.float32), (15, 20))
                ).numpy(),
            ]
        )
    )

    # Create a Features dictionary
    features = tf.train.Features(
        feature={
            "image": image_feature,
            "camera": camera_feature,
            "intrinsics": intrinsics_feature,
            "object_ball": object_feature_ball,
            "offsets_ball": offset_feature_ball,
            "loss_mask_ball": loss_mask_feature_ball,
            "object_penaltyMark": object_feature_penaltyMark,
            "offsets_penaltyMark": offset_feature_penaltyMark,
            "loss_mask_penaltyMark": loss_mask_feature_penaltyMark,
        }
    )

    example = tf.train.Example(features=features)
    # print(example)

    return example


def make_example_from_sample(sample):
    """Generate a Tensorflow example for a given data sample. Tensorflow examples are used to serialize data into .tfrecords files.

    Args:
        sample: the sample that is to be serialized. The samples already contains all the data that is needed for training

    Returns:
        instance of tf.Example
    """

    image_feature = tf.train.Feature(
        bytes_list=tf.train.BytesList(
            value=[
                tf.io.serialize_tensor(sample["image"]).numpy(),
            ]
        )
    )
    camera_feature = tf.train.Feature(
        bytes_list=tf.train.BytesList(
            value=[
                tf.io.serialize_tensor(sample["camera"]).numpy(),
            ]
        )
    )
    intrinsics_feature = tf.train.Feature(
        bytes_list=tf.train.BytesList(
            value=[
                tf.io.serialize_tensor(sample["intrinsics"]).numpy(),
            ]
        )
    )
    object_feature_ball = tf.train.Feature(
        bytes_list=tf.train.BytesList(
            value=[
                tf.io.serialize_tensor(sample["ball"]["object_mask"]).numpy(),
            ]
        )
    )
    offset_feature_ball = tf.train.Feature(
        bytes_list=tf.train.BytesList(
            value=[
                tf.io.serialize_tensor(sample["ball"]["offset_mask"]).numpy(),
            ]
        )
    )
    loss_mask_feature_ball = tf.train.Feature(
        bytes_list=tf.train.BytesList(
            value=[
                tf.io.serialize_tensor(sample["ball"]["loss_mask"]).numpy(),
            ]
        )
    )
    object_feature_penaltyMark = tf.train.Feature(
        bytes_list=tf.train.BytesList(
            value=[
                tf.io.serialize_tensor(sample["penaltyMark"]["object_mask"]).numpy(),
            ]
        )
    )
    offset_feature_penaltyMark = tf.train.Feature(
        bytes_list=tf.train.BytesList(
            value=[
                tf.io.serialize_tensor(sample["penaltyMark"]["offset_mask"]).numpy(),
            ]
        )
    )
    loss_mask_feature_penaltyMark = tf.train.Feature(
        bytes_list=tf.train.BytesList(
            value=[
                tf.io.serialize_tensor(sample["penaltyMark"]["loss_mask"]).numpy(),
            ]
        )
    )

    # Create a Features dictionary
    features = tf.train.Features(
        feature={
            "image": image_feature,
            "camera": camera_feature,
            "intrinsics": intrinsics_feature,
            "object_ball": object_feature_ball,
            "offsets_ball": offset_feature_ball,
            "loss_mask_ball": loss_mask_feature_ball,
            "object_penaltyMark": object_feature_penaltyMark,
            "offsets_penaltyMark": offset_feature_penaltyMark,
            "loss_mask_penaltyMark": loss_mask_feature_penaltyMark,
        }
    )

    example = tf.train.Example(features=features)
    # print(example)

    return example
