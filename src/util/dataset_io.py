import glob
import json
import os
from pathlib import Path

import numpy as np
import tensorflow as tf

from . import dataset as u_dataset
from . import image as u_image

dataset_utils = u_dataset.DatasetUtils(u_dataset.DatasetConfig())


def get_label_path(directory: str) -> str:
    """Just a helper function to get the label path.

    Args:
        directory: directory of the dataset

    Returns:
        path to the JSON file with the labels

    """
    return Path(directory) / "labels.json"


def get_image_path(directory: str, name: str) -> str:
    """Get the path to an image with a given name from a given directory.

    Args:
        directory: directory of the dataset
        name: name of the image

    Returns:
        path to the image

    """
    return Path(directory) / f"{name}.jpg"


def save_labels(directory: str, labels: str):
    """Save labels to a JSON file in a given directory.

    Args:
        directory: path to where to save the labels.json file.
        labels: dictionary with labels to be saved

    """
    with Path(get_label_path(directory)).open("w") as f:
        json.dump(labels, f, indent=0)


def load_labels(directory: str) -> dict:
    """Load labels from a given directory.

    Args:
        directory: directory of the dataset

    Returns:
        json file with the labels of a log file

    """
    with Path(get_label_path(directory)).open() as f:
        return json.load(f)


def load_image(directory: str, label: dict, **kwargs) -> np.ndarray:
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


def load_image_direct(path: str, **kwargs) -> np.ndarray:
    """Return an image from a direct path.

    Args:
        path: path to the image file
        **kwargs: image format

    Returns:
        the image

    """
    with Path(path).open("rb") as f:
        return u_image.load_bhuman_jpeg_image(f.read(), **kwargs)


def camera_from_label(label: dict) -> tuple[float, float, float]:
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


def intrinsics_from_label(label: dict) -> tuple[float, float, float, float]:
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


def log_name_from_label(label: dict) -> str:
    """Returns the name of log which the label is from.

    Args:
        label: A label from the dataset

    Returns:
        The log name.
    """

    return "_".join(label["name"].numpy().decode("utf-8").split("_")[0:-1])


def image_name_from_label(label: dict) -> str:
    """Returns the name of image that this label is for.

    Args:
        label: A label from the dataset

    Returns:
        The image name.
    """

    return label["name"].numpy().decode("utf-8").split("_")[-1]


def get_sample_name(label: dict, directory: str) -> str:
    """Returns a unique identifier for the label that corresponds to the log and image

    Args:
        label: The label from the dataset
        directory: The directory of the label

    Returns:
        A unique identifier. Consists of the image name and the log name.
    """
    image_name = label["name"]
    log_name = os.path.normpath(directory).split(os.sep)[-1]

    return f"{log_name}_{image_name}"


def get_dataset(directory: str) -> tf.data.Dataset:
    """Read and parse dataset from a given directory

    Args:
        directory: the directory that contains the .tfrecords files

    Returns:
        A parsed dataset where each sample consists of 6 tensors.
    """
    raw_dataset = tf.data.TFRecordDataset(directory)

    feature_description = {
        "name": tf.io.FixedLenFeature([], tf.string),
        "frame_time": tf.io.FixedLenFeature([], tf.string),
        "image": tf.io.FixedLenFeature([], tf.string),
        "camera": tf.io.FixedLenFeature([], tf.string),
        "intrinsics": tf.io.FixedLenFeature([], tf.string),
        "object_ball": tf.io.FixedLenFeature([], tf.string),
        "offsets_ball": tf.io.FixedLenFeature([], tf.string),
        "loss_mask_ball": tf.io.FixedLenFeature([], tf.string),
        "object_penaltyMark": tf.io.FixedLenFeature([], tf.string),
        "offsets_penaltyMark": tf.io.FixedLenFeature([], tf.string),
        "loss_mask_penaltyMark": tf.io.FixedLenFeature([], tf.string),
        "object_intersections": tf.io.FixedLenFeature([], tf.string),
        "offsets_intersections": tf.io.FixedLenFeature([], tf.string),
        "loss_mask_intersections": tf.io.FixedLenFeature([], tf.string),
        "classification_intersections": tf.io.FixedLenFeature([], tf.string),
    }

    @tf.function
    def _parse_tensor(serialized_tensor: dict[str, str]) -> dict[str, tf.Tensor]:
        """Parse the feature tensors using tf.io.parse_tensors

        tf.ensure_shape ensures that the shape of the tensors is not unknown at runtime.

        Args:
            serialized_tensor: a dict of 6 serialized tensors of type string

        Returns:
            a dict of 6 parsed tensors with known shapes.
        """
        return {
            "name": tf.ensure_shape(
                tf.io.parse_tensor(serialized_tensor["name"], out_type=tf.string), []
            ),
            "frame_time": tf.ensure_shape(
                tf.io.parse_tensor(serialized_tensor["frame_time"], out_type=tf.int32), []
            ),
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
            "intersections": {
                "object_mask": tf.ensure_shape(
                    tf.io.parse_tensor(
                        serialized_tensor["object_intersections"], out_type=tf.float32
                    ),
                    [15, 20],
                ),
                "offset_mask": tf.ensure_shape(
                    tf.io.parse_tensor(
                        serialized_tensor["offsets_intersections"], out_type=tf.float32
                    ),
                    [15, 20, 2],
                ),
                "loss_mask": tf.ensure_shape(
                    tf.io.parse_tensor(
                        serialized_tensor["loss_mask_intersections"], out_type=tf.float32
                    ),
                    [15, 20],
                ),
                "classification_mask": tf.ensure_shape(
                    tf.io.parse_tensor(
                        serialized_tensor["classification_intersections"], out_type=tf.float32
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


def get_data_info(directory: str = "data"):
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


def get_sample_at_index(batched_data: dict[str, tf.Tensor], index: int, keep_batch: bool = True):
    """Extracts the element at the given index from batched data, handling any number of object types. Leave the batch dimension intact

    Args:
        batched_data: one batch of the dataset
        index: the index of the sample in the batch that is to be returned
        keep_batch: Whether the batch dimension should be preserved. Defaults to True
    """

    def maybe_batch_dim(tensor):
        """If keep_batch=True take element at index and preserve batch dimension. If keep_batch=False do not add batch dimension"""
        element = tensor[index]
        return tf.expand_dims(element, axis=0) if keep_batch else element

    result = {
        "name": maybe_batch_dim(batched_data["name"]),
        "frame_time": maybe_batch_dim(batched_data["frame_time"]),
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
                "classification_mask": maybe_batch_dim(batched_data[category]["loss_mask"])
                if category == "intersections"
                else None,
            }
            for category in batched_data
            if category
            not in ["name", "frame_time", "image", "camera", "intrinsics"]  # Skip non-object fields
        }
    )

    return result


def make_example(directory: str = None, label: dict = None, sample: dict = None):
    """Generate a Tensorflow example for a given data label or sample. Tensorflow examples are used to serialize data into .tfrecords files.

    Args:
        directory: The directory of the data that is to be serialized.
        label: The labels that are to be serialized (dict)
        sample: The sample that is to be serialized. The samples already contains all the data that is needed for training

    Returns:
        instance of tf.Example
    """
    # TODO: use dataset.config for tensor dimensions
    if directory is not None and label is not None:
        from_sample = False

        masks_ball = dataset_utils.get_masks(label, u_dataset.CategoryNames.BALL.value)
        masks_penaltyMark = dataset_utils.get_masks(
            label, u_dataset.CategoryNames.PENALTYMARK.value
        )
        masks_intersections = dataset_utils.get_masks(
            label, u_dataset.CategoryNames.INTERSECTIONS.value
        )

    elif sample is not None:
        from_sample = True

    else:
        raise ValueError("Either (directory and label) or sample must be provided.")

    name_feature = tf.train.Feature(
        bytes_list=tf.train.BytesList(
            value=[
                tf.io.serialize_tensor(sample["name"]).numpy(),
            ]
            if from_sample
            else [
                tf.io.serialize_tensor(
                    tf.constant(get_sample_name(label, directory), dtype=tf.string)
                ).numpy()
            ]
        )
    )
    frame_time_feature = tf.train.Feature(
        bytes_list=tf.train.BytesList(
            value=[
                tf.io.serialize_tensor(sample["frame_time"]).numpy(),
            ]
            if from_sample
            else [tf.io.serialize_tensor(label["frame_time"]).numpy()]
        )
    )
    image_feature = tf.train.Feature(
        bytes_list=tf.train.BytesList(
            value=[
                tf.io.serialize_tensor(sample["image"]).numpy(),
            ]
            if from_sample
            else [
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
                tf.io.serialize_tensor(sample["camera"]).numpy(),
            ]
            if from_sample
            else [
                tf.io.serialize_tensor(
                    tf.constant(camera_from_label(label), dtype=tf.float32)
                ).numpy(),
            ]
        )
    )
    intrinsics_feature = tf.train.Feature(
        bytes_list=tf.train.BytesList(
            value=[
                tf.io.serialize_tensor(sample["intrinsics"]).numpy(),
            ]
            if from_sample
            else [
                tf.io.serialize_tensor(
                    tf.constant(intrinsics_from_label(label), dtype=tf.float32)
                ).numpy(),
            ]
        )
    )
    object_feature_ball = tf.train.Feature(
        bytes_list=tf.train.BytesList(
            value=[
                tf.io.serialize_tensor(sample["ball"]["object_mask"]).numpy(),
            ]
            if from_sample
            else [
                tf.io.serialize_tensor(
                    tf.reshape(tf.cast(masks_ball["object_mask"], dtype=tf.float32), (15, 20))
                ).numpy(),
            ]
        )
    )
    offset_feature_ball = tf.train.Feature(
        bytes_list=tf.train.BytesList(
            value=[
                tf.io.serialize_tensor(sample["ball"]["offset_mask"]).numpy(),
            ]
            if from_sample
            else [
                tf.io.serialize_tensor(tf.reshape(masks_ball["offsets"], (15, 20, 2))).numpy(),
            ]
        )
    )
    loss_mask_feature_ball = tf.train.Feature(
        bytes_list=tf.train.BytesList(
            value=[
                tf.io.serialize_tensor(sample["ball"]["loss_mask"]).numpy(),
            ]
            if from_sample
            else [
                tf.io.serialize_tensor(
                    tf.reshape(tf.cast(masks_ball["loss_mask"], dtype=tf.float32), (15, 20))
                ).numpy(),
            ]
        )
    )
    object_feature_penaltyMark = tf.train.Feature(
        bytes_list=tf.train.BytesList(
            value=[
                tf.io.serialize_tensor(sample["penaltyMark"]["object_mask"]).numpy(),
            ]
            if from_sample
            else [
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
                tf.io.serialize_tensor(sample["penaltyMark"]["offset_mask"]).numpy(),
            ]
            if from_sample
            else [
                tf.io.serialize_tensor(
                    tf.reshape(masks_penaltyMark["offsets"], (15, 20, 2))
                ).numpy(),
            ]
        )
    )
    loss_mask_feature_penaltyMark = tf.train.Feature(
        bytes_list=tf.train.BytesList(
            value=[
                tf.io.serialize_tensor(sample["penaltyMark"]["loss_mask"]).numpy(),
            ]
            if from_sample
            else [
                tf.io.serialize_tensor(
                    tf.reshape(tf.cast(masks_penaltyMark["loss_mask"], dtype=tf.float32), (15, 20))
                ).numpy(),
            ]
        )
    )
    object_feature_intersections = tf.train.Feature(
        bytes_list=tf.train.BytesList(
            value=[
                tf.io.serialize_tensor(sample["intersections"]["object_mask"]).numpy(),
            ]
            if from_sample
            else [
                tf.io.serialize_tensor(
                    tf.reshape(
                        tf.cast(masks_intersections["object_mask"], dtype=tf.float32), (15, 20)
                    )
                ).numpy(),
            ]
        )
    )
    offset_feature_intersections = tf.train.Feature(
        bytes_list=tf.train.BytesList(
            value=[
                tf.io.serialize_tensor(sample["intersections"]["offset_mask"]).numpy(),
            ]
            if from_sample
            else [
                tf.io.serialize_tensor(
                    tf.reshape(
                        tf.cast(masks_intersections["offsets"], dtype=tf.float32), (15, 20, 2)
                    )
                ).numpy(),
            ]
        )
    )
    loss_mask_feature_intersections = tf.train.Feature(
        bytes_list=tf.train.BytesList(
            value=[
                tf.io.serialize_tensor(sample["intersections"]["loss_mask"]).numpy(),
            ]
            if from_sample
            else [
                tf.io.serialize_tensor(
                    tf.reshape(
                        tf.cast(masks_intersections["loss_mask"], dtype=tf.float32), (15, 20)
                    )
                ).numpy(),
            ]
        )
    )
    classification_feature_intersections = tf.train.Feature(
        bytes_list=tf.train.BytesList(
            value=[
                tf.io.serialize_tensor(sample["intersections"]["classification_mask"]).numpy(),
            ]
            if from_sample
            else [
                tf.io.serialize_tensor(
                    tf.reshape(
                        tf.cast(masks_intersections["classification_mask"], dtype=tf.float32),
                        (15, 20),
                    )
                ).numpy(),
            ]
        )
    )

    # Create a Features dictionary
    features = tf.train.Features(
        feature={
            "name": name_feature,
            "frame_time": frame_time_feature,
            "image": image_feature,
            "camera": camera_feature,
            "intrinsics": intrinsics_feature,
            # ball
            "object_ball": object_feature_ball,
            "offsets_ball": offset_feature_ball,
            "loss_mask_ball": loss_mask_feature_ball,
            # penaltyMark
            "object_penaltyMark": object_feature_penaltyMark,
            "offsets_penaltyMark": offset_feature_penaltyMark,
            "loss_mask_penaltyMark": loss_mask_feature_penaltyMark,
            # intersections
            "object_intersections": object_feature_intersections,
            "offsets_intersections": offset_feature_intersections,
            "loss_mask_intersections": loss_mask_feature_intersections,
            "classification_intersections": classification_feature_intersections,
        }
    )

    example = tf.train.Example(features=features)

    return example
