"""Take multiple .tfrecords files, shuffles them and split in them into train, validation and test sets.
validation and test split as well as the saving directiory can be specified when executing the script

Usage:
    python split_dataset.py --save-dir data/ --val 0.2 --test 0.15

Dependencies:
    - tensorflow
    - utils.dataset
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

from pathlib import Path

import tensorflow as tf

from util import dataset as u_dataset
from util import dataset_io as u_dataset_io


def load_data(
    directory: Path, dataset_utils: u_dataset.DatasetUtils, val_split: float, test_split: float
):
    data = u_dataset_io.get_data_info(directory)

    dataset = u_dataset_io.get_dataset(data["file_names"], dataset_utils)

    num_samples = data["num_samples"]
    train_samples = round(num_samples * (1 - val_split - test_split))
    val_samples = round(num_samples * val_split)
    test_samples = round(num_samples - train_samples - val_samples)

    print("Number of samples: ", num_samples)
    print("Train Size: ", train_samples)
    print("Val Samples: ", val_samples)
    print("Test Samples: ", test_samples)

    dataset = dataset.shuffle(num_samples, seed=42)

    train_ds = dataset.take(train_samples)
    val_ds = dataset.skip(train_samples).take(val_samples)
    test_ds = dataset.skip(train_samples + val_samples)

    return {
        "train_ds": train_ds,
        "val_ds": val_ds,
        "test_ds": test_ds,
        "train_samples": train_samples,
        "val_samples": val_samples,
        "test_samples": test_samples,
    }


def write_file(
    src_dir: Path,
    save_dir: Path,
    image_res: list[int, int],
    cell_dims: list[int, int] = None,
    val_split: float = 0.2,
    test_split: float = 0.15,
):
    resolution = f"{image_res[1]}x{image_res[0]}"
    dataset_utils = u_dataset.DatasetUtils(
        u_dataset.DatasetConfig(input_dims=image_res, cell_dims=cell_dims)
    )

    # Load the dataset
    data = load_data(src_dir, dataset_utils, val_split, test_split)
    print("Dataset loaded.")

    # Write .tfrecords files
    print("Writing Train Dataset...")
    train_ds_file = (
        save_dir
        / resolution
        / f"train_ds_{data['train_samples']}({round(1 - test_split - val_split, 2)}).tfrecords"
    )
    os.makedirs(train_ds_file.parent, exist_ok=True)
    with tf.io.TFRecordWriter(train_ds_file.as_posix()) as writer:
        for sample in data["train_ds"]:
            example = u_dataset_io.make_example(dataset_utils, sample=sample)
            writer.write(example.SerializeToString())

    print("Writing Test Dataset...")
    test_ds_file = save_dir / resolution / f"test_ds_{data['test_samples']}({test_split}).tfrecords"

    with tf.io.TFRecordWriter(test_ds_file.as_posix()) as writer:
        for sample in data["test_ds"]:
            example = u_dataset_io.make_example(dataset_utils, sample=sample)
            writer.write(example.SerializeToString())

    print("Writing Validation Dataset...")
    val_ds_file = save_dir / resolution / f"val_ds_{data['val_samples']}({val_split}).tfrecords"

    with tf.io.TFRecordWriter(val_ds_file.as_posix()) as writer:
        for sample in data["val_ds"]:
            example = u_dataset_io.make_example(dataset_utils, sample=sample)
            writer.write(example.SerializeToString())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="This script takes multiple .tfrecords files and split them into train, val and test datasets"
    )
    # Directory where the .tfrecords files are saved to
    parser.add_argument("--src_dir", required=True)
    parser.add_argument("--save_dir", required=True)
    parser.add_argument("--val_split", type=float)
    parser.add_argument("--test_split", type=float)
    parser.add_argument(
        "--image_res",
        type=int,
        nargs=2,
        required=True,
        help="Image resolution as height and width. e. g. 480 640",
    )
    parser.add_argument(
        "--cell_dims",
        type=int,
        nargs=2,
        required=False,
        help="The dimensions of a cell in the cellgrid as height and width. e. g. 32 32",
    )

    args = parser.parse_args()

    write_file(
        Path(args.src_dir),
        Path(args.save_dir),
        args.image_res,
        args.cell_dims,
        args.val_split,
        args.test_split,
    )
    print("Done!")
