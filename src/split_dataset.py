"""Take multiple .tfrecords files, shuffles them and split in them into train, validation and test sets.
validation and test split as well as the saving directiory can be specified when executing the script

Usage:
    python split_dataset.py --dir data/ --val 0.2 --test 0.15

Dependencies:
    - tensorflow
    - utils.dataset
"""

import argparse

import tensorflow as tf

from util import dataset as u_dataset


def load_data(val_split, test_split):
    data = u_dataset.get_data_info(directory="/data/groundtruth")
    dataset = u_dataset.get_dataset(data["file_names"])

    num_samples = data["num_samples"]
    train_samples = round(num_samples * (1 - val_split - test_split))
    val_samples = round(num_samples * val_split)
    test_samples = round(num_samples - train_samples - val_samples)

    print("Number of samples: ", num_samples)
    print("Train Size: ", train_samples)
    print("Val Samples: ", val_samples)
    print("Test Samples: ", test_samples)

    dataset = dataset.shuffle(num_samples)

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


def write_file(directory, val_split=0.2, test_split=0.15):
    # Load the dataset
    train_ds, val_ds, test_ds = load_data(val_split, test_split)
    print("Dataset loaded.")

    # Write .tfrecords files
    print("Writing Train Dataset...")
    train_ds_file = directory + "train_ds.tfrecords"
    with tf.io.TFRecordWriter(train_ds_file) as writer:
        for sample in train_ds:
            example = u_dataset.make_example_from_sample(sample)
            writer.write(example.SerializeToString())

    print("Writing Test Dataset...")
    test_ds_file = directory + f"test_ds_{test_split}.tfrecords"
    with tf.io.TFRecordWriter(test_ds_file) as writer:
        for sample in test_ds:
            example = u_dataset.make_example_from_sample(sample)
            writer.write(example.SerializeToString())

    print("Writing Validation Dataset...")
    val_ds_file = directory + f"val_ds_{val_split}.tfrecords"
    with tf.io.TFRecordWriter(val_ds_file) as writer:
        for sample in val_ds:
            example = u_dataset.make_example_from_sample(sample)
            writer.write(example.SerializeToString())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="This script takes multiple .tfrecords files and split them into train, val and test datasets"
    )
    # Directory where the .tfrecords files are saved to
    parser.add_argument("--dir")
    parser.add_argument("--val", type=float)
    parser.add_argument("--test", type=float)

    args = parser.parse_args()

    write_file(args.dir, args.val, args.test)
    print("Done!")
