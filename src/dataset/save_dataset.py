import argparse
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

from pathlib import Path

import tensorflow as tf

from util import dataset_io as u_dataset_io


def write_file(directory):
    # Load the dataset
    labels = u_dataset_io.load_labels(directory)
    print("Dataset loaded.")

    record_file = Path(directory).with_suffix(".tfrecords")
    print("Writing file...")
    with tf.io.TFRecordWriter(str(record_file)) as writer:
        for label in labels:
            example = u_dataset_io.make_example(directory, label)
            writer.write(example.SerializeToString())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="This script saves labels into a TFRecord file.")
    parser.add_argument("directory")
    args = parser.parse_args()

    write_file(args.directory)
    print("Done!")
