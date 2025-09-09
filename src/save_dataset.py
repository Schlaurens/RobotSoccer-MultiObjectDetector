import argparse
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


import tensorflow as tf

from util import dataset as u_dataset


def write_file(directory):
    # Load the dataset
    # TODO: must be divisible by 32
    labels = u_dataset.load_labels(directory)
    print("Dataset loaded.")

    record_file = directory + ".tfrecords"
    print("Writing file...")
    with tf.io.TFRecordWriter(record_file) as writer:
        for label in labels:
            example = u_dataset.make_example(directory, label)
            writer.write(example.SerializeToString())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="This script saves labels into a TFRecord file.")
    parser.add_argument("directory")
    args = parser.parse_args()

    write_file(args.directory)
    print("Done!")
