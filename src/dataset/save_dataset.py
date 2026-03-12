import argparse
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

from pathlib import Path

import tensorflow as tf

from util import dataset as u_dataset
from util import dataset_io as u_dataset_io


def write_file(
    source: Path, destination: Path, image_res: list[int, int], cell_dims: list[int, int] = None
):
    dataset_utils = u_dataset.DatasetUtils(
        u_dataset.DatasetConfig(input_dims=image_res, cell_dims=cell_dims)
    )
    destination = destination / f"{image_res[1]}x{image_res[0]}"
    # Load the dataset
    labels = u_dataset_io.load_labels(source)
    print("Dataset loaded.")

    record_file = (destination / source.name).with_suffix(".tfrecords")
    os.makedirs(destination, exist_ok=True)

    print("Writing file...")
    with tf.io.TFRecordWriter(str(record_file)) as writer:
        for label in labels:
            example = u_dataset_io.make_example(dataset_utils, source, label)
            writer.write(example.SerializeToString())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="This script saves labels into a TFRecord file.")
    parser.add_argument("--src_dir", type=str, help="The source groundtruth directory.")
    parser.add_argument("--dest_dir", type=str, help="The destination directory.")
    parser.add_argument(
        "--image_res", type=int, nargs=2, help="Image resolution as height and width. e. g. 480 640"
    )
    args = parser.parse_args()

    write_file(Path(args.src_dir), Path(args.dest_dir), args.image_res)
    print("Done!")
