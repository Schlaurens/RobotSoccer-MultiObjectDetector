import argparse
import csv
import os
import sys

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import glob
from pathlib import Path

import tensorflow as tf
import yaml

from training.models import FullModel
from util import dataset as u_dataset
from util import dataset_io as u_dataset_io


def load_config(config_path: Path):
    with open(config_path) as f:
        config = yaml.safe_load(f)
    return config


def load_dataset(config):
    input_dims = config["model"]["encoder"]["input_dims"]

    dataset_utils = u_dataset.DatasetUtils(
        u_dataset.DatasetConfig(
            input_dims=input_dims,
            cell_dims=config["model"]["encoder"]["cell_dims"],
        )
    )

    path_to_test_data = glob.glob(
        Path("data/tfrecords/", f"{input_dims[1]}x{input_dims[0]}", "test_ds*.tfrecords").as_posix()
    )

    test_ds = u_dataset_io.get_dataset(path_to_test_data, dataset_utils)
    batch_size = 32
    test_ds = test_ds.batch(batch_size, drop_remainder=False)

    return test_ds


def load_model(config: dict, path_to_models: str, model_name: str):
    encoder_architecture = config["model"]["encoder"]["architecture"]
    classifier_architecture = config["model"]["classifier"]["architecture"]

    model = FullModel.load(
        encoder_architecture,
        classifier_architecture,
        filepath=path_to_models,
        filename=model_name,
        input_dims=config["model"]["encoder"]["input_dims"],
        cell_dims=config["model"]["encoder"]["cell_dims"],
        n_context=config["model"]["encoder"]["n_context"],
        only_train_encoder=config["model"]["encoder"]["only_train_encoder"],
        classifier_offsets=config["model"]["classifier"]["with_offsets"],
        encoder_only=True,
        verbose=True,
        n_meta=config["model"]["classifier"]["n_meta"],
        encoder_use_batch_norm=config["model"]["encoder"]["use_batch_norm"],
        classifier_use_batch_norm=config["model"]["classifier"]["use_batch_norm"],
        categories_config=config["categories"],
    )

    model.compile(optimizer=tf.keras.optimizers.Adam(), jit_compile=False)
    model.run_eagerly = True

    return model


def append_to_csv(file_path, data):
    """
    Append data to a CSV file. If the file does not exist, it will be created.

    Args:
        file_path (str): Path to the CSV file.
        data (dict): Dictionary containing the data to be written to the CSV file.
    """
    file_exists = os.path.isfile(file_path)

    if not file_exists:
        with open(file_path, mode="w", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=data.keys())
            writer.writeheader()
            writer.writerow(data)

    else:
        existing_data = []
        with open(file_path, newline="") as file:
            reader = csv.DictReader(file)
            for row in reader:
                existing_data.append(row)
        # Check if the timestamp already exists
        timestamp_exists = any(
            row["model_timestamp"] == data["model_timestamp"] for row in existing_data
        )

        if timestamp_exists:
            # Rewrite the file without the existing row
            with open(file_path, mode="w", newline="") as file:
                writer = csv.DictWriter(file, fieldnames=data.keys())
                writer.writeheader()
                for row in existing_data:
                    if row["model_timestamp"] != data["model_timestamp"]:
                        writer.writerow(row)

        # Write the new row
        with open(file_path, mode="a", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=data.keys())
            writer.writerow(data)


def create_metrics_csv(file_path, resolution, architecture, config, model_timestamp, metrics_list):
    """
    Create or append to a CSV file with the specified metrics data.

    Args:
        file_path (str): Path to the CSV file.
        resolution (str): Resolution of the model.
        architecture (str): Architecture of the model.
        model_timestamp (str): Timestamp of the model.
        config (dict): Configuration dictionary containing the architecture.
        metrics_list (list): List of metrics to be included in the CSV file.
    """
    data = {
        "resolution": resolution,
        "architecture": architecture,
        "input_channels": config["model"]["encoder"]["channels_in"],
        "model_timestamp": model_timestamp,
        "config_architecture": config["model"]["encoder"]["architecture"],
    }

    # Add each metric in metrics_list to the data dictionary
    for key, value in metrics_list:
        data[key] = value  # Assuming the metric value is the same as the metric name

    append_to_csv(file_path, data)


def main(args):
    model_name = f"{args.model_timestamp}.keras"

    resolution = glob.glob(os.path.join(args.log_dir, "*", args.model_timestamp))[0].split(os.sep)[
        -2
    ]
    architecture = glob.glob(
        os.path.join(args.model_dir, resolution, "*", f"{args.model_timestamp}.keras")
    )[0].split(os.sep)[-2]
    mode = args.log_dir.split("-")[-1]

    path_to_models = Path(args.model_dir, resolution, architecture)
    config_dir = Path(args.log_dir, resolution, args.model_timestamp)

    print("Loading Config...")
    config = load_config(Path(config_dir, "config.yaml"))

    print("Loading Model...")
    model = load_model(config, path_to_models, model_name)

    print("Loading Dataset...")
    test_ds = load_dataset(config)

    metrics_list = model.evaluate(x=test_ds, return_dict=True)

    # Save Metrics to YAML file
    metrics_to_save = {}
    for key, value in metrics_list.items():
        # if "encoder_recall@k" in key or "encoder_mae" in key:
        metrics_to_save[key] = float(value)

    with open(Path(config_dir, "cpn_metrics.yaml"), "w") as f:
        yaml.dump(metrics_to_save, f)

    create_metrics_csv(
        f"data/evaluation/cpn-{mode}.csv",
        resolution,
        architecture,
        config,
        args.model_timestamp,
        metrics_list.items(),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="This script compares the prediction of the model with the given the given timestamp and the current B-Human detectors."
    )
    parser.add_argument("model_timestamp", type=str)
    parser.add_argument("log_dir", type=str)
    parser.add_argument("model_dir", type=str)
    args = parser.parse_args()

    main(args)
