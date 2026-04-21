import argparse
import csv
import os
import sys

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import glob
from pathlib import Path

import numpy as np
import sklearn
import tensorflow as tf
import yaml

from training.models import FullModel
from util import dataset as u_dataset
from util import dataset_io as u_dataset_io
from util import metrics as u_metrics


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
    test_ds = test_ds.take(20)
    batch_size = 32
    test_ds = test_ds.batch(batch_size, drop_remainder=False)

    return test_ds


def load_model(config: dict, path_to_models: str, model_name: str):
    encoder_architecture = config["model"]["encoder"]["architecture"]
    classifier_architecture = config["model"]["classifier"]["architecture"]

    channels_in = config["model"]["encoder"].get("channels_in", 4)

    model = FullModel.load(
        encoder_architecture,
        classifier_architecture,
        filepath=path_to_models,
        filename=model_name,
        input_dims=config["model"]["encoder"]["input_dims"],
        encoder_channels=channels_in,
        cell_dims=config["model"]["encoder"]["cell_dims"],
        n_context=config["model"]["encoder"]["n_context"],
        only_train_encoder=config["model"]["encoder"]["only_train_encoder"],
        classifier_offsets=config["model"]["classifier"]["with_offsets"],
        encoder_only=False,
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
        "input_channels": config["model"]["encoder"].get("channels_in", 4),
        "model_timestamp": model_timestamp,
        "encoder_architecture": config["model"]["encoder"]["architecture"],
        "classifier_architecture": config["model"]["classifier"]["architecture"],
    }

    # Add each metric in metrics_list to the data dictionary
    for key, value in metrics_list:
        data[key] = value  # Assuming the metric value is the same as the metric name

    append_to_csv(file_path, data)


def evaluate_cpn(model, dataset):
    metrics_list = model.evaluate(x=dataset, return_dict=True)

    # Save Metrics to YAML file
    # metrics_to_save = {}
    # for key, value in metrics_list.items():
    #     if "encoder_recall_at_k" in key or "encoder_mae" in key:
    #         metrics_to_save[key] = float(value)

    # with open(Path(config_dir, "cpn_metrics.yaml"), "w") as f:
    #     yaml.dump(metrics_to_save, f)

    return metrics_list


def evaluate_classifier(model, dataset, config):
    def _get_metrics(
        predictions,
        groundtruth,
        config,
        thresholds: dict,
        threshold_mode: str,
        encoder_threshold: float = 0.1,
        nms_iou_threshold: float = None,
    ) -> dict:
        metrics = {}

        dataset_utils = u_dataset.DatasetUtils(
            u_dataset.DatasetConfig(
                input_dims=config["model"]["encoder"]["input_dims"],
                cell_dims=config["model"]["encoder"]["cell_dims"],
            )
        )

        for object in u_dataset.CategoryNames:
            if object.value not in predictions[0]["results"]:
                print(f"No {object.value} in results.")
                continue

            if object.name == u_dataset.CategoryNames.INTERSECTIONS.name:
                results = [
                    u_metrics.calculate_metrics(
                        dataset_utils,
                        predictions[0]["results"][object.value],
                        groundtruth[0][object.value],
                        config["categories"][object.value]["n_classes"],
                        cla,
                        encoder_threshold,
                        threshold_mode,
                        groundtruth[0]["camera"],
                        groundtruth[0]["intrinsics"],
                        config["categories"][object.value]["max_distance"],
                        iou_threshold=nms_iou_threshold,
                    )
                    for cla in thresholds[object.value]
                ]
            else:
                results = [
                    u_metrics.calculate_metrics(
                        dataset_utils,
                        predictions[0]["results"][object.value],
                        groundtruth[0][object.value],
                        config["categories"][object.value]["n_classes"],
                        cla,
                        encoder_threshold,
                        threshold_mode,
                        groundtruth[0]["camera"],
                        groundtruth[0]["intrinsics"],
                        config["categories"][object.value]["max_distance"],
                        config["categories"][object.value]["padding"],
                    )
                    for cla in thresholds[object.value]
                ]

            if len(thresholds[object.value]) == 1:
                metrics[object.value] = results[0]
            else:
                metrics[object.value] = results

        return metrics

    def _calculate_ap_metrics(metrics_threshold_range_additive):
        metrics = {object.value: {} for object in u_dataset.CategoryNames}

        for object in u_dataset.CategoryNames:
            if object.value not in metrics_threshold_range_additive:
                continue

            precision_additive = np.array(
                [x["precision_pooled"] for x in metrics_threshold_range_additive[object.value]]
            )
            recall_additive = np.array(
                [x["recall_pooled"] for x in metrics_threshold_range_additive[object.value]]
            )

            sorted_indices_additive = np.argsort(recall_additive)
            recall_additive_sorted = recall_additive[sorted_indices_additive]
            precision_additive_sorted = precision_additive[sorted_indices_additive]

            ap_additive = sklearn.metrics.auc(recall_additive_sorted, precision_additive_sorted)

            metrics[object.value] = {
                "precision": precision_additive_sorted,
                "recall": recall_additive_sorted,
                "ap": ap_additive,
            }

        return metrics

    groundtruth_dataset = []
    for x in dataset:
        groundtruth_dataset.append(x)

    # Take only image, camera, intrinsics for input
    input_data = dataset.map(lambda x: (x["image"], x["camera"], x["intrinsics"]))

    # Convert tf.Data.Dataset into list
    input_list = []
    for x, y, z in input_data:
        input_list.append((x, y, z))

    predictions = [model.predict(x=batch) for batch in input_list]

    nms_iou_threshold = None
    encoder_threshold = 0.0
    threshold_range_additive = np.linspace(0, 1 + 1, num=100)

    classifier_threshold_ranges_additive = {
        u_dataset.CategoryNames.BALL.value: threshold_range_additive,
        u_dataset.CategoryNames.PENALTYMARK.value: threshold_range_additive,
        u_dataset.CategoryNames.INTERSECTIONS.value: threshold_range_additive,
    }

    metrics_threshold_range_additive = _get_metrics(
        predictions,
        groundtruth_dataset,
        config,
        classifier_threshold_ranges_additive,
        "additive",
        encoder_threshold,
        nms_iou_threshold,
    )

    return _calculate_ap_metrics(metrics_threshold_range_additive)


def main(args):
    eval_cpn = args.cpn
    eval_classifier = args.classifier

    config_dir = glob.glob(
        os.path.join(args.log_dir, "**", args.model_timestamp, "config.yaml"), recursive=True
    )[0]

    print("Loading Config...")
    config = load_config(Path(config_dir))

    print("Loading Dataset...")
    test_ds = load_dataset(config)

    resolution = f"{config['model']['encoder']['input_dims'][0]}x{config['model']['encoder']['input_dims'][1]}"
    cpn_architecture = config["model"]["encoder"]["architecture"]
    classifier_architecture = config["model"]["classifier"]["architecture"]

    mode = args.log_dir.split("-")[-1]

    if eval_cpn:
        architecture = glob.glob(
            os.path.join(args.model_dir, resolution, "**", f"{args.model_timestamp}.keras"),
            recursive=True,
        )[0].split(os.sep)[-3]

        path_to_models = Path(args.model_dir, resolution, architecture)

        print("Loading Model...")
        model = load_model(config, path_to_models, args.model_timestamp)

        cpn_metrics = evaluate_cpn(model, test_ds)
        create_metrics_csv(
            f"data/evaluation/cpn-{mode}.csv",
            resolution,
            cpn_architecture,
            config,
            args.model_timestamp,
            cpn_metrics.items(),
        )

    if eval_classifier:
        path_to_models = Path(args.model_dir, args.model_timestamp)

        print("Loading Model...")
        model = load_model(config, path_to_models, args.model_timestamp)

        classifier_metrics = evaluate_classifier(model, test_ds, config)

        metrics_to_save = {}
        for category in u_dataset.CategoryNames:
            if category.value in classifier_metrics:
                # metrics_to_save[f"{category.value}_precision"] = classifier_metrics[category.value]["precision"]
                # metrics_to_save[f"{category.value}_recall"] = classifier_metrics[category.value]["recall"]
                metrics_to_save[f"{category.value}_ap"] = classifier_metrics[category.value]["ap"]

        create_metrics_csv(
            f"data/evaluation/classifier-{mode}.csv",
            resolution,
            classifier_architecture,
            config,
            args.model_timestamp,
            metrics_to_save.items(),
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="This script compares the prediction of the model with the given the given timestamp and the current B-Human detectors."
    )
    parser.add_argument("--model_timestamp", type=str)
    parser.add_argument("--log_dir", type=str)
    parser.add_argument("--model_dir", type=str)
    parser.add_argument("--cpn", type=bool, default=False, required=False)
    parser.add_argument("--classifier", type=bool, default=False, required=False)
    args = parser.parse_args()

    main(args)
