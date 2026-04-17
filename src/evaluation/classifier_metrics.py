import os
import re
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


def extract_batch_size_from_path(path: str) -> int | None:
    # Match the number before the parentheses
    match = re.search(r"test_ds_(\d+)\(\d+\.\d+\)\.tfrecords", path)
    if match:
        return int(match.group(1))
    return None


def load_dataset(config, dataset_utils):
    input_dims = config["model"]["encoder"]["input_dims"]

    path_to_test_data = glob.glob(
        Path("data/tfrecords/", f"{input_dims[1]}x{input_dims[0]}", "test_ds*.tfrecords").as_posix()
    )

    test_ds = u_dataset_io.get_dataset(path_to_test_data, dataset_utils)

    batch_size = extract_batch_size_from_path(path_to_test_data[0])  # takes the first glob match
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
        encoder_channels=config["model"]["encoder"]["channels_in"],
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


def get_metrics(
    predictions,
    groundtruth,
    dataset_utils,
    config,
    thresholds: dict,
    threshold_mode: str,
    encoder_threshold: float = 0.1,
    nms_iou_threshold: float = None,
) -> dict:
    metrics = {}

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


def calculate_ap_metrics(metrics_threshold_range_additive):
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
        
        metrics[object.value] = {"precision": precision_additive_sorted, "recall": recall_additive_sorted, "ap": ap_additive}
        
    return metrics


def main():
    model_name = "20260410-230858.keras"
    resolution = "288x384"

    model_timestamp = model_name.split(".")[0]
    path_to_models = Path("models/finished", resolution)
    config_dir = Path("logs/fit", resolution, model_timestamp)

    print("Loading Config...")
    config = load_config(Path(config_dir, "config.yaml"))

    dataset_utils = u_dataset.DatasetUtils(
        u_dataset.DatasetConfig(
            input_dims=config["model"]["encoder"]["input_dims"],
            cell_dims=config["model"]["encoder"]["cell_dims"],
        )
    )

    print("Loading Model...")
    model = load_model(config, path_to_models, model_timestamp)

    print("Loading Dataset...")
    test_ds = load_dataset(config, dataset_utils)

    groundtruth_dataset = []
    for x in test_ds:
        groundtruth_dataset.append(x)

    # Take only image, camera, intrinsics for input
    input_data = test_ds.map(lambda x: (x["image"], x["camera"], x["intrinsics"]))

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

    metrics_threshold_range_additive = get_metrics(
        predictions,
        groundtruth_dataset,
        dataset_utils,
        config,
        classifier_threshold_ranges_additive,
        "additive",
        encoder_threshold,
        nms_iou_threshold,
    )
    
    final_metrics = calculate_ap_metrics(metrics_threshold_range_additive)
    
    print(final_metrics)
    
if __name__ == "__main__":
    main()
