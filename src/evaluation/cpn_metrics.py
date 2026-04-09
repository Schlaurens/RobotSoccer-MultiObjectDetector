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


def main():
    model_name = "20260401-191434.keras"
    resolution = "240x320"
    architecture = "cpn_3"

    model_timestamp = model_name.split(".")[0]
    path_to_models = Path("models/final_cpn_test", resolution, architecture)
    config_dir = Path("logs/fit", resolution, model_timestamp)

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

    # Print Metrics
    print("===== Metrics =====")
    print("=== Recall@k ===")
    for key, value in metrics_list.items():
        if "encoder_recall@k" in key:
            print(f"{key}: {value:.7f}")

    print("=== MAE ===")
    for key, value in metrics_list.items():
        if "encoder_mae" in key:
            print(f"{key}: {value:.7f}")


if __name__ == "__main__":
    main()
