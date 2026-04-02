import os
import sys

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import argparse
from pathlib import Path

import tensorflow as tf
import yaml

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from training.models import FullModel


def load_model(config, path_to_model, model_name):
    print("Loading Model...")
    model = FullModel.load(
        encoder_architecture=config["model"]["encoder"]["architecture"],
        classifier_architecture=config["model"]["classifier"]["architecture"],
        input_dims=config["model"]["encoder"]["input_dims"],
        cell_dims=config["model"]["encoder"]["input_dims"],
        filepath=path_to_model,
        filename=model_name,
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

    return model


def load_config(config_path):
    print("Loading Config File...")
    with open(config_path) as f:
        config = yaml.safe_load(f)
    return config


def main(model_path: Path, save_path: Path) -> None:
    path_to_model = "/".join(model_path.split("/")[:-2])
    model_name = model_path.split("/")[-1]
    model_timestamp = model_name.split(".")[0]
    resolution_string = model_path.split("/")[-3]

    config = load_config(f"logs/fit/{resolution_string}/{model_timestamp}/config.yaml")
    model = load_model(config, path_to_model, model_name)

    model.save(filepath=save_path + f"/{resolution_string}", filename=model_timestamp)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="This script converts a .keras model to an .h5 model."
    )
    parser.add_argument("save_path", type=str)
    parser.add_argument("model_path", type=str)

    args = parser.parse_args()

    main(args.model_path, args.save_path)
