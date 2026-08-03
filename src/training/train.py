import argparse
import datetime
import glob
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"

import tensorflow as tf
import yaml

from training.models import FullModel
from util import callbacks as u_callbacks
from util import dataset as u_dataset
from util import dataset_io as u_dataset_io
from util import image as u_image


def load_config(config_path):
    with open(config_path) as f:
        config = yaml.safe_load(f)
    return config


def log_config(timestamp: str, input_dims_str: str, config):
    # if the training is started from a checkpoint, do not log the config a in a new directory
    if config["training"]["from_checkpoint"]:
        return

    log_dir = Path(config["callbacks"]["log_dir"]) / input_dims_str / timestamp

    # Create a directory
    os.makedirs(log_dir, exist_ok=True)

    config["metadata"] = {
        "timestamp": timestamp,
        "python_version": sys.version,
        "tensorflow_version": tf.__version__,
    }
    log_file = f"{log_dir}/config.yaml"
    with open(log_file, "w") as f:
        # dump config file into log
        yaml.dump(config, f, sort_keys=False, default_flow_style=False)
    print(f"Configuration logged to {log_file}")


def get_callbacks(timestamp: str, input_dims_str: str, config):
    log_dir = Path(config["callbacks"]["log_dir"]) / input_dims_str / timestamp

    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

    checkpoint_callback = u_callbacks.CustomCheckpointCallback(
        filepath=f"models/checkpoints/{input_dims_str}/{timestamp}", overwrite=True, verbose=False
    )

    csv_logger = tf.keras.callbacks.CSVLogger(
        log_dir.as_posix() + "/log.csv", separator=",", append=True
    )

    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
        monitor=config["callbacks"]["reduce_lr"]["monitor"],
        mode=config["callbacks"]["reduce_lr"]["mode"],
        factor=config["callbacks"]["reduce_lr"]["factor"],
        patience=config["callbacks"]["reduce_lr"]["patience"],
        min_lr=config["callbacks"]["reduce_lr"]["min_lr"],
    )

    # Stop training when no improvement has happened
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor=config["callbacks"]["early_stopping"]["monitor"],
        mode=config["callbacks"]["early_stopping"]["mode"],
        patience=config["callbacks"]["early_stopping"]["patience"],
    )

    # Terminate training, when NaN loss is encountered
    terminate_on_nan = tf.keras.callbacks.TerminateOnNaN()

    return [
        tensorboard_callback,
        checkpoint_callback,
        csv_logger,
        reduce_lr,
        # early_stopping,
        terminate_on_nan,
    ]


def load_datasets(config, cpn, cpn_channels):
    batch_size = config["training"]["batch_size"]
    input_dims = config["model"]["cpn"]["input_dims"]
    dataset_utils = u_dataset.DatasetUtils(
        u_dataset.DatasetConfig(input_dims, cell_dims=config["model"]["cpn"]["cell_dims"])
    )

    path_to_train = glob.glob(
        (
            Path(config["data"]["path"])
            / f"{input_dims[1]}x{input_dims[0]}"
            / "train_ds*.tfrecords"
        ).as_posix()
    )

    path_to_val = glob.glob(
        (
            Path(config["data"]["path"]) / f"{input_dims[1]}x{input_dims[0]}" / "val_ds*.tfrecords"
        ).as_posix()
    )

    train_ds = u_dataset_io.get_dataset(path_to_train, dataset_utils)
    val_ds = u_dataset_io.get_dataset(path_to_val, dataset_utils)

    if cpn is not None and config["model"]["cpn"]["n_context"] == 0:

        def apply_cpn(sample):
            image = sample["image"]
            image_grayscale = u_image.convert_yuyv_to_yuv(image)[..., 0:1]

            enc_input = tf.expand_dims(image_grayscale if cpn_channels == 1 else image, axis=0)
            maps = cpn(enc_input, training=False)
            if isinstance(maps, list):  # If there is a context vector
                maps = dict(zip(cpn.output_names, maps, strict=True))
            else:
                maps = {
                    cpn.output_names[0]: maps
                }  # [B, H_out, W_out, 3] CPN results for the first category
            for k, v in maps.items():
                sample[f"cpn_{k}"] = tf.squeeze(v, axis=0)  # remove batch_dim
            return sample

        train_ds = (
            train_ds.map(
                apply_cpn, num_parallel_calls=tf.data.AUTOTUNE
            ).cache()  # Epoch 1: fill cache, ab Epoch 2: train with cache
        )
        val_ds = val_ds.map(apply_cpn, num_parallel_calls=tf.data.AUTOTUNE).cache()

    train_ds = train_ds.repeat(-1).shuffle(2000).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    val_ds = val_ds.repeat(-1).batch(batch_size).prefetch(tf.data.AUTOTUNE)

    # Get number of train/val samples from file name of .tfrecords file.
    train_samples = int(path_to_train[0].split("_")[2].split("(")[0])
    val_samples = int(path_to_val[0].split("_")[2].split("(")[0])
    print("Number of samples: ", train_samples + val_samples)
    print("Train Size: ", train_samples)
    print("Val Samples: ", val_samples)

    return {
        "train_ds": train_ds,
        "val_ds": val_ds,
        "train_samples": train_samples,
        "val_samples": val_samples,
    }


def main(config):
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    epochs = config["training"]["epochs"]
    initial_epoch = (
        config["training"]["load_checkpoint"]["initial_epoch"]
        if config["training"]["from_checkpoint"]
        else config["training"]["initial_epoch"]
    )
    batch_size = config["training"]["batch_size"]
    cpn_channels = config["model"]["cpn"]["channels_in"]

    if cpn_channels != 1:
        model_input_dims = config["model"]["cpn"]["input_dims"] // np.array((1, 2))
    else:
        model_input_dims = config["model"]["cpn"]["input_dims"]

    model_cell_dims = config["model"]["cpn"]["cell_dims"]
    cpn_architecture = config["model"]["cpn"]["architecture"]
    classifier_architecture = config["model"]["classifier"]["architecture"]
    train_cpn = config["model"]["cpn"]["train_cpn"]
    train_classifier = config["model"]["classifier"]["train_classifier"]

    input_dims_str = f"{config['model']['cpn']['input_dims'][0]}x{config['model']['cpn']['input_dims'][1]}"

    log_config(timestamp, input_dims_str, config)

    model = FullModel(
        cpn_architecture,
        classifier_architecture,
        *model_input_dims,
        cpn_channels=cpn_channels,
        cell_dims=model_cell_dims,
        n_context=config["model"]["cpn"]["n_context"],
        train_cpn=train_cpn,
        train_classifier=train_classifier,
        classifier_offsets=config["model"]["classifier"]["with_offsets"],
        n_meta=config["model"]["classifier"]["n_meta"],
        categories_config=config["categories"],
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=config["training"]["learning_rate"]),
        jit_compile=False,
    )

    if config["training"]["from_checkpoint"] or config["training"]["from_model"]:
        # ==== When loading an existing model ====
        if config["training"]["from_model"]:
            model_timestamp = config["training"]["load_model"]["timestamp"]

            filepath = os.path.join(
                config["training"]["load_model"]["filepath"], input_dims_str, model_timestamp
            )

            filename = model_timestamp
            cpn_only = config["training"]["load_model"]["cpn_only"]
            verbose = config["training"]["load_model"]["verbose"]

        # ==== When loading a checkpoint ====
        else:
            timestamp = config["training"]["load_checkpoint"]["timestamp"]
            filename = f"epoch_{initial_epoch}"
            filepath = os.path.join(
                config["callbacks"]["checkpoint_dir"], input_dims_str, timestamp, filename
            )
            cpn_only = config["training"]["load_checkpoint"]["cpn_only"]
            verbose = config["training"]["load_checkpoint"]["verbose"]

        model = FullModel.load(
            cpn_architecture,
            classifier_architecture,
            filepath=filepath,
            filename=filename,
            input_dims=model_input_dims,
            cpn_channels=cpn_channels,
            cell_dims=model_cell_dims,
            n_context=config["model"]["cpn"]["n_context"],
            train_cpn=train_cpn,
            train_classifier=train_classifier,
            classifier_offsets=config["model"]["classifier"]["with_offsets"],
            cpn_only=cpn_only,
            verbose=verbose,
            n_meta=config["model"]["classifier"]["n_meta"],
            learning_rate=config["training"]["learning_rate"],
            categories_config=config["categories"],
        )

    dataset = load_datasets(
        config, model.cpn if not train_cpn else None, cpn_channels=cpn_channels
    )

    callbacks = get_callbacks(timestamp, input_dims_str, config)

    model.fit(
        x=dataset["train_ds"],
        validation_data=dataset["val_ds"],
        epochs=epochs,
        steps_per_epoch=dataset["train_samples"] // batch_size,
        validation_steps=dataset["val_samples"] // batch_size,
        callbacks=callbacks,
        verbose=config["training"]["verbose"],
        initial_epoch=initial_epoch,
    )

    model.save(f"models/finished/{input_dims_str}", f"{timestamp}")

    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="This script starts the training process of the model."
    )
    parser.add_argument("config_file")
    args = parser.parse_args()

    config = load_config(f"settings/{args.config_file}")

    main(config)
