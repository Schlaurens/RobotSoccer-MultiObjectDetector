import argparse
import datetime
import glob
import os
import sys

import tensorflow as tf
import yaml

from train.models import FullModel
from util import callbacks as u_callbacks
from util import dataset as u_dataset


def load_config(config_path):
    with open(config_path) as f:
        config = yaml.safe_load(f)
    return config


def log_config(timestamp: str, config):
    log_dir = os.path.join(config["callbacks"]["log_dir"], timestamp)

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


def get_callbacks(timestamp: str, config):
    log_dir = config["callbacks"]["log_dir"] + timestamp

    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

    checkpoint_callback = u_callbacks.CustomCheckpointCallback(
        filepath=f"checkpoints/{timestamp}", overwrite=True, verbose=False
    )

    csv_logger = tf.keras.callbacks.CSVLogger(log_dir + "/log.csv", separator=",", append=True)

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


def load_datasets(config):
    path_to_train = glob.glob(config["data"]["train_path"])
    path_to_val = glob.glob(config["data"]["val_path"])

    train_ds = u_dataset.get_dataset(path_to_train)
    val_ds = u_dataset.get_dataset(path_to_val)

    # Get number of train/val samples from file name of .tfrecords file.
    train_samples = int(path_to_train[0].split("_")[2].split("(")[0])
    val_samples = int(path_to_val[0].split("_")[2].split("(")[0])

    print("Number of samples: ", train_samples + val_samples)
    print("Train Size: ", train_samples)
    print("Val Samples: ", val_samples)

    batch_size = config["training"]["batch_size"]
    train_ds = train_ds.batch(batch_size)
    val_ds = val_ds.batch(batch_size)

    # To counteract "Local rendezvous warning"
    train_ds = train_ds.repeat(-1)
    val_ds = val_ds.repeat(-1)

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
        if config["training"]["load_checkpoint"]["from_checkpoint"]
        else config["training"]["initial_epoch"]
    )
    batch_size = config["training"]["batch_size"]
    model_input_dims = config["training"]["model_input_dims"]
    encoder_architecture = config["training"]["encoder_architecture"]
    only_train_encoder = config["training"]["only_train_encoder"]

    log_config(timestamp, config)

    dataset = load_datasets(config)

    # Upper camera dimensions. Width is halved because of YUYV format
    model = FullModel(
        encoder_architecture, *model_input_dims, only_train_encoder=only_train_encoder
    )
    model.compile(optimizer=tf.keras.optimizers.Adam(), jit_compile=False)

    # ==== When loading a checkpoint ====
    if config["training"]["load_checkpoint"]["from_checkpoint"]:
        timestamp = config["training"]["load_checkpoint"]["timestamp"]
        model = FullModel.load(
            input_dims=model_input_dims,
            filepath=f"{config['callbacks']['checkpoint_dir']}{timestamp}",  # TODO: do this with pathlib
            filename=f"epoch_{initial_epoch}.keras",
            encoder_only=config["training"]["load_checkpoint"]["encoder_only"],
            verbose=config["training"]["load_checkpoint"]["verbose"],
        )

    # ==== When loading from models ====
    if config["training"]["load_model"]["from_model"]:
        model_timestamp = config["training"]["load_model"]["timestamp"]
        model = FullModel.load(
            input_dims=model_input_dims,
            filepath=config["training"]["load_model"]["filepath"],
            filename=f"{model_timestamp}.keras",
            encoder_only=config["training"]["load_model"]["encoder_only"],
            verbose=config["training"]["load_model"]["verbose"],
        )

    callbacks = get_callbacks(timestamp, config)

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

    model.save("models", f"{timestamp}", only_save_encoder=only_train_encoder)

    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="This script starts the training process of the model."
    )
    parser.add_argument("config_file")
    args = parser.parse_args()

    config = load_config(f"settings/{args.config_file}")

    main(config)
