import datetime
import glob

import tensorflow as tf

from train.models import FullModel
from util import dataset as u_dataset


def save_models(model, timestamp: str) -> None:
    # Save Encoder
    model.get_layer("encoder").save(f"models/encoder/encoder_{timestamp}.keras")

    # Save Classifier
    model.get_layer("classifier").save(f"models/classifier/classifier_{timestamp}.keras")


def get_callbacks(timestamp: str):
    log_dir = "logs/fit/" + timestamp

    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=f"checkpoints/{timestamp}/checkpoint-" + "{epoch:03d}.weights.h5",
        save_weights_only=True,
        monitor="val_total_loss",
        mode="min",
        save_best_only=False,
        verbose=0,
    )

    csv_logger = tf.keras.callbacks.CSVLogger(log_dir + "/log.csv", separator=",", append=True)

    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_total_loss", mode="min", factor=0.2, patience=10, min_lr=0.0
    )

    # Stop training when no improvement has happened
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor="val_total_loss", mode="min", patience=40
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


def load_datasets(batch_size=32):
    path_to_train = glob.glob("data/train_ds*.tfrecords")
    path_to_val = glob.glob("data/val_ds*.tfrecords")

    train_ds = u_dataset.get_dataset(path_to_train)
    val_ds = u_dataset.get_dataset(path_to_val)

    # Get number of train/val samples from file name of .tfrecords file.
    train_samples = int(path_to_train[0].split("_")[2].split("(")[0])
    val_samples = int(path_to_val[0].split("_")[2].split("(")[0])

    print("Number of samples: ", train_samples + val_samples)
    print("Train Size: ", train_samples)
    print("Val Samples: ", val_samples)

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


def main():
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    epochs = 200
    batch_size = 32

    dataset = load_datasets(batch_size)

    # Upper camera dimensions. Width is halved because of YUYV format
    model = FullModel(480, 320)
    model.compile(optimizer=tf.keras.optimizers.Adam(), jit_compile=False)

    callbacks = get_callbacks(timestamp)

    model.fit(
        x=dataset["train_ds"],
        validation_data=dataset["val_ds"],
        epochs=epochs,
        steps_per_epoch=dataset["train_samples"] // batch_size,
        validation_steps=dataset["val_samples"] // batch_size,
        callbacks=callbacks,
        verbose=0,
    )

    save_models(model, timestamp)

    return


if __name__ == "__main__":
    main()
