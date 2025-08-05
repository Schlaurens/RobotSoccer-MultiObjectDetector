import datetime

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
        filepath=f"checkpoints/{timestamp}/checkpoint-" + "{epoch:02d}.weights.h5",
        save_weights_only=True,
        monitor="val_total_loss",
        mode="min",
        save_best_only=True,
        verbose=0,
    )

    csv_logger = tf.keras.callbacks.CSVLogger(log_dir + "/log.csv", separator=",", append=True)

    # TODO: Implement Reduce Learning Rate on Plateau callback

    return [tensorboard_callback, checkpoint_callback, csv_logger]


def load_datasets(validation_split=0.3, batch_size=32):
    data = u_dataset.get_data_info(directory="/data/groundtruth")
    dataset = u_dataset.get_dataset(data["file_names"])

    num_samples = num_samples = data["num_samples"]
    train_samples = round(num_samples * (1 - validation_split))
    val_samples = round(num_samples * validation_split)

    print("Number of samples: ", num_samples)
    print("Train Size: ", train_samples)
    print("Val Samples: ", val_samples)

    dataset = dataset.shuffle(batch_size, seed=42)

    train_ds = dataset.take(train_samples)
    val_ds = dataset.skip(val_samples)

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
    validation_split = 0.3
    batch_size = 32

    dataset = load_datasets(validation_split, batch_size)

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
    )

    save_models(model, timestamp)

    return


if __name__ == "__main__":
    main()
