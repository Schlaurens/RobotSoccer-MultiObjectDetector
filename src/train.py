import tensorflow as tf

from train.models import FullModel
from util import dataset as u_dataset


def main():
    data = u_dataset.get_data_info(directory="/data/groundtruth")
    dataset = u_dataset.get_dataset(data["file_names"])

    epochs = 200
    validation_split = 0.3
    batch_size = 32
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

    # Upper camera dimensions. Width is halved because of YUYV format
    model = FullModel(480, 320)
    model.compile(optimizer=tf.keras.optimizers.Adam())
    model.fit(
        x=train_ds,
        validation_data=val_ds,
        epochs=epochs,
        steps_per_epoch=train_samples // batch_size,
        validation_steps=val_samples // batch_size,
    )

    return model


if __name__ == "__main__":
    main()
