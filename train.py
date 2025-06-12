import tensorflow as tf

from train.models import FullModel
from util import dataset as u_dataset
from util import image as u_image


def get_dataset(directory):
    raw_dataset = tf.data.TFRecordDataset(directory)

    feature_description = {
        "image": tf.io.FixedLenFeature([], tf.string),
        "camera": tf.io.FixedLenFeature([], tf.string),
        "intrinsics": tf.io.FixedLenFeature([], tf.string),
        "objectness": tf.io.FixedLenFeature([], tf.string),
        "offsets": tf.io.FixedLenFeature([], tf.string),
        "loss_mask": tf.io.FixedLenFeature([], tf.string),
    }

    def _parse_tensor(serialized_tensor):
        data = {
            "image": tf.ensure_shape(
                tf.io.parse_tensor(serialized_tensor["image"], out_type=tf.uint8), [480, 320, 4]
            ),
            "camera": tf.ensure_shape(
                tf.io.parse_tensor(serialized_tensor["camera"], out_type=tf.float32), [3]
            ),
            "intrinsics": tf.ensure_shape(
                tf.io.parse_tensor(serialized_tensor["intrinsics"], out_type=tf.float32), [4]
            ),
            "objectness_mask": tf.ensure_shape(
                tf.io.parse_tensor(serialized_tensor["objectness"], out_type=tf.float32), [15, 20]
            ),
            "offsets": tf.ensure_shape(
                tf.io.parse_tensor(serialized_tensor["offsets"], out_type=tf.float32), [15, 20, 2]
            ),
            "loss_mask": tf.ensure_shape(
                tf.io.parse_tensor(serialized_tensor["loss_mask"], out_type=tf.float32), [15, 20]
            ),
        }
        return data

    def _parse_function(example_proto):
        # Parse the input tf.train.Example proto using the dictionary above.
        return _parse_tensor(tf.io.parse_single_example(example_proto, feature_description))

    return raw_dataset.map(_parse_function)


def main():
    batch_size = 32
    num_samples = 64

    train_ds = get_dataset("data/Joerg_Joerg_CompetitionWalk_GO2025__HULKs_1stHalf_5.tfrecords")

    train_ds = train_ds.shuffle(32, seed=42)
    train_ds = train_ds.batch(batch_size)
    train_ds = train_ds.repeat(-1)

    # Upper camera dimensions. Width is halved because of YUYV format
    model = FullModel(480, 320)
    model.compile(optimizer=tf.keras.optimizers.Adam())
    model.fit(x=train_ds, epochs=200, steps_per_epoch=num_samples // batch_size)


if __name__ == "__main__":
    main()
