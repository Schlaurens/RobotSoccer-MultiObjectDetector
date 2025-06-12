import tensorflow as tf

from train.models import FullModel


def get_dataset(directory):
    """Read and parse dataset from a given directory

    Args:
        directory: the directory that contains the .tfrecords files

    Returns:
        A parsed dataset where each sample consists of 6 tensors.
    """
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
        """Parse the feature tensors using tf.io.parse_tensors

        tf.ensure_shape ensures that the shape of the tensors is not unknown at runtime.

        Args:
            serialized_tensor: a dict of 6 serialized tensors

        Returns:
            a dict of 6 parsed tensors with known shapes.
        """
        return {
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

    def _parse_function(example_proto):
        """Parse the given example using tf.io.parse_single_example

        Uses the feature_description from above that maps the feature keys to it's datatype.
        Also parses the tensors that are inside the given example.

        Args:
            example_proto: a single serialized example

        Returns:
            the parsed example
        """
        # Parse the input tf.train.Example proto using the dictionary above.
        return _parse_tensor(tf.io.parse_single_example(example_proto, feature_description))

    return raw_dataset.map(_parse_function)


def main():
    # TODO: find number of all samples over all files (counter images in folders)
    batch_size = 32
    num_samples = 64

    # TODO: input list of all filenames in data folder
    train_ds = get_dataset("data/Joerg_Joerg_CompetitionWalk_GO2025__HULKs_1stHalf_5.tfrecords")

    train_ds = train_ds.shuffle(32, seed=42)
    train_ds = train_ds.batch(batch_size)

    # To counteract "Local rendezvous warning"
    train_ds = train_ds.repeat(-1)

    # Upper camera dimensions. Width is halved because of YUYV format
    model = FullModel(480, 320)
    model.compile(optimizer=tf.keras.optimizers.Adam())
    model.fit(x=train_ds, epochs=200, steps_per_epoch=num_samples // batch_size)


if __name__ == "__main__":
    main()
