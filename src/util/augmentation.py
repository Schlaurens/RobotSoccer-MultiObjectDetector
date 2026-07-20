import tensorflow as tf

from . import dataset as u_dataset


def augment_cell_indices(
    dataset_config: u_dataset.DatasetConfig,
    indices: tf.Tensor,
    minval: int = -1,
    maxval: int = 2,
    seed: int = 42,
):
    delta_row = tf.random.uniform(
        tf.shape(indices), minval=minval, maxval=maxval, seed=seed, dtype=tf.int32
    )
    delta_col = tf.random.uniform(
        tf.shape(indices), minval=minval, maxval=maxval, seed=seed, dtype=tf.int32
    )

    flat_shift = delta_row * dataset_config.output_dims[1] + delta_col
    patch_indices_augmented = indices + flat_shift

    patch_indices_augmented_clipped = tf.clip_by_value(
        patch_indices_augmented, 0, tf.reduce_prod(dataset_config.output_dims) - 1
    )

    return patch_indices_augmented_clipped
