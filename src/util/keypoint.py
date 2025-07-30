import keras
import tensorflow as tf


@tf.function
def get_coords_from_offsets(offset_mask, image_dims=(480, 640)) -> tuple:
    """Extract the image coordinates from the offset mask

    Args:
        mask: the offset mask [B, H, W, 2]
        image_dims: the dimensions of the input image

    Returns:
        The coordinates of the object (x, y). (-1.0, -1.0) if the object is not in the image
    """
    # TODO: implement solution for offset_masks with multiple objects

    # Take the offset from the first cell of each offset_mask
    offset_cell = offset_mask[..., 0, 0, :]  # [B, 2]

    # Generate mask that is [0, 0] at every index where the offsets are other than [-1.0, -1.0]. And [1, 1] everywhere else.
    mask = tf.where(
        tf.math.equal(offset_cell, -1.0),
        tf.zeros_like(offset_cell, dtype=tf.float32),
        tf.ones_like(offset_cell, dtype=tf.float32),
    )  # [B, 2]

    # Get the output dims from the offset_mask
    output_dims = tf.cast(tf.shape(offset_mask)[-3:-1], dtype=tf.int32)

    scale = tf.cast(keras.ops.array(output_dims) / keras.ops.array(image_dims), dtype=tf.float32)

    # Scale the first offset_cell up
    coords = offset_cell / scale * mask

    # Put [-1.0, -1.0] at every index where the coords were set to [0, 0] by the mask.
    coords_masked = tf.where(tf.math.equal(coords, [0, 0]), tf.fill([2], -1.0), coords)

    return coords_masked


def are_coords_in_patch(coords: tf.Tensor, boxes: tf.Tensor) -> tf.Tensor:
    """_summary_

    Args:
        coords: the coordinates to check [B, ..., 2]
        boxes: the corner points of the patch (y1, x1, y2, x2) [B, ..., 4]

    Returns:
        A binary tensor where every row is True when the corresponding coords are inside the box. Else False. [B, ..., ]
    """

    # Check if the x coordinates are in the range of the boxes
    x_check = tf.logical_and(
        tf.greater(coords[..., 0], boxes[..., 1]), tf.less(coords[..., 0], boxes[..., 3])
    )  # [B, ..., ]
    # Check if the y coordinates are in the range of the boxes
    y_check = tf.logical_and(
        tf.greater(coords[..., 1], boxes[..., 0]), tf.less(coords[..., 1], boxes[..., 2])
    )  # [B, ..., ]

    # tf.print("x_check: ", tf.shape(x_check))
    # tf.print("y_check: ", tf.shape(y_check))
    return tf.where(
        tf.logical_and(x_check, y_check),
        tf.constant([True], dtype=tf.bool),
        tf.constant([False], dtype=tf.bool),
    )  # [B, ..., ]
