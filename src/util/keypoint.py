import tensorflow as tf


def are_coords_in_patch(coords: tf.Tensor, boxes: tf.Tensor, padding: float = 0.2) -> tf.Tensor:
    """Checks if a set of normalized coordinates are inside a patch.

    Args:
        coords: The normalized coordinates to check [B, ..., 2]
        boxes: The corner points of the patch (y1, x1, y2, x2) [B, ..., 4]
        padding: The amount of padding as a fraction of the width of the box. Coordinates at the edge of the box inside the padding do not count as "inside the box". Defaults to 0.2

    Returns:
        A binary tensor where every row is True when the corresponding coords are inside the box. Else False. [B, ..., ]
    """

    width = tf.math.subtract(boxes[..., 3], boxes[..., 1])
    padding = width * padding

    # Check if the x coordinates are in the range of the boxes
    x_check = tf.logical_and(
        tf.greater(coords[..., 0], boxes[..., 1] + padding),
        tf.less(coords[..., 0], boxes[..., 3] - padding),
    )  # [B, ..., ]

    # Check if the y coordinates are in the range of the boxes
    y_check = tf.logical_and(
        tf.greater(coords[..., 1], boxes[..., 0] + padding),
        tf.less(coords[..., 1], boxes[..., 2] - padding),
    )  # [B, ..., ]

    return tf.where(
        tf.logical_and(x_check, y_check),
        tf.constant([True], dtype=tf.bool),
        tf.constant([False], dtype=tf.bool),
    )  # [B, ..., ]
