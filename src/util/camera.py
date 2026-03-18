import keras
import numpy as np
import tensorflow as tf


def camera_pose_to_vec(camera_pose):
    h = camera_pose[..., 2]

    rot_world_in_camera = rot_camera_in_world(camera_pose).T

    g_in_camera = rot_world_in_camera[..., -1]  # last column of rot_world_in_camera

    theta, phi = (
        np.acos(g_in_camera[2]),
        np.sign(g_in_camera[1])
        * np.acos(
            g_in_camera[0]
            / np.sqrt(g_in_camera[0] * g_in_camera[0] + g_in_camera[1] * g_in_camera[1])
        ),
    )

    return h, theta, phi


def vec_to_camera_pose(h, theta, phi):
    s_theta = np.sin(theta)
    # this is the third column of rot_world_in_camera or the third row of rot_camera_in_world
    g_in_camera = np.array([s_theta * np.cos(phi), s_theta * np.cos(phi), np.cos(theta)])

    # "Extending a Unit Vector to an Orthonormal Basis of 3-space", Tomas Möller and John F. Hughes
    i = 0 if np.abs(g_in_camera[0]) > np.abs(g_in_camera[1]) else 1
    v = np.zeros(3)
    v[i] = -g_in_camera[2]
    v[2] = g_in_camera[i]

    v /= np.linalg.norm(v)

    w = np.linalg.cross(g_in_camera, v)

    rot_world_in_camera = np.concatenate(w, v, g_in_camera)

    rot_camera_in_world = rot_world_in_camera.T
    # x, y are DoFs that don't need to be described here
    # - maybe add an auxiliary vector that has x,y and the third angle?
    trans_camera_in_world = np.array([0, 0, h])

    return rot_camera_in_world, trans_camera_in_world


def world_to_image(camera_pose, camera_intr, point_in_world):
    h, theta, phi = camera_pose_to_vec(camera_pose)
    rot_camera_in_world, trans_camera_in_world = vec_to_camera_pose(h, theta, phi)

    # point_in_camera = rot_world_in_camera @ point_in_world + trans_world_in_camera
    point_in_camera = rot_camera_in_world.T @ point_in_world + trans_camera_in_world.T
    if point_in_camera[0] < 1:  # mm
        return None
    point_in_camera /= point_in_camera[0]
    # px = cx - fx * point_in_camera[1]
    px = camera_intr[0] - camera_intr[2] * point_in_camera[1]
    py = camera_intr[1] - camera_intr[3] * point_in_camera[2]
    return np.array([px, py])  # TODO: check if outside [0,w|h]?


def image_to_world(
    camera: tf.Tensor | tuple[float],
    camera_intr: tf.Tensor | tuple[float],
    point_in_image: tf.Tensor | tuple[float],
    object_height: float = 0.0,
) -> tf.Tensor:
    """Transforms image coordinates in to world coodinates using the camera parameters

    Args:
        camera: A tuple of camera roll, pitch and height (B, 3)
        camera_intr: the intrinsic camera parameters (cx, cy, fx, fy) (B, 4)
        point_in_image: A tuple of the image coordinates (x, y) (B, 2)
        object_size: The size of the object that the image coordinates point to (in m). Defaults to 0.0.

    Returns:
        A vector in world coordinates of the given point. If a coordinate pair is invalid (is [-1, -1]) then the result is a 3d-Vector of [-1, -1, -1].  (B, 3)
    """

    # Ensure point_in_image, camera inputs and camera intrinsics are batched
    if isinstance(point_in_image, list | tuple):
        point_in_image = keras.ops.convert_to_tensor(point_in_image, dtype=tf.float32)
        if len(keras.ops.shape(point_in_image)) == 1:
            point_in_image = keras.ops.expand_dims(point_in_image, axis=0)
    if isinstance(camera, list | tuple):
        camera = keras.ops.convert_to_tensor(camera, dtype=tf.float32)
        if len(keras.ops.shape(camera)) == 1:
            camera = keras.ops.expand_dims(camera, axis=0)
    if isinstance(camera_intr, list | tuple):
        camera_intr = keras.ops.convert_to_tensor(camera_intr, dtype=tf.float32)
        if len(keras.ops.shape(camera_intr)) == 1:
            camera_intr = keras.ops.expand_dims(camera_intr, axis=0)

    invalid_mask = tf.reduce_all(point_in_image == -1.0, axis=-1)  # Shape: (B,)

    camera_height = camera[..., 2]  # [B, ]
    object_height = 0.5 * object_height

    dir_in_camera = tf.concat(
        [
            tf.ones_like(point_in_image[..., :1]),
            (camera_intr[..., :2] - point_in_image) / camera_intr[..., 2:],
        ],
        -1,
    )  # [B, 3]

    # Rotate the camera ray
    dir_in_world = keras.ops.einsum(
        "...ij,...j->...i", rot_camera_in_world(camera), dir_in_camera
    )  # [B, 3]

    # Find intersection with plane
    factor = keras.ops.nan_to_num(
        keras.ops.divide(object_height - camera_height, dir_in_world[..., 2])
    )  # [B, ]

    factor = keras.ops.divide(
        object_height - camera_height, dir_in_world[..., 2]
    )  # (B, ) -- may contain Inf/NaN

    valid_factor = tf.math.is_finite(factor) & (factor > 0)  # [B,]

    # If the point cannot be projected on the plane the position is [-1.0, -1.0, -1.0]
    position_in_world = keras.ops.where(
        valid_factor[..., tf.newaxis],
        factor[..., tf.newaxis] * dir_in_world,
        tf.fill(tf.shape(dir_in_world), -1.0),
    )

    # Set batch elements to [-1, -1, -1] where coordinates are invalid (are [-1, -1]).
    position_in_world_filtered = tf.where(
        tf.expand_dims(invalid_mask, axis=-1),
        tf.fill(tf.shape(dir_in_world), -1.0),
        position_in_world,
    )

    return position_in_world_filtered


def rot_camera_in_world(camera):
    """Converts the (roll, pitch, height) representation to a rotation matrix according to the rodrigues formula.

    :param camera: The extrinsic camera parameters (roll, pitch, height).
        [B, 3]
    :return: The corresponding rotation matrix.
        [B, 3, 3]
    """
    angle = keras.ops.norm(camera[..., :2], axis=-1)
    x = camera[..., 0] / angle
    y = camera[..., 1] / angle
    c, s = keras.ops.cos(angle), keras.ops.sin(angle)
    return keras.ops.stack(
        [
            keras.ops.stack([x * x * (1 - c) + c, x * y * (1 - c), y * s], axis=-1),
            keras.ops.stack([y * x * (1 - c), y * y * (1 - c) + c, -x * s], axis=-1),
            keras.ops.stack([-y * s, x * s, c], axis=-1),
        ],
        axis=-2,
    )
