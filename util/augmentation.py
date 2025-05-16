import copy

import albumentations as A  # noqa: N812
import numpy as np

from . import labels as u_labels

TRANSFORM = A.Compose(
    [
        A.HorizontalFlip(),
        A.SomeOf(
            [
                A.MultiplicativeNoise(multiplier=(0.8, 1.2), per_channel=True, elementwise=False),
                A.RGBShift(r_shift_limit=20, g_shift_limit=20, b_shift_limit=20),
                A.GaussianBlur(blur_limit=0, sigma_limit=(0.5, 3)),
                A.GaussNoise(std_range=(0, 1), per_channel=False),
            ],
            2,
            replace=False,
            p=0.5,
        ),
    ],
    keypoint_params=A.KeypointParams(format="xy", remove_invisible=False),
    is_check_shapes=False,
)


def _get_ball_keypoint(label):
    if u_labels.has_ball(label):
        x, y, _ = u_labels.get_ball(label)
        return (x, y)
    return (0, 0)


def _get_penalty_mark_keypoint(label):
    if u_labels.has_penalty_mark(label):
        x, y = u_labels.get_penalty_mark(label)
        return (x, y)
    return (0, 0)


def _get_obstacles_heatmap(label):
    if u_labels.has_obstacles(label):
        return np.asarray(u_labels.get_obstacles(label), dtype=np.float32)
    return np.zeros(
        (u_labels.OBSTACLES_HEIGHT, u_labels.OBSTACLES_WIDTH),
        dtype=np.float32,
    )


def _get_label_from_augmentables(label, keypoints, heatmap):
    label = copy.deepcopy(label)
    if u_labels.has_ball(label):
        u_labels.set_ball(label, keypoints[0][0], keypoints[0][1], u_labels.get_ball(label)[2])
    if u_labels.has_penalty_mark(label):
        u_labels.set_penalty_mark(label, keypoints[1][0], keypoints[1][1])
    if u_labels.has_obstacles(label):
        u_labels.set_obstacles_direct(label, heatmap.tolist())
    return label


def apply(image, label):
    print(image.dtype)
    transformed = TRANSFORM(
        image=image,
        mask=_get_obstacles_heatmap(label),
        keypoints=[_get_ball_keypoint(label), _get_penalty_mark_keypoint(label)],
    )

    return transformed["image"], _get_label_from_augmentables(
        label, transformed["keypoints"], transformed["mask"]
    )


def apply2(image, b, p, o):
    print(image.dtype)
    keypoints = b.tolist() + p.tolist()
    transformed = TRANSFORM(image=image, mask=o, keypoints=keypoints)
    return (
        transformed["image"],
        np.array(transformed["keypoints"][: len(b)], dtype=np.float32),
        np.array(transformed["keypoints"][-len(p) :], dtype=np.float32),
        transformed["mask"],
    )
