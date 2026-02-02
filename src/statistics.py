import itertools
import os

import numpy as np
import tensorflow as tf

from util import camera as u_camera
from util import dataset as u_dataset
from util import dataset_io as u_dataset_io
from util import labels as u_labels


def get_bce_baseline(p: int, n: int):
    """Calculates the baseline binary cross entropy that would accur if the classifier would simply guess.

    Args:
        p: Number of positive samples.
        n: Number of samples.

    Returns:
        The baseline binary crossentropy.
    """
    pos_percentage = p / n
    return -(
        pos_percentage * np.log(pos_percentage) + (1 - pos_percentage) * np.log(1 - pos_percentage)
    )


def _clean_up_list(input):
    filtered = [x for x in input if x is not None]
    flattened = [
        item for d in filtered for item in (d if isinstance(d, (list | np.ndarray)) else [d])
    ]

    return flattened


def get_distance_from_label(
    label: dict, object_name: str, object_height: float, intersection_type: str = None
):
    if object_name not in label:
        return None
    if len(label[object_name]) == 0:
        return None

    camera = u_dataset_io.camera_from_label(label)
    intrinsics = u_dataset_io.intrinsics_from_label(label)

    if object_name == u_dataset.CategoryNames.INTERSECTIONS.value:
        if intersection_type is None:
            raise ValueError("The intersection type is not specified.")

        image_coords = [[coord["x"], coord["y"]] for coord in label[object_name][intersection_type]]
        if label[object_name]["ignore_sample"]:
            return None

        if len(image_coords) == 0:
            return None

    else:
        image_coords = (label[object_name]["x"], label[object_name]["y"])

    world_coords = u_camera.image_to_world(camera, intrinsics, image_coords, object_height)

    if tf.reduce_all(world_coords == -1.0):
        return None
    distance = tf.linalg.norm(world_coords, axis=-1, keepdims=True)

    return tf.reshape(distance, [-1]).numpy()


def main(data_path: str, calculate_distances: bool = False):
    label_dirs = [dir[0] for dir in os.walk(data_path)][1:]
    labels = [u_dataset_io.load_labels(dir) for dir in label_dirs]

    log_names = [label_dir.split("/")[-1] for label_dir in label_dirs]

    labels_concat = list(itertools.chain.from_iterable(labels))

    number_of_samples = len(labels_concat)
    number_of_intersection_samples = len(
        [
            sample
            for sample in labels_concat
            if u_labels.has_intersections(sample) and not sample["intersections"]["ignore_sample"]
        ]
    )
    number_of_non_empty_samples = len(
        [
            _
            for _ in labels_concat
            if u_labels.has_ball(_) or u_labels.has_penalty_mark(_) or u_labels.has_intersections(_)
        ]
    )
    number_of_ball_samples = len([_ for _ in labels_concat if u_labels.has_ball(_)])
    number_of_penalty_mark_samples = len([_ for _ in labels_concat if u_labels.has_penalty_mark(_)])

    number_of_l_intersections_for_each_log = [
        sum(
            [
                len(sample["intersections"][u_labels.IntersectionType.L.value])
                for sample in x
                if u_labels.has_intersections(sample)
            ]
        )
        for x in labels
    ]
    number_of_t_intersections_for_each_log = [
        sum(
            [
                len(sample["intersections"][u_labels.IntersectionType.T.value])
                for sample in x
                if u_labels.has_intersections(sample)
            ]
        )
        for x in labels
    ]
    number_of_x_intersections_for_each_log = [
        sum(
            [
                len(sample["intersections"][u_labels.IntersectionType.X.value])
                for sample in x
                if u_labels.has_intersections(sample)
            ]
        )
        for x in labels
    ]

    number_of_ignored_intersection_samples_for_each_log = [
        sum(
            [
                int(sample["intersections"]["ignore_sample"])
                for sample in x
                if u_labels.has_intersections(sample)
            ]
        )
        for x in labels
    ]

    log_zip_l_intersections = zip(log_names, number_of_l_intersections_for_each_log, strict=True)
    log_zip_t_intersections = zip(log_names, number_of_t_intersections_for_each_log, strict=True)
    log_zip_x_intersections = zip(log_names, number_of_x_intersections_for_each_log, strict=True)

    number_of_l_intersection_samples = sum(number_of_l_intersections_for_each_log)
    number_of_t_intersection_samples = sum(number_of_t_intersections_for_each_log)
    number_of_x_intersection_samples = sum(number_of_x_intersections_for_each_log)

    # ===== Calculate Distances ======
    if calculate_distances:
        print("Calulcating Distances for Balls...")
        distances_ball = _clean_up_list(
            [
                get_distance_from_label(
                    label,
                    u_dataset.CategoryNames.BALL.value,
                    0,
                )
                for label in labels_concat
            ]
        )
        print("Calulcating Distances for PenaltyMarks...")
        distances_penaltyMark = _clean_up_list(
            [
                get_distance_from_label(
                    label,
                    u_dataset.CategoryNames.PENALTYMARK.value,
                    0,
                )
                for label in labels_concat
            ]
        )
        print("Calulcating Distances for L-Intersections...")
        distances_l_intersections = _clean_up_list(
            [
                get_distance_from_label(
                    label,
                    u_dataset.CategoryNames.INTERSECTIONS.value,
                    0,
                    u_dataset.IntersectionType.L.name,
                )
                for label in labels_concat
            ]
        )
        print("Calulcating Distances for T-Intersections...")
        distances_t_intersections = _clean_up_list(
            [
                get_distance_from_label(
                    label,
                    u_dataset.CategoryNames.INTERSECTIONS.value,
                    0,
                    u_dataset.IntersectionType.T.name,
                )
                for label in labels_concat
            ]
        )
        print("Calulcating Distances for X-Intersections...")
        distances_x_intersections = _clean_up_list(
            [
                get_distance_from_label(
                    label,
                    u_dataset.CategoryNames.INTERSECTIONS.value,
                    0,
                    u_dataset.IntersectionType.X.name,
                )
                for label in labels_concat
            ]
        )

        print(
            "Mean ball distances: ", np.mean(distances_ball) if len(distances_ball) > 0 else "NaN"
        )
        print(
            "Variance ball distances: ",
            np.var(distances_ball) if len(distances_ball) > 0 else "NaN",
        )
        print(
            "Mean penaltyMark distances: ",
            np.mean(distances_penaltyMark) if len(distances_penaltyMark) > 0 else "NaN",
        )
        print(
            "Variance penaltyMark distances: ",
            np.var(distances_penaltyMark) if len(distances_penaltyMark) > 0 else "NaN",
        )
        print(
            "Mean L-Intersection distances: ",
            np.mean(distances_l_intersections) if len(distances_l_intersections) > 0 else "NaN",
        )
        print(
            "Variance L-Intersection distances: ",
            np.var(distances_l_intersections) if len(distances_l_intersections) > 0 else "NaN",
        )
        print(
            "Mean T-Intersection distances: ",
            np.mean(distances_t_intersections) if len(distances_t_intersections) > 0 else "NaN",
        )
        print(
            "Variance T-Intersection distances: ",
            np.var(distances_t_intersections) if len(distances_t_intersections) > 0 else "NaN",
        )
        print(
            "Mean X-Intersection distances: ",
            np.mean(distances_x_intersections) if len(distances_x_intersections) > 0 else "NaN",
        )
        print(
            "Variance X-Intersection distances: ",
            np.var(distances_x_intersections) if len(distances_x_intersections) > 0 else "NaN",
        )

    # ===== Cross-Entropy baselines ======
    ball_bce_baseline = get_bce_baseline(number_of_ball_samples, number_of_samples)
    penalty_mark_bce_baseline = get_bce_baseline(number_of_penalty_mark_samples, number_of_samples)

    print("Number of logs: ", len(labels))
    print("Number of samples: ", number_of_samples)
    print("Number of intersection samples: ", number_of_intersection_samples)
    print("Number of non empty samples: ", number_of_non_empty_samples)
    print("Number of ball samples:", number_of_ball_samples)
    print("Number of penaltyMark samples:", number_of_penalty_mark_samples)
    print("Number of L intersection samples:", number_of_l_intersection_samples)
    print("Number of T intersection samples:", number_of_t_intersection_samples)
    print("Number of X intersection samples:", number_of_x_intersection_samples)

    print(
        "Number of ignored intersections samples per log: ",
        number_of_ignored_intersection_samples_for_each_log,
    )
    print("L intersections per log: ", number_of_l_intersections_for_each_log)
    print("T intersections per log: ", number_of_t_intersections_for_each_log)
    print("X intersections per log: ", number_of_x_intersections_for_each_log)
    print("=========")
    print("% of ball samples: ", round((number_of_ball_samples / number_of_samples) * 100, 2))
    print(
        f"% of penaltyMark samples: {((number_of_penalty_mark_samples / number_of_samples) * 100):.2f}"
    )
    print(
        f"% of L intersection samples: {((number_of_l_intersection_samples / number_of_intersection_samples) * 100):.2f}"
    )
    print(
        f"% of T intersection samples: {((number_of_t_intersection_samples / number_of_intersection_samples) * 100):.2f}"
    )
    print(
        f"% of X intersection samples: {((number_of_x_intersection_samples / number_of_intersection_samples) * 100):.2f}"
    )
    print("=========")
    print(f"Ball BCE Baseline: {ball_bce_baseline:.5f}")
    print(f"PenaltyMark BCE Baseline: {penalty_mark_bce_baseline:.5f}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="This script shows statistics about the dataset.")
    parser.add_argument("data_path")
    parser.add_argument("--calculate_distances", default=False)
    args = parser.parse_args()

    main(args.data_path, args.calculate_distances)
