import itertools
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import tensorflow as tf
import yaml

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


def _clean_up_list(input: list):
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


def compute_distances(
    labels: dict, object_name: str, object_height: float, intersection_type: str = None
):
    distances = _clean_up_list(
        [
            get_distance_from_label(label, object_name, object_height, intersection_type)
            for label in labels
        ]
    )
    return distances


def count_intersections_per_log(labels: dict, intersection_type: str):
    return [
        sum(
            [
                len(sample["intersections"][intersection_type])
                for sample in log_labels
                if u_labels.has_intersections(sample)
            ]
        )
        for log_labels in labels
    ]


def main(
    data_path: str,
    calculate_distances: bool,
    no_filesave: bool,
    print_output: bool,
):
    label_dirs = [dir[0] for dir in os.walk(data_path)][1:]
    labels = [u_dataset_io.load_labels(dir) for dir in label_dirs]

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
    number_of_ignored_intersection_samples_per_log = [
        sum(
            [
                int(sample["intersections"]["ignore_sample"])
                for sample in x
                if u_labels.has_intersections(sample)
            ]
        )
        for x in labels
    ]

    number_of_ball_samples = len([_ for _ in labels_concat if u_labels.has_ball(_)])
    number_of_penalty_mark_samples = len([_ for _ in labels_concat if u_labels.has_penalty_mark(_)])

    number_of_l_intersections_per_log = count_intersections_per_log(
        labels, u_dataset.IntersectionType.L.name
    )
    number_of_t_intersections_per_log = count_intersections_per_log(
        labels, u_dataset.IntersectionType.T.name
    )
    number_of_x_intersections_per_log = count_intersections_per_log(
        labels, u_dataset.IntersectionType.X.name
    )

    number_of_l_intersection_samples = sum(number_of_l_intersections_per_log)
    number_of_t_intersection_samples = sum(number_of_t_intersections_per_log)
    number_of_x_intersection_samples = sum(number_of_x_intersections_per_log)
    number_of_ignored_intersection_samples = sum(number_of_ignored_intersection_samples_per_log)

    # ===== Cross-Entropy baselines ======
    ball_bce_baseline = float(get_bce_baseline(number_of_ball_samples, number_of_samples))
    penalty_mark_bce_baseline = float(
        get_bce_baseline(number_of_penalty_mark_samples, number_of_samples)
    )

    # ===== Calculate Distances ======
    moments = {}
    if calculate_distances:
        object_types = {
            "ball": {
                "name": u_dataset.CategoryNames.BALL.value,
                "height": 0,
            },
            "penalty_mark": {
                "name": u_dataset.CategoryNames.PENALTYMARK.value,
                "height": 0,
            },
            "l_intersection": {
                "name": u_dataset.CategoryNames.INTERSECTIONS.value,
                "height": 0,
                "intersection_type": u_dataset.IntersectionType.L.name,
            },
            "t_intersection": {
                "name": u_dataset.CategoryNames.INTERSECTIONS.value,
                "height": 0,
                "intersection_type": u_dataset.IntersectionType.T.name,
            },
            "x_intersection": {
                "name": u_dataset.CategoryNames.INTERSECTIONS.value,
                "height": 0,
                "intersection_type": u_dataset.IntersectionType.X.name,
            },
        }

        distances = {}
        for obj_key, obj_config in object_types.items():
            print(f"Calculating Distances for {obj_key}...")
            distances[obj_key] = compute_distances(
                labels_concat,
                obj_config["name"],
                obj_config["height"],
                obj_config.get("intersection_type"),
            )
            if not no_filesave:
                os.makedirs("data/distances/", exist_ok=True)
                np.save(f"data/distances/distances_{obj_key}.npy", np.array(distances[obj_key]))

        # Calculate mean and variance for each object type
        for obj_key in distances:
            mean_distance = float(np.mean(distances[obj_key]))
            var_distance = float(np.var(distances[obj_key]))
            moments[f"mean_{obj_key}_distance"] = mean_distance
            moments[f"var_{obj_key}_distance"] = var_distance
            print(f"Mean {obj_key} distance: ", mean_distance)
            print(f"Variance {obj_key} distances: ", var_distance)

    if not no_filesave:
        stats = {
            "number_of_logs": len(labels),
            "number_of_samples": number_of_samples,
            "number_of_intersection_samples": number_of_intersection_samples,
            "number_of_non_empty_samples": number_of_non_empty_samples,
            "number_of_ball_samples": number_of_ball_samples,
            "number_of_penalty_mark_samples": number_of_penalty_mark_samples,
            "number_of_ignored_intersection_samples": number_of_ignored_intersection_samples,
            "number_of_l_intersection_samples": number_of_l_intersection_samples,
            "number_of_t_intersection_samples": number_of_t_intersection_samples,
            "number_of_x_intersection_samples": number_of_x_intersection_samples,
            "number_of_ignored_intersection_samples_per_log": number_of_ignored_intersection_samples_per_log,
            "number_of_l_intersections_per_log": number_of_l_intersections_per_log,
            "number_of_t_intersections_per_log": number_of_t_intersections_per_log,
            "number_of_x_intersections_per_log": number_of_x_intersections_per_log,
            "distance_moments": moments,
            "percentages": {
                "percent_ball_samples": round(
                    (number_of_ball_samples / number_of_samples) * 100, 2
                ),
                "percent_penalty_mark_samples": round(
                    (number_of_penalty_mark_samples / number_of_samples) * 100, 2
                ),
                "percent_l_intersection_samples": round(
                    (number_of_l_intersection_samples / number_of_intersection_samples) * 100, 2
                ),
                "percent_t_intersection_samples": round(
                    (number_of_t_intersection_samples / number_of_intersection_samples) * 100, 2
                ),
                "percent_x_intersection_samples": round(
                    (number_of_x_intersection_samples / number_of_intersection_samples) * 100, 2
                ),
            },
            "baselines": {
                "ball_bce_baseline": round(ball_bce_baseline, 5),
                "penalty_mark_bce_baseline": round(penalty_mark_bce_baseline, 5),
            },
        }

        with open("data/statistics.yaml", "w") as yaml_file:
            yaml.dump(stats, yaml_file, default_flow_style=False, sort_keys=False)

    # ===== Printing ======
    if print_output:
        print("Number of logs: ", len(labels))
        print("Number of samples: ", number_of_samples)
        print("Number of intersection samples: ", number_of_intersection_samples)
        print("Number of non empty samples: ", number_of_non_empty_samples)
        print("Number of ball samples:", number_of_ball_samples)
        print("Number of penaltyMark samples:", number_of_penalty_mark_samples)
        print("Number of ignored intersection samples:", number_of_ignored_intersection_samples)
        print("Number of L intersection samples:", number_of_l_intersection_samples)
        print("Number of T intersection samples:", number_of_t_intersection_samples)
        print("Number of X intersection samples:", number_of_x_intersection_samples)
        print("=========")
        print(
            "Number of ignored intersections samples per log: ",
            number_of_ignored_intersection_samples_per_log,
        )
        print("L intersections per log: ", number_of_l_intersections_per_log)
        print("T intersections per log: ", number_of_t_intersections_per_log)
        print("X intersections per log: ", number_of_x_intersections_per_log)
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
        print("=========")
        for key, value in moments.items():
            print(f"{key.replace('_', ' ').capitalize()}: ", value)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="This script shows statistics about the dataset.")
    parser.add_argument("data_path")
    parser.add_argument("--calculate_distances", action="store_true", default=False)
    parser.add_argument("--no_filesave", action="store_true")
    parser.add_argument("--print_output", action="store_true", default=False)
    args = parser.parse_args()

    main(args.data_path, args.calculate_distances, args.no_filesave, args.print_output)
