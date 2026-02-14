import json
import os
import sys
from pathlib import Path

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import argparse

import numpy as np
import tensorflow as tf
import yaml

from util import camera as u_camera
from util import dataset as u_dataset
from util import dataset_io as u_dataset_io
from util import labels as u_labels
from util import metrics as u_metrics


def load_data(path_to_data: Path, model_timestamp: str):
    test_groundtruth_path = Path(path_to_data, "test_groundtruth.json")
    test_bhuman_path = Path(path_to_data, "test_b-human_predictions.json")

    test_model_intersections_path = Path(
        path_to_data, model_timestamp, "predictions", "intersections.json"
    )
    test_model_ball_path = Path(path_to_data, model_timestamp, "predictions", "ball.json")
    test_model_penaltymark_path = Path(
        path_to_data, model_timestamp, "predictions", "penaltyMark.json"
    )

    test_groundtruth = None
    test_bhuman = None
    test_intersections_model = None
    test_ball_model = None
    test_penaltymark_model = None

    with open(test_groundtruth_path) as f:
        test_groundtruth = json.load(f)
    with open(test_bhuman_path) as f:
        test_bhuman = json.load(f)
    with open(test_model_intersections_path) as f:
        test_intersections_model = json.load(f)
    with open(test_model_ball_path) as f:
        test_ball_model = json.load(f)
    with open(test_model_penaltymark_path) as f:
        test_penaltymark_model = json.load(f)

    assert len(test_groundtruth) == len(test_bhuman)
    assert len(test_bhuman) == len(test_intersections_model)
    assert len(test_bhuman) == len(test_ball_model)
    assert len(test_bhuman) == len(test_penaltymark_model)

    return {
        "test_groundtruth": test_groundtruth,
        "test_bhuman": test_bhuman,
        "test_intersections_model": test_intersections_model,
        "test_ball_model": test_ball_model,
        "test_penaltymark_model": test_penaltymark_model,
    }


def load_config(model_timestamp: str) -> dict:
    config_path = f"logs/fit/{model_timestamp}/config.yaml"

    with open(config_path) as f:
        config = yaml.safe_load(f)

    return config


def extract_coordinates(intersections: list) -> list:
    # Convert list of dicts to list of lists: [[x1, y1], [x2, y2], ...]
    return [[coord["x"], coord["y"]] for coord in intersections]


def filter_coords_by_distance(
    coords_tensor: list | tf.Tensor, camera: tuple, intrinsics: tuple, max_distance: float
) -> tf.Tensor:
    if tf.shape(coords_tensor)[0] > 0:
        if len(tf.shape(coords_tensor)) == 3:
            gt_coords_tensor = coords_tensor[:, 1]  # (N, 2)
            pred_coords_tensor = coords_tensor[:, 0]  # (N, 2)
        else:
            gt_coords_tensor = coords_tensor  # (N, 2)
            pred_coords_tensor = None

        img_to_wlrd_gt = u_camera.image_to_world(camera, intrinsics, gt_coords_tensor)  # (N, 3)
        valid_distances = ~tf.reduce_all(img_to_wlrd_gt == -1.0, axis=-1)  # (N, )

        distances_gt = tf.linalg.norm(img_to_wlrd_gt, axis=-1, keepdims=True)  # (N, )

        mask = tf.squeeze(distances_gt <= max_distance, axis=-1) & valid_distances  # (N, )

        filtered_gt_tensor = tf.boolean_mask(gt_coords_tensor, mask)  # (M, 2)

        if pred_coords_tensor is not None:
            img_to_wlrd_pred = u_camera.image_to_world(camera, intrinsics, pred_coords_tensor)
            distances_pred = tf.linalg.norm(img_to_wlrd_pred, axis=-1, keepdims=True)  # (N, )

            distances = tf.boolean_mask(
                tf.squeeze(tf.stack([distances_pred, distances_gt], axis=-1), axis=1), mask
            )  # (M, 2)
            filtered_pred_tensor = tf.boolean_mask(pred_coords_tensor, mask)  # (M, 2)

            assert tf.reduce_all(tf.shape(distances_gt) == tf.shape(distances_pred))
            assert tf.reduce_all(tf.shape(distances)[0] == tf.shape(filtered_pred_tensor)[0])
            assert tf.reduce_all(tf.shape(filtered_gt_tensor) == tf.shape(filtered_pred_tensor))

            return (
                tf.stack([filtered_pred_tensor, filtered_gt_tensor], axis=1),
                distances,
            )  # (M, 2, 2), (M, 2)
        else:
            return filtered_gt_tensor, distances_gt  # (M, 2), (M, )
    else:
        return (tf.zeros((0, 2, 2), tf.float32), tf.zeros((0, 2), tf.float32))  # (0, 2, 2), (0, 2)


def process_object_metrics(
    preds: dict[tf.Tensor],
    gt_frame: dict[tf.Tensor],
    object_name: str,
    threshold: float,
    intersection_type: str | None = None,
    ball_status_only_seen: bool | None = None,
) -> dict:
    # Extract coordinates for the given object type
    if object_name == u_dataset.CategoryNames.INTERSECTIONS.value:
        pred_coords = extract_coordinates(preds[object_name][intersection_type])
        gt_coords = extract_coordinates(gt_frame[object_name][intersection_type])
    elif object_name in [
        u_dataset.CategoryNames.BALL.value,
        u_dataset.CategoryNames.PENALTYMARK.value,
    ]:
        if object_name not in preds or (
            ball_status_only_seen
            and object_name == u_dataset.CategoryNames.BALL.value
            and preds[object_name]["status"] == 2
        ):  # status == 2 is "guessed"
            pred_coords = []
        else:
            if isinstance(preds[object_name], list):
                pred_coords = extract_coordinates(preds[object_name])
            else:
                pred_coords = [[preds[object_name]["x"], preds[object_name]["y"]]]
        if object_name not in gt_frame:
            gt_coords = []
        else:
            if isinstance(gt_frame[object_name], list):
                gt_coords = extract_coordinates(gt_frame[object_name])
            else:
                gt_coords = [[gt_frame[object_name]["x"], gt_frame[object_name]["y"]]]
    else:
        raise ValueError("Invalid object_name.")

    # Convert coordinates to tensors
    pred_tensor = tf.constant(pred_coords, dtype=tf.float32)  # (N, 2)
    gt_tensor = tf.constant(gt_coords, dtype=tf.float32)  # (N, 2)

    # Match keypoints and return metrics
    return u_metrics.match_keypoints_image(pred_tensor, gt_tensor, threshold)


def compare_predictions(
    groundtruth: dict,
    model_preds: dict,
    bhuman_preds: dict,
    object_name: str,
    max_distance: float,
    threshold: float,
    save_path_for_matches: str,
    ball_status_only_seen: bool | None = None,
) -> dict:
    # TODO: lose the for-loop and only do tensor operations to save a lot of time.
    if object_name == u_dataset.CategoryNames.INTERSECTIONS.value:
        metrics = {
            "model_true_positives": {k.name: 0 for k in u_labels.IntersectionType},
            "model_false_negatives": {k.name: 0 for k in u_labels.IntersectionType},
            "model_false_positives": {k.name: 0 for k in u_labels.IntersectionType},
            "bhuman_true_positives": {k.name: 0 for k in u_labels.IntersectionType},
            "bhuman_false_negatives": {k.name: 0 for k in u_labels.IntersectionType},
            "bhuman_false_positives": {k.name: 0 for k in u_labels.IntersectionType},
        }
        tp_matches = {
            "model": {k.name: {"matches": [], "distances": []} for k in u_labels.IntersectionType},
            "bhuman": {k.name: {"matches": [], "distances": []} for k in u_labels.IntersectionType},
        }
        intersection_types = [t.name for t in list(u_labels.IntersectionType)]

    else:
        metrics = {
            "model_true_positives": 0,
            "model_false_negatives": 0,
            "model_false_positives": 0,
            "bhuman_true_positives": 0,
            "bhuman_false_negatives": 0,
            "bhuman_false_positives": 0,
        }
        tp_matches = {
            "model": {"matches": [], "distances": []},
            "bhuman": {"matches": [], "distances": []},
        }
        intersection_types = [None]

    # Iterate over each frame
    for idx, gt_frame in enumerate(groundtruth):
        frame_time = gt_frame["frame_time"]

        assert frame_time == model_preds[idx]["frame_time"]
        assert frame_time == bhuman_preds[idx]["frame_time"]

        camera = u_dataset_io.camera_from_label(gt_frame)
        intrinsics = u_dataset_io.intrinsics_from_label(gt_frame)

        if (
            object_name == u_dataset.CategoryNames.INTERSECTIONS.value
            and gt_frame[object_name]["ignore_sample"]
        ):
            continue

        for intersection_type in intersection_types:
            model_matches = process_object_metrics(
                model_preds[idx],
                gt_frame,
                object_name,
                threshold,
                intersection_type,
            )
            bhuman_matches = process_object_metrics(
                bhuman_preds[idx],
                gt_frame,
                object_name,
                threshold,
                intersection_type,
                ball_status_only_seen,
            )

            for prefix, matches in [("model", model_matches), ("bhuman", bhuman_matches)]:
                # Filter and get lengths for false positives, false negatives, and matched ground truth
                fp_len = tf.shape(
                    filter_coords_by_distance(
                        matches["fp_tensor"], camera, intrinsics, max_distance
                    )[0]
                )[0]
                fn_len = tf.shape(
                    filter_coords_by_distance(
                        matches["fn_tensor"], camera, intrinsics, max_distance
                    )[0]
                )[0]

                filtered_matches, distances = filter_coords_by_distance(
                    matches["matches"],
                    camera,
                    intrinsics,
                    max_distance,
                )
                matches_len = tf.shape(filtered_matches)[0]

                if object_name == u_dataset.CategoryNames.INTERSECTIONS.value:
                    tp_matches[prefix][intersection_type]["matches"].append(filtered_matches)
                    tp_matches[prefix][intersection_type]["distances"].append(distances)

                    # Update metrics
                    metrics[f"{prefix}_true_positives"][intersection_type] += int(matches_len)
                    metrics[f"{prefix}_false_negatives"][intersection_type] += int(fn_len)
                    metrics[f"{prefix}_false_positives"][intersection_type] += int(fp_len)
                else:
                    tp_matches[prefix]["matches"].append(filtered_matches)
                    tp_matches[prefix]["distances"].append(distances)

                    # Update metrics
                    metrics[f"{prefix}_true_positives"] += int(matches_len)
                    metrics[f"{prefix}_false_negatives"] += int(fn_len)
                    metrics[f"{prefix}_false_positives"] += int(fp_len)

    status_str = ""
    if ball_status_only_seen is not None:
        status_str = "_1" if ball_status_only_seen else "_2"

    for key, value in tp_matches.items():
        for intersection_type in intersection_types:
            object_list = (
                value if len(value.keys()) != len(intersection_types) else value[intersection_type]
            )

            model_tp_matches_concat = tf.concat(object_list["matches"], axis=0)  # (B, 2, 2)
            model_tp_distances_concat = tf.concat(object_list["distances"], axis=0)  # (B, 2)
            save_path = Path(save_path_for_matches, f"{object_name}{status_str}", intersection_type if intersection_type is not None else "")
            os.makedirs(save_path, exist_ok=True)

            # Save matches in .npy file
            np.save(
                save_path
                / f"{key}_matches",
                model_tp_matches_concat.numpy(),
            )
            # Save distances in .npy file
            np.save(
                save_path
                / f"{key}_distances",
                model_tp_distances_concat.numpy(),
            )

    return metrics


def print_results(metrics: dict, object_name: str, status: str = "") -> None:
    print(f"==== {object_name.capitalize()}{f', status: {status}' if len(status) > 0 else ''} ====")
    print("=== Model ===")
    print("TP: ", metrics["model_true_positives"])
    print("FP: ", metrics["model_false_positives"])
    print("FN: ", metrics["model_false_negatives"])

    print("=== B-Human ===")
    print("TP : ", metrics["bhuman_true_positives"])
    print("FP: ", metrics["bhuman_false_positives"])
    print("FN: ", metrics["bhuman_false_negatives"])


def main(args) -> None:

    save_path_for_matches = Path(args.directory, args.model_timestamp, "matches")
    os.makedirs(save_path_for_matches, exist_ok=True)

    config = load_config(args.model_timestamp)
    data = load_data(Path(args.directory), args.model_timestamp)

    print("Calculating Comparisons for Balls...")
    metrics_ball_seen = compare_predictions(
        data["test_groundtruth"],
        data["test_ball_model"],
        data["test_bhuman"],
        u_dataset.CategoryNames.BALL.value,
        max_distance=config["categories"][u_dataset.CategoryNames.BALL.value]["max_distance"],
        threshold=args.distance_threshold,
        save_path_for_matches=save_path_for_matches,
        ball_status_only_seen=True,
    )
    print_results(metrics_ball_seen, u_dataset.CategoryNames.BALL.value, "seen")

    print("Calculating Comparisons for Balls...")
    metrics_ball_seen_guessed = compare_predictions(
        data["test_groundtruth"],
        data["test_ball_model"],
        data["test_bhuman"],
        u_dataset.CategoryNames.BALL.value,
        max_distance=config["categories"][u_dataset.CategoryNames.BALL.value]["max_distance"],
        threshold=args.distance_threshold,
        save_path_for_matches=save_path_for_matches,
        ball_status_only_seen=False,
    )
    print_results(metrics_ball_seen_guessed, u_dataset.CategoryNames.BALL.value, "seen+guessed")

    print("Calculating Comparisons for PenaltyMarks...")
    metrics_penaltymark = compare_predictions(
        data["test_groundtruth"],
        data["test_penaltymark_model"],
        data["test_bhuman"],
        u_dataset.CategoryNames.PENALTYMARK.value,
        max_distance=config["categories"][u_dataset.CategoryNames.PENALTYMARK.value][
            "max_distance"
        ],
        threshold=args.distance_threshold,
        save_path_for_matches=save_path_for_matches,
    )
    print_results(metrics_penaltymark, u_dataset.CategoryNames.PENALTYMARK.value)

    print("Calculating Comparisons for Intersections...")
    metrics_intersections = compare_predictions(
        data["test_groundtruth"],
        data["test_intersections_model"],
        data["test_bhuman"],
        u_dataset.CategoryNames.INTERSECTIONS.value,
        max_distance=config["categories"][u_dataset.CategoryNames.INTERSECTIONS.value][
            "max_distance"
        ],
        threshold=args.distance_threshold,
        save_path_for_matches=save_path_for_matches,
    )
    print_results(metrics_intersections, u_dataset.CategoryNames.INTERSECTIONS.value)

    # Save all the comparison into a single yaml file.
    with open(Path(args.directory, args.model_timestamp, "comparisons.yaml"), "w") as file:
        yaml.dump(
            {
                "balls_seen": metrics_ball_seen,
                "balls_seen_guessed": metrics_ball_seen_guessed,
                "penaltyMark": metrics_penaltymark,
                "intersections": metrics_intersections,
            },
            file,
            indent=4,
            default_flow_style=False,
            sort_keys=False,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="This script compares the prediction of the model with the given the given timestamp and the current B-Human detectors."
    )
    parser.add_argument("--model_timestamp")
    parser.add_argument("--directory", default="data/evaluation")
    parser.add_argument("--distance_threshold", default=30.0)
    args = parser.parse_args()

    main(args)
