import argparse
import csv
import os
import sys

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import glob
from pathlib import Path

import numpy as np
import sklearn
import tensorflow as tf
import yaml

from training.models import FullModel
from util import dataset as u_dataset
from util import dataset_io as u_dataset_io
from util import metrics as u_metrics


class Evaluator:
    def __init__(self, args):
        self.eval_cpn = args.cpn
        self.eval_classifier = args.classifier

        self.distance_filter = 9 if args.distance is None else args.distance
        self.n_candidates = (4, 4, 10) if args.n_candidates is None else args.n_candidates

        print("Distance Filter:", self.distance_filter)
        print("K_c:", self.n_candidates)
        
        self.config_dir = Path(
            glob.glob(
                os.path.join(args.log_dir, "**", args.model_timestamp, "config.yaml"),
                recursive=True,
            )[0]
        )

        print("Loading Config...")
        self.config = self.load_config()

        print("Loading Dataset...")
        self.test_ds = self.load_dataset()

        self.resolution = f"{self.config['model']['encoder']['input_dims'][0]}x{self.config['model']['encoder']['input_dims'][1]}"
        self.cpn_architecture = self.config["model"]["encoder"]["architecture"]
        self.classifier_architecture = self.config["model"]["classifier"]["architecture"]

        self.mode = args.log_dir.split("-")[-1].split("/")[0]
        self.run = args.log_dir.split("/")[-1]

    def main(self):
        if self.eval_cpn:
            print("Evaluating CPN...")
            save_path = f"data/evaluation/cpn-{self.mode}/{self.run}.csv"

            architecture = glob.glob(
                os.path.join(
                    args.model_dir, self.resolution, "**", f"{args.model_timestamp}.keras"
                ),
                recursive=True,
            )[0].split(os.sep)[-3]

            path_to_models = Path(args.model_dir, self.resolution, architecture)

            print("Loading Model...")
            self.model = self.load_model(path_to_models, args.model_timestamp)

            cpn_metrics = self.evaluate_cpn()
            self.create_metrics_csv(
                save_path,
                args.model_timestamp,
                cpn_metrics.items(),
            )
        if self.eval_classifier:
            print("Evaluating Classifier...")

            # Folder name that contains information about the distance_filter and n_candidates. Example: "d_9-K_4-4-10"
            self.specification_string = f"d_{self.distance_filter}-K_{self.n_candidates[0]}-{self.n_candidates[1]}-{self.n_candidates[2]}"

            save_dir = (
                f"data/evaluation/classifier-{self.mode}/{self.run}.csv"
                if args.save_dir is None
                else f"{args.save_dir}/{self.specification_string}/{self.run}.csv"
            )

            end_to_end = True

            if self.run == "final":
                path_to_models = Path(
                    args.model_dir,
                    args.model_timestamp,
                )
            else:
                architecture_version = self.config["model"]["classifier"]["architecture"].split(
                    "_"
                )[-1]
                use_architecture_version = "v" in args.model_dir.split("/")[-1]
                n_context = self.config["model"]["encoder"]["n_context"]

                path_to_models = Path(
                    args.model_dir,
                    architecture_version
                    if n_context == 0 or use_architecture_version
                    else str(n_context),
                    args.model_timestamp,
                )

            print("Loading Model...")
            self.model = self.load_model(path_to_models, args.model_timestamp)

            predicted_metrics = self.evaluate_classifier(args, end_to_end)
            inference_metrics = self.evaluate_cpn()

            metrics_to_save = {}
            for category in u_dataset.CategoryNames:
                if category.value in predicted_metrics:
                    # metrics_to_save[f"{category.value}_precision"] = classifier_metrics[category.value]["precision"]
                    # metrics_to_save[f"{category.value}_recall"] = classifier_metrics[category.value]["recall"]
                    metrics_to_save[f"{category.value}_ap_pooled"] = predicted_metrics[
                        category.value
                    ]["ap_pooled"]
                    # metrics_to_save[f"{category.value}_ap"] = predicted_metrics[category.value]["ap_per_class"]
                    metrics_to_save[f"{category.value}_mAP"] = predicted_metrics[category.value][
                        "mAP"
                    ]

            metrics_to_save.update(inference_metrics)

            self.create_metrics_csv(
                save_dir,
                args.model_timestamp,
                metrics_to_save.items(),
            )

    def load_config(self):
        with open(Path(self.config_dir)) as f:
            config = yaml.safe_load(f)
        return config

    def load_dataset(self):
        input_dims = self.config["model"]["encoder"]["input_dims"]

        dataset_utils = u_dataset.DatasetUtils(
            u_dataset.DatasetConfig(
                input_dims=input_dims,
                cell_dims=self.config["model"]["encoder"]["cell_dims"],
            )
        )

        path_to_test_data = glob.glob(
            Path(
                "data/tfrecords/", f"{input_dims[1]}x{input_dims[0]}", "val_ds*.tfrecords"
            ).as_posix()
        )

        test_ds = u_dataset_io.get_dataset(path_to_test_data, dataset_utils)
        batch_size = 32
        test_ds = test_ds.batch(batch_size, drop_remainder=False)

        return test_ds

    def load_model(self, path_to_models: str, model_name: str):
        encoder_architecture = self.config["model"]["encoder"]["architecture"]
        classifier_architecture = self.config["model"]["classifier"]["architecture"]

        channels_in = self.config["model"]["encoder"].get("channels_in", 4)

        self.config["categories"]["ball"]["n_candidates"] = self.n_candidates[0]
        self.config["categories"]["penaltyMark"]["n_candidates"] = self.n_candidates[1]
        self.config["categories"]["intersections"]["n_candidates"] = self.n_candidates[2]

        self.config["categories"]["ball"]["max_distance"] = self.distance_filter
        self.config["categories"]["penaltyMark"]["max_distance"] = self.distance_filter
        self.config["categories"]["intersections"]["max_distance"] = self.distance_filter

        model = FullModel.load(
            encoder_architecture,
            classifier_architecture,
            filepath=path_to_models,
            filename=model_name,
            input_dims=self.config["model"]["encoder"]["input_dims"],
            encoder_channels=channels_in,
            cell_dims=self.config["model"]["encoder"]["cell_dims"],
            n_context=self.config["model"]["encoder"]["n_context"],
            train_encoder=True,
            train_classifier=self.config["model"]["classifier"]["train_classifier"],
            classifier_offsets=self.config["model"]["classifier"]["with_offsets"],
            encoder_only=False,
            verbose=True,
            n_meta=self.config["model"]["classifier"]["n_meta"],
            encoder_use_batch_norm=self.config["model"]["encoder"]["use_batch_norm"],
            classifier_use_batch_norm=self.config["model"]["classifier"]["use_batch_norm"],
            categories_config=self.config["categories"],
        )

        model.compile(optimizer=tf.keras.optimizers.Adam(), jit_compile=False)
        model.run_eagerly = True

        return model

    def append_to_csv(self, file_path, data):
        """
        Append data to a CSV file. If the file does not exist, it will be created.

        Args:
            file_path (str): Path to the CSV file.
            data (dict): Dictionary containing the data to be written to the CSV file.
        """
        file_exists = os.path.isfile(file_path)

        if not file_exists:
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            with open(file_path, mode="x", newline="") as file:
                writer = csv.DictWriter(file, fieldnames=data.keys())
                writer.writeheader()
                writer.writerow(data)

        else:
            existing_data = []
            with open(file_path, newline="") as file:
                reader = csv.DictReader(file)
                for row in reader:
                    existing_data.append(row)
            # Check if the timestamp already exists
            timestamp_exists = any(
                row["model_timestamp"] == data["model_timestamp"] for row in existing_data
            )

            if timestamp_exists:
                # Rewrite the file without the existing row
                with open(file_path, mode="w", newline="") as file:
                    writer = csv.DictWriter(file, fieldnames=data.keys())
                    writer.writeheader()
                    for row in existing_data:
                        if row["model_timestamp"] != data["model_timestamp"]:
                            writer.writerow(row)

            # Write the new row
            with open(file_path, mode="a", newline="") as file:
                writer = csv.DictWriter(file, fieldnames=data.keys())
                writer.writerow(data)

    def create_metrics_csv(self, file_path, model_timestamp, metrics_list):
        """
        Create or append to a CSV file with the specified metrics data.

        Args:
            file_path (str): Path to the CSV file.
            resolution (str): Resolution of the model.
            architecture (str): Architecture of the model.
            model_timestamp (str): Timestamp of the model.
            config (dict): Configuration dictionary containing the architecture.
            metrics_list (list): List of metrics to be included in the CSV file.
        """
        data = {
            "resolution": self.resolution,
            "input_channels": self.config["model"]["encoder"].get("channels_in", 4),
            "model_timestamp": model_timestamp,
            "encoder_architecture": self.config["model"]["encoder"]["architecture"],
            "classifier_architecture": self.config["model"]["classifier"]["architecture"],
            "n_context": self.config["model"]["encoder"]["n_context"],
            "n_dist": self.config["model"]["classifier"]["n_meta"],
            "distance_filter": self.distance_filter,
            "n_candidates_ball": self.n_candidates[0],
            "n_candidates_penaltyMark": self.n_candidates[1],
            "n_candidates_intersections": self.n_candidates[2],
        }

        # Add each metric in metrics_list to the data dictionary
        for key, value in metrics_list:
            data[key] = float(value)  # Assuming the metric value is the same as the metric name

        self.append_to_csv(file_path, data)

    def evaluate_cpn(self):
        metrics_list = self.model.evaluate(x=self.test_ds, return_dict=True)
        return metrics_list

    def evaluate_classifier(self, args, end_to_end):
        def _get_metrics(
            predictions,
            groundtruth,
            config,
            thresholds: dict,
            threshold_mode: str,
            encoder_threshold: float = 0.01,
            nms_iou_threshold: float = None,
            end_to_end: bool = True,
        ) -> dict:
            metrics = {}

            dataset_utils = u_dataset.DatasetUtils(
                u_dataset.DatasetConfig(
                    input_dims=config["model"]["encoder"]["input_dims"],
                    cell_dims=config["model"]["encoder"]["cell_dims"],
                )
            )

            for object in u_dataset.CategoryNames:
                if object.value not in predictions["results"]:
                    print(f"No {object.value} in results.")
                    continue

                if object.name == u_dataset.CategoryNames.INTERSECTIONS.name:
                    results = [
                        u_metrics.calculate_metrics(
                            dataset_utils,
                            predictions["results"][object.value],
                            groundtruth[object.value],
                            config["categories"][object.value]["n_classes"],
                            cla,
                            encoder_threshold,
                            threshold_mode,
                            end_to_end,
                            groundtruth["camera"],
                            groundtruth["intrinsics"],
                            config["categories"][object.value]["max_distance"],
                            iou_threshold=nms_iou_threshold,
                        )
                        for cla in thresholds[object.value]
                    ]
                else:
                    results = [
                        u_metrics.calculate_metrics(
                            dataset_utils,
                            predictions["results"][object.value],
                            groundtruth[object.value],
                            config["categories"][object.value]["n_classes"],
                            cla,
                            encoder_threshold,
                            threshold_mode,
                            end_to_end,
                            groundtruth["camera"],
                            groundtruth["intrinsics"],
                            config["categories"][object.value]["max_distance"],
                            config["categories"][object.value]["padding"],
                        )
                        for cla in thresholds[object.value]
                    ]

                if len(thresholds[object.value]) == 1:
                    metrics[object.value] = results[0]
                else:
                    metrics[object.value] = results

            return metrics

        def _calculate_ap_metrics(metrics_threshold_range_additive):
            metrics = {object.value: {} for object in u_dataset.CategoryNames}

            for object in u_dataset.CategoryNames:
                if object.value not in metrics_threshold_range_additive:
                    continue

                results = metrics_threshold_range_additive[object.value]

                # Pooled AP (binary)
                precision_pooled = np.array([x["precision_pooled"] for x in results])
                recall_pooled = np.array([x["recall_pooled"] for x in results])

                # Sort
                sorted_idx = np.argsort(recall_pooled)
                recall_pooled_sorted = recall_pooled[sorted_idx]
                precision_pooled_sorted = precision_pooled[sorted_idx]

                # Anchor point at Recall 0.0
                recalls_anchored = np.concatenate([[0.0], recall_pooled_sorted])
                precisions_anchored = np.concatenate(
                    [[precision_pooled_sorted[0]], precision_pooled_sorted]
                )

                # Interpolate precisions
                precisions_interp = np.maximum.accumulate(precisions_anchored[::-1])[::-1]

                # Remove duplicate recalls to avoid sklearn error
                unique_mask = np.concatenate([[True], np.diff(recalls_anchored) > 0])
                ap_pooled = sklearn.metrics.auc(
                    recalls_anchored[unique_mask], precisions_interp[unique_mask]
                )

                precisions_path = Path(
                    self.config_dir.parent.as_posix(),
                    "evaluation",
                    self.specification_string,
                    "precisions",
                ).as_posix()
                recalls_path = Path(
                    self.config_dir.parent.as_posix(),
                    "evaluation",
                    self.specification_string,
                    "recalls",
                ).as_posix()
                os.makedirs(precisions_path, exist_ok=True)
                os.makedirs(recalls_path, exist_ok=True)

                if object != u_dataset.CategoryNames.INTERSECTIONS:
                    np.save(
                        precisions_path + f"/{object.value}",
                        precisions_interp[unique_mask],
                    )
                    np.save(
                        recalls_path + f"/{object.value}",
                        recalls_anchored[unique_mask],
                    )

                if object == u_dataset.CategoryNames.INTERSECTIONS:
                    # Per-class AP (for mAP, skip background class 0)
                    num_classes = len(results[0]["precisions"])
                    per_class_aps = []
                    for class_idx in range(1, num_classes):
                        precisions = np.array([x["precisions"][class_idx] for x in results])
                        recalls = np.array([x["recalls"][class_idx] for x in results])

                        # Sort
                        sorted_idx = np.argsort(recalls)
                        recall_sorted = recalls[sorted_idx]
                        precision_sorted = precisions[sorted_idx]

                        # Anchor point at Recall 0.0
                        recalls_anchored = np.concatenate([[0.0], recall_sorted])
                        precisions_anchored = np.concatenate(
                            [[precision_sorted[0]], precision_sorted]
                        )

                        # Interpolate precisions
                        precisions_interp = np.maximum.accumulate(precisions_anchored[::-1])[::-1]

                        unique_mask = np.concatenate([[True], np.diff(recalls_anchored) > 0])
                        ap = sklearn.metrics.auc(
                            recalls_anchored[unique_mask], precisions_interp[unique_mask]
                        )
                        per_class_aps.append(ap)

                        np.save(
                            precisions_path
                            + f"/{object.value}_{list(u_dataset.IntersectionType)[class_idx].name}",
                            precisions_interp[unique_mask],
                        )
                        np.save(
                            recalls_path
                            + f"/{object.value}_{list(u_dataset.IntersectionType)[class_idx].name}",
                            recalls_anchored[unique_mask],
                        )

                else:
                    per_class_aps = ap_pooled

                metrics[object.value] = {
                    "precision_pooled": precision_pooled_sorted,
                    "recall_pooled": recall_pooled_sorted,
                    "ap_pooled": ap_pooled,
                    "per_class_aps": per_class_aps,
                    "mAP": np.mean(per_class_aps),
                }

            return metrics

        predictions_list = []
        for batch in self.test_ds:
            predictions_list.append(self.model.predict(batch))

        # Then concat manually
        predictions_concat = {"results": {}}
        for object in u_dataset.CategoryNames:
            key = object.value
            if key not in predictions_list[0]["results"]:
                continue
            predictions_concat["results"][key] = {
                field: tf.concat([p["results"][key][field] for p in predictions_list], axis=0)
                for field in predictions_list[0]["results"][key]
            }

        groundtruth_dataset = []
        for x in self.test_ds:
            groundtruth_dataset.append(x)

        groundtruth_concat = {}
        for key in groundtruth_dataset[0]:
            if isinstance(groundtruth_dataset[0][key], tf.Tensor):
                # Top-level tensors like intrinsics, camera
                groundtruth_concat[key] = tf.concat([g[key] for g in groundtruth_dataset], axis=0)
            elif isinstance(groundtruth_dataset[0][key], dict):
                # Nested dicts like groundtruth["ball"], groundtruth["penaltymark"]
                groundtruth_concat[key] = {}
                for subkey in groundtruth_dataset[0][key]:
                    if isinstance(groundtruth_dataset[0][key][subkey], tf.Tensor):
                        groundtruth_concat[key][subkey] = tf.concat(
                            [g[key][subkey] for g in groundtruth_dataset], axis=0
                        )
                    else:
                        groundtruth_concat[key][subkey] = groundtruth_dataset[0][key][subkey]
            else:
                groundtruth_concat[key] = groundtruth_dataset[0][key]

        nms_iou_threshold = 0.35
        encoder_threshold = 0.01
        threshold_range_additive = np.linspace(0, 1, num=100)

        classifier_threshold_ranges_additive = {
            u_dataset.CategoryNames.BALL.value: threshold_range_additive,
            u_dataset.CategoryNames.PENALTYMARK.value: threshold_range_additive,
            u_dataset.CategoryNames.INTERSECTIONS.value: threshold_range_additive,
        }
        metrics_threshold_range_additive = _get_metrics(
            predictions_concat,
            groundtruth_concat,
            self.config,
            classifier_threshold_ranges_additive,
            "additive",
            encoder_threshold,
            nms_iou_threshold,
            end_to_end,
        )

        return _calculate_ap_metrics(metrics_threshold_range_additive)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="This script compares the prediction of the model with the given the given timestamp and the current B-Human detectors."
    )
    parser.add_argument("--model_timestamp", type=str)
    parser.add_argument("--log_dir", type=str)
    parser.add_argument("--save_dir", type=str, required=False, default=None)
    parser.add_argument("--distance", type=int, required=False, default=None)
    parser.add_argument(
        "--n_candidates", type=lambda x: tuple(map(int, x.split(","))), required=False, default=None
    )
    parser.add_argument("--model_dir", type=str)
    parser.add_argument("--cpn", type=bool, default=False, required=False)
    parser.add_argument("--classifier", type=bool, default=False, required=False)
    args = parser.parse_args()

    evaluator = Evaluator(args)
    evaluator.main()
