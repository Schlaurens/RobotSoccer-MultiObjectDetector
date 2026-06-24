import argparse
import csv
import json
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

        self.beta = args.beta

        self.nms_iou_threshold = args.nms_iou
        self.encoder_threshold = 0.01

        self.end_to_end = True

        self.threshold_range = np.linspace(0, 1, num=100, dtype=np.float32)

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

        self.cell_dims = self.config["model"]["encoder"]["cell_dims"]
        self.input_dims = self.config["model"]["encoder"]["input_dims"]
        self.dataset_utils = u_dataset.DatasetUtils(
            u_dataset.DatasetConfig(self.input_dims, cell_dims=self.cell_dims)
        )

        print("Loading Dataset...")
        self.val_ds = self.load_dataset("val")
        self.test_ds = self.load_dataset("test")

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

            predicted_metrics = self.evaluate_classifier()

            self.optimal_metrics = self.calculate_optimal_metrics(predicted_metrics)
            print(self.optimal_metrics)
            # inference_metrics = self.evaluate_cpn()

            metrics_to_save = {}
            for category in u_dataset.CategoryNames:
                if category.value in predicted_metrics:
                    # metrics_to_save[f"{category.value}_precision"] = classifier_metrics[category.value]["precision"]
                    # metrics_to_save[f"{category.value}_recall"] = classifier_metrics[category.value]["recall"]
                    metrics_to_save[f"{category.value}_ap"] = predicted_metrics[category.value][
                        "ap"
                    ]
                    # metrics_to_save[f"{category.value}_ap"] = predicted_metrics[category.value]["ap_per_class"]
                    metrics_to_save[f"{category.value}_mAP"] = predicted_metrics[category.value][
                        "mAP"
                    ]

            # metrics_to_save.update(inference_metrics)

            self.create_metrics_csv(
                save_dir,
                args.model_timestamp,
                metrics_to_save.items(),
            )

    def load_config(self):
        with open(Path(self.config_dir)) as f:
            config = yaml.safe_load(f)
        return config

    def load_dataset(self, mode: str = "val"):
        path_to_data = glob.glob(
            Path(
                "data/tfrecords/",
                f"{self.input_dims[1]}x{self.input_dims[0]}",
                f"{mode}_ds*.tfrecords",
            ).as_posix()
        )

        ds = u_dataset_io.get_dataset(path_to_data, self.dataset_utils)
        batch_size = 32
        ds = ds.batch(batch_size, drop_remainder=False)

        return ds

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

        if channels_in != 1:
            input_dims = self.config["model"]["encoder"]["input_dims"] // np.array((1, 2))
        else:
            input_dims = self.config["model"]["encoder"]["input_dims"]

        model = FullModel.load(
            encoder_architecture,
            classifier_architecture,
            filepath=path_to_models,
            filename=model_name,
            input_dims=input_dims,
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
        metrics_list = self.model.evaluate(x=self.val_ds, return_dict=True)
        return metrics_list

    def get_classifier_metrics(
        self,
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

    def evaluate_classifier(self):
        def _calculate_ap_metrics(metrics_threshold_range: dict) -> dict:
            metrics = {category.value: {} for category in u_dataset.CategoryNames}
            eval_path = Path(self.config_dir.parent, "evaluation", self.specification_string)

            for category in u_dataset.CategoryNames:
                if category.value not in metrics_threshold_range:
                    continue

                results = metrics_threshold_range[category.value]
                is_intersection = category == u_dataset.CategoryNames.INTERSECTIONS
                intersection_types = list(u_dataset.IntersectionType)

                class_indices = (
                    range(1, len(results[0]["precisions"])) if is_intersection else [None]
                )
                per_class_aps = []
                per_class_precisions = []
                per_class_recalls = []

                for class_idx in class_indices:
                    if class_idx is not None:
                        precisions = np.array([x["precisions"][class_idx] for x in results])
                        recalls = np.array([x["recalls"][class_idx] for x in results])
                        name = f"{category.value}_{intersection_types[class_idx].name}"
                    else:
                        precisions = np.array([x["precisions"] for x in results])
                        recalls = np.array([x["recalls"] for x in results])
                        name = category.value

                    processed_pr = u_metrics.process_precision_recall(precisions, recalls)
                    ap = sklearn.metrics.auc(processed_pr["recalls"], processed_pr["precisions"])
                    per_class_aps.append(ap)
                    per_class_precisions.append(precisions)
                    per_class_recalls.append(recalls)

                    for subfolder in ("precisions", "recalls"):
                        path = eval_path / subfolder
                        os.makedirs(path, exist_ok=True)
                        np.save(path / name, locals()[subfolder])

                metrics[category.value] = {
                    "precisions": per_class_precisions,
                    "recalls": per_class_recalls,
                    "ap": ap,
                    "per_class_aps": per_class_aps if is_intersection else ap,
                    "mAP": np.mean(per_class_aps),
                }

            return metrics

        classifier_threshold_ranges_additive = {
            u_dataset.CategoryNames.BALL.value: self.threshold_range,
            u_dataset.CategoryNames.PENALTYMARK.value: self.threshold_range,
            u_dataset.CategoryNames.INTERSECTIONS.value: [
                {
                    u_dataset.IntersectionType.L.value: t,
                    u_dataset.IntersectionType.T.value: t,
                    u_dataset.IntersectionType.X.value: t,
                }
                for t in self.threshold_range
            ],
        }

        predictions_val, groundtruth_val = self.predict_on_data(self.val_ds)

        print("Calculating Classifier Metrics...")
        metrics_threshold_range_additive = self.get_classifier_metrics(
            predictions_val,
            groundtruth_val,
            self.config,
            classifier_threshold_ranges_additive,
            "additive",
            self.encoder_threshold,
            self.nms_iou_threshold,
            self.end_to_end,
        )

        print("Done!")
        return _calculate_ap_metrics(metrics_threshold_range_additive)

    def predict_on_data(self, dataset):
        print("Predicting on dataset...")
        predictions_list = []
        for batch in dataset:
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
        for x in dataset:
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

        print("Predictions Done!")
        return predictions_concat, groundtruth_concat

    def calculate_optimal_metrics(self, metrics: dict):
        print("Calculating Optimal Metrics...")
        optimal_thresholds = {}
        intersection_types = list(u_dataset.IntersectionType)[1:]  # Ignore None-Class

        for category in u_dataset.CategoryNames:
            if category.value not in metrics:
                continue

            if category == u_dataset.CategoryNames.INTERSECTIONS:
                optimal_thresholds[category.value] = []
                type_thresholds = {}
                for i, (precisions, recalls) in enumerate(
                    zip(
                        metrics[category.value]["precisions"],
                        metrics[category.value]["recalls"],
                        strict=True,
                    ),
                ):
                    name = intersection_types[i].value
                    threshold, _ = u_metrics.calculate_optimal_threshold(
                        self.beta, precisions, recalls, self.threshold_range
                    )
                    type_thresholds[name] = threshold

                optimal_thresholds[category.value].append(type_thresholds)
            else:
                threshold, _ = u_metrics.calculate_optimal_threshold(
                    self.beta,
                    metrics[category.value]["precisions"][0],
                    metrics[category.value]["recalls"][0],
                    self.threshold_range,
                )
                optimal_thresholds[category.value] = [threshold]

        predictions_test, ground_truth_test = self.predict_on_data(self.test_ds)

        optimal_metrics = self.get_classifier_metrics(
            predictions_test,
            ground_truth_test,
            self.config,
            optimal_thresholds,
            "additive",
            self.encoder_threshold,
            self.nms_iou_threshold,
            self.end_to_end,
        )

        print("Saving Results...")
        self.save_optimal_metrics(
            optimal_metrics, optimal_thresholds, predictions_test, ground_truth_test
        )

        return optimal_metrics

    def save_optimal_metrics(
        self, optimal_metrics: dict, optimal_thresholds: dict, predictions, ground_truth
    ):
        def to_serializable(obj):
            if isinstance(obj, dict):
                return {k: to_serializable(v) for k, v in obj.items()}
            if isinstance(obj, list):
                return [to_serializable(v) for v in obj]
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if hasattr(obj, "numpy"):  # TF tensor
                return obj.numpy().tolist()
            if isinstance(obj, np.generic):  # np.float32, np.int32, etc.
                return obj.item()
            return obj

        output = {}

        for category, values in optimal_metrics.items():
            output[category] = {
                "beta": self.beta,
                "thresholds": to_serializable(optimal_thresholds.get(category)),
                "precisions": to_serializable(values["precisions"]),
                "recalls": to_serializable(values["recalls"]),
                "confusion_matrix": to_serializable(values["confusion_matrix"]),
            }

        save_path = Path(
            self.config_dir.parent,
            "thresholded_metrics",
            f"{self.specification_string}.json",
        )
        save_path.parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, "w") as f:
            json.dump(output, f, indent=2)

        for object in u_dataset.CategoryNames:
            if object.value not in predictions["results"]:
                continue

            u_metrics.save_predictions(
                predictions["results"][object.value],
                ground_truth,
                object.value,
                Path(self.config_dir.parent, "predictions", self.specification_string),
                optimal_thresholds[object.value][0],
                self.encoder_threshold,
                self.nms_iou_threshold,
                image_res_scale=self.dataset_utils.config.image_res_scale,
            )


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
    parser.add_argument("--beta", type=float, required=False, default=0.5)
    parser.add_argument("--nms_iou", type=float, required=False, default=0.35)
    parser.add_argument("--model_dir", type=str)
    parser.add_argument("--cpn", type=bool, default=False, required=False)
    parser.add_argument("--classifier", type=bool, default=False, required=False)
    args = parser.parse_args()

    evaluator = Evaluator(args)
    evaluator.main()

    print("FINISHED!!!")
