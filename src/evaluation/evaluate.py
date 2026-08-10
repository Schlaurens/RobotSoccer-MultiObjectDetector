"""evaluate.py
This script provides an interactive visualization tool for evaluating the predictions of a trained TensorFlow model on a labeled image dataset. It loads a model and dataset, displays images and their corresponding predictions in a heatmap, and allows navigation through the dataset using a slider or keyboard keys.

Usage:
    Run this script from the command line with the following arguments:
        python evaluate.py <data_path> <log_path>
    where <data_path> is the path to the dataset and <log_path> is the path to the log of the trained model.
Features:
    - Loads a trained model and dataset.
    - Displays the input image and model predictions for different object categories.
    - Interactive navigation through images using a slider or left/right arrow keys.
"""

import os
import sys

import yaml

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from matplotlib import patches
from matplotlib.gridspec import GridSpec
from matplotlib.widgets import Slider

from training.models import FullModel
from util import dataset as u_dataset
from util import dataset_io as u_dataset_io
from util import image as u_image
from util import metrics as u_metrics


class EvaluateApplication:
    def __init__(self, log_path, data_path, distance):
        self.config = self.load_config(log_path + "/config.yaml")

        self.categories = self.config["categories"]

        self.categories["ball"]["max_distance"] = distance
        # self.categories["penaltyMark"]["max_distance"] = distance
        # self.categories["intersections"]["max_distance"] = distance
        self.categories["robotBase"]["max_distance"] = distance

        self.categories["ball"]["n_candidates"] = 5
        # self.categories["penaltyMark"]["n_candidates"] = 4
        # self.categories["intersections"]["n_candidates"] = 11
        self.categories["robotBase"]["n_candidates"] = 11

        input_dims = self.config["model"]["cpn"]["input_dims"]
        cell_dims = self.config["model"]["cpn"]["cell_dims"]
        self.dataset_utils = u_dataset.DatasetUtils(
            u_dataset.DatasetConfig(input_dims, cell_dims=cell_dims)
        )

        self.data = list(
            u_dataset_io.get_dataset(data_path, self.dataset_utils).as_numpy_iterator()
        )

        self.full_utils = self.dataset_utils
        self.full_data = self.data

        path_to_model = self.get_model_path()

        self.model = self.load_model(
            self.config, path_to_model, self.config["metadata"]["timestamp"]
        )

        self.index = 0
        self.thresholds = {
            "cpn": {
                "ball": 0.01,
                "penaltyMark": 0.01,
                "intersections": 0.01,
                "robotBase": 0.01,
            },
            # "classifier": {
            #     "ball": 1.0,
            #     "penaltyMark": 1.0,
            #     "intersections": {
            #         u_dataset.IntersectionType.L.value: 0.949,
            #         u_dataset.IntersectionType.T.value: 0.9595959782600403,
            #         u_dataset.IntersectionType.X.value: 0.9898989796638489,
            #     },
            # },
            "classifier": {
                "ball": 0.0001,
                "penaltyMark": 1.0,
                "intersections": 0.9595959782600403,
                "robotBase": 0.0001,
            },
        }

        self.initialize_figures()
        self.select_image()

    def get_model_path(self):
        print("Finding Model...")
        timestamp = self.config["metadata"]["timestamp"]

        for root, dirs, _ in os.walk("models"):
            for dir_name in dirs[:]:
                if dir_name == timestamp:
                    full_path = os.path.join(root, dir_name)

        return full_path

    def update_threshold(self, cpn: bool, object_name: str, val: float):
        self.thresholds["cpn" if cpn else "classifier"][object_name] = val

        print(
            f"Updated threshold for {'cpn' if cpn else 'classifier'} for {object_name} with new value {val}"
        )
        self.update_predictions()

    def run(self):
        plt.show()

    def select_image(self):
        self.update_predictions()
        self.fig.canvas.draw()

    def update_predictions(self):
        self.remove_artists()

        image_rgb = u_image.convert_yuyv_to_rgb(self.data[self.index]["image"])  # (H_in, W_in, 3)

        image_rgb_full = u_image.convert_yuyv_to_rgb(
            self.full_data[self.index]["image"]
        )  # (H_in, W_in, 3)

        output = self.model(
            {
                "image": self.data[self.index]["image"][None, ...],
                "camera": self.data[self.index]["camera"][None, ...],
                "intrinsics": self.data[self.index]["intrinsics"][None, ...],
                "ball_size": self.data[self.index]["ball_size"][None, ...],
            },
            training=False,
        )

        # Set prediction figures
        for category in self.categories:
            self.images[f"im_ax_{category}_patches"] = self.axes[f"ax_{category}_patches"].imshow(
                image_rgb_full
            )

            output_logits = output["results"][category]["logits"][0].numpy()
            # self.images[f"im_ax_{category}"].set_data(
            #     np.reshape(output_logits, self.dataset_utils.config.output_dims)
            # )

            output_logits_sampled = tf.gather(
                tf.reshape(output_logits, [-1]),
                output["results"][category]["patch_indices"][0],
            )
            scatter_indices = tf.expand_dims(
                output["results"][category]["patch_indices"][0], axis=1
            )
            result_flat = tf.scatter_nd(
                scatter_indices,
                output_logits_sampled,
                [tf.reduce_prod(self.dataset_utils.config.output_dims)],
            )

            self.images[f"im_ax_{category}"].set_data(
                np.reshape(result_flat, self.dataset_utils.config.output_dims)
            )

            self.images[f"im_ax_{category}_gt"].set_data(
                self.data[self.index][category]["object_mask"]
            )

            if self.config["model"]["classifier"]["train_classifier"]:
                iou_threshold = 0.35
                processed_predictions = u_metrics.handle_predictions(
                    output["results"][category],
                    self.thresholds["cpn"][category],
                    self.thresholds["classifier"][category],
                    iou_threshold,
                )

                if category in [
                    u_dataset.CategoryNames.BALL.value,
                    u_dataset.CategoryNames.PENALTYMARK.value,
                ]:
                    self.images[f"im_ax_{category}_result"] = self.get_best_patch(
                        self.axes[f"ax_{category}_result"],
                        output["results"][category],
                        processed_predictions,
                        category,
                    )

                self.images[f"im_ax_{category}_patches"].set_data(image_rgb)

                self.draw_patch_candidates(
                    image_rgb,
                    self.axes[f"ax_{category}_patches"],
                    output["results"][category],
                    processed_predictions,
                    category,
                )

    def get_best_patch(self, axes, output, processed_predictions, object_name):
        """Find the best candidate and draw the patch with the predicted object position in the gives pyplot axes.

        Args:
            axes: Axes that will contain the patch and the predicted coordinates.
            output: The output of the classifier.
            object_name: The object name for which the best patch should be drawn

        Returns:
            The axes with the prediction. Or a zeros array if no object has been found that exceeds the combined threshold of cpn and classifier confidence.
        """
        if not processed_predictions["valid_samples"]:
            return axes.imshow(np.zeros(self.dataset_utils.config.cell_dims))

        best_score_index = processed_predictions["best_candidate_indices"][0]
        # Groundtruth coords
        coords_true = self.dataset_utils.get_coords_from_offsets(
            self.data[self.index][object_name]["offset_mask"]
        )[0]

        # Coords predicted by the cpn
        cpn_coords_pred = output["coords"][0][best_score_index]
        # Coords corrected by the classifier
        position_pred = output["positions"][0][best_score_index]

        # abs_error = np.linalg.norm(
        #     (coords_true - position_pred) / self.dataset_utils.config.image_res_scale[::-1]
        # )
        best_width = output["pixel_sizes"][0][best_score_index]

        patch_center = (np.array(self.model.patch_size) - 1) / 2
        coords_true_patch = patch_center + (tf.squeeze(coords_true) - cpn_coords_pred) * (
            self.model.patch_size / best_width
        )

        if ~tf.reduce_all(coords_true == -1.0):
            # tf.print(coords_true_patch)

            # axes.plot(*(coords_true_patch), "gx")
            pass

        # axes.plot(*(patch_center), "rx")  # CPN prediction is always in the middle of the patch
        # axes.plot(*(patch_center + output["classifier_offsets"][best_score_index]), "bx")

        axes.text(0, 2, f"cand.: {best_score_index + 1}", color="lime")
        axes.text(
            0,
            4,
            f"cpn: {processed_predictions['cpn_confidences'][0].numpy():.3f}",
            color="lime",
        )
        axes.text(
            0,
            6,
            f"cla.: {processed_predictions['classifier_confidences'][0].numpy():.3f}",
            color="lime",
        )
        return axes.imshow(
            u_image.convert_yuv_to_rgb(output["patches"][0][best_score_index][..., 0:3])
        )

    def remove_artists(self):
        """Remove all the Artists (texts, patches and lines) for all the axes."""
        for ax in self.axes.values():
            for artist_type in ["lines", "texts", "patches"]:
                artists = getattr(ax, artist_type, [])
                for artist in artists:
                    artist.remove()

    def draw_patch_candidates(self, image, axes, output, processed_predictions, object_name):
        suppressed_indices = []
        if object_name in [
            u_dataset.CategoryNames.INTERSECTIONS.value,
            u_dataset.CategoryNames.ROBOT_BASE.value,
        ]:
            suppressed_indices = tf.slice(
                processed_predictions["nms_selected_indices"][0],
                tf.constant([0]),
                processed_predictions["nms_num_valid"],
            )
        for i, box in enumerate(output["boxes"][0]):  # take index 0 to remove batch dimension
            patch_index = output["patch_indices"][0][i]
            logit = output["logits"][0][patch_index]
            coords_pred = output["coords"][0][i]
            position_pred = output["positions"][0][i]

            # dont draw patch if its prediction is under the threshold
            if (
                logit < self.thresholds["cpn"][object_name]
                or tf.reduce_max(output["classification"][0][i], -1)
                < self.thresholds["classifier"][object_name]
            ):
                continue

            # Apply nms for multi-class categories
            if (
                object_name
                in [
                    u_dataset.CategoryNames.INTERSECTIONS.value,
                    u_dataset.CategoryNames.ROBOT_BASE.value,
                ]
                and i not in suppressed_indices
            ):
                continue

            # Coordinates for each box are y1, x1, y2, x2
            # Upscale the normalized coordinates
            box_coords = (box[1] * (image.shape[1] - 1), box[0] * (image.shape[0] - 1))
            width = (box[3] - box[1]) * (image.shape[1] - 1)
            height = (box[2] - box[0]) * (image.shape[0] - 1)

            rect = patches.Rectangle(
                box_coords / self.dataset_utils.config.image_res_scale[::-1],
                width / self.dataset_utils.config.image_res_scale[0],
                height / self.dataset_utils.config.image_res_scale[1],
                linewidth=1,
                edgecolor="blue",
                facecolor=(255 / 255, 123 / 255, 0 / 255, 0 / 255),
            )

            # Each patch has a number to identify the ordering
            axes.text(
                x=(box_coords[0] / self.dataset_utils.config.image_res_scale[0] + 4.0),
                y=box_coords[1] / self.dataset_utils.config.image_res_scale[1] + 17.0,
                s=i,
                color="red",
            )
            if object_name == u_dataset.CategoryNames.INTERSECTIONS.value:
                pred_patch_class = processed_predictions["classes_of_candidates"][0][i]

                axes.text(
                    x=(box_coords[0] / self.dataset_utils.config.image_res_scale[0] + 4.0),
                    y=box_coords[1] / self.dataset_utils.config.image_res_scale[1] + 17.0,
                    s=list(u_dataset.IntersectionType)[pred_patch_class.numpy()].value,
                    color="red",
                )
                if pred_patch_class.numpy() != 0:
                    axes.add_patch(rect)
                    axes.plot(
                        *(position_pred / self.dataset_utils.config.image_res_scale[::-1]), "bx"
                    )
            else:
                axes.add_patch(rect)
                axes.plot(*coords_pred, "rx")
                # axes.plot(*(position_pred / self.dataset_utils.config.image_res_scale[::-1]), "bx")

                # axes.plot(*(position_pred), "bx")

            coords_true = self.dataset_utils.get_coords_from_offsets(
                self.data[self.index][object_name]["offset_mask"]
            )[0]
            for c_true in coords_true:
                if tf.reduce_all(c_true == -1.0):
                    continue
                axes.plot(*c_true / self.dataset_utils.config.image_res_scale[::-1], "gx")

    def image_slider_changed(self, val):
        self.index = int(val)
        self.select_image()

    def key_released(self, event):
        if event.key in ["left", "right"]:
            current = int(self.slider_image.val)
            sign = 1 if event.key == "right" else -1
            current += sign
            self.slider_image.set_val(max(0, min(current, len(self.data) - 1)))

    def load_config(self, config_path):
        print("Loading Config File...")
        with open(config_path) as f:
            config = yaml.safe_load(f)
        return config

    def load_model(self, config, path_to_model, model_name):
        print("Loading Model...")
        model = FullModel.load(
            cpn_architecture=config["model"]["cpn"]["architecture"],
            classifier_architecture=config["model"]["classifier"]["architecture"],
            input_dims=config["model"]["cpn"]["input_dims"] // np.array((1, 2)),
            cell_dims=config["model"]["cpn"]["cell_dims"],
            cpn_channels=config["model"]["cpn"]["channels_in"],
            filepath=path_to_model,
            filename=model_name,
            n_context=config["model"]["cpn"]["n_context"],
            train_cpn=True,
            train_classifier=config["model"]["classifier"]["train_classifier"],
            classifier_offsets=config["model"]["classifier"]["with_offsets"],
            cpn_only=False,
            verbose=True,
            n_meta=config["model"]["classifier"]["n_meta"],
            categories_config=config["categories"],
        )
        model.compile(optimizer=tf.keras.optimizers.Adam(), jit_compile=False)
        return model

    def initialize_figures(self):
        print("Initializing Figures...")
        self.fig = plt.figure(figsize=(15, 8))
        self.gs = GridSpec(20, 18, figure=self.fig)

        # Define subplot configurations
        subplot_configs = [
            {"name": "ball", "rows": [0, 4], "cols": [[0, 5], [5, 10], [10, 15], [15, 18]]},
            {"name": "penaltyMark", "rows": [5, 9], "cols": [[0, 5], [5, 10], [10, 15], [15, 18]]},
            {
                "name": "intersections",
                "rows": [10, 14],
                "cols": [[0, 5], [5, 10], [10, 15]],
            },
            {
                "name": "robotBase",
                "rows": [15, 19],
                "cols": [[0, 5], [5, 10], [10, 15]],
            },
        ]
        self.axes = {}
        for config in subplot_configs:
            name = config["name"]
            rows = config["rows"]
            for i, cols in enumerate(config["cols"]):
                ax_name = f"ax_{name}{['_patches', '', '_gt', '_result'][i]}"
                self.axes[ax_name] = self.fig.add_subplot(
                    self.gs[rows[0] : rows[1], cols[0] : cols[1]]
                )
                self.axes[ax_name].axis("off")
                self.axes[ax_name].set_title(
                    f"{name.replace('_', ' ').title()} {['Patches', '', 'Groundtruth', 'Result'][i]}"
                )

        # Initialize sliders
        slider_configs = [
            {
                "type": "cpn",
                "name": "ball",
                "pos": [0.1, 0.74, 0.0225, 0.14],
                "label": "cpn",
            },
            {
                "type": "cpn",
                "name": "penaltyMark",
                "pos": [0.1, 0.54, 0.0225, 0.14],
                "label": "cpn",
            },
            {
                "type": "cpn",
                "name": "intersections",
                "pos": [0.1, 0.35, 0.0225, 0.14],
                "label": "cpn",
            },
            {
                "type": "cpn",
                "name": "robotBase",
                "pos": [0.1, 0.16, 0.0225, 0.14],
                "label": "cpn",
            },
            {
                "type": "classifier",
                "name": "ball",
                "pos": [0.075, 0.74, 0.0225, 0.14],
                "label": "cla",
            },
            {
                "type": "classifier",
                "name": "penaltyMark",
                "pos": [0.075, 0.54, 0.0225, 0.14],
                "label": "cla",
            },
            {
                "type": "classifier",
                "name": "intersections",
                "pos": [0.075, 0.35, 0.0225, 0.14],
                "label": "cla",
            },
            {
                "type": "classifier",
                "name": "robotBase",
                "pos": [0.075, 0.16, 0.0225, 0.14],
                "label": "cla",
            },
        ]

        self.sliders = {}
        for config in slider_configs:
            name = config["name"]
            slider_type = config["type"]
            pos = config["pos"]
            label = config["label"]

            axis = self.fig.add_axes(pos)
            self.sliders[f"{name}_{slider_type}_slider"] = Slider(
                ax=axis,
                label=label,
                valmin=0,
                valmax=1,
                valinit=self.thresholds[slider_type][name],
                orientation="vertical",
            )

        # Image slider
        self.ax_slider_image = self.fig.add_subplot(self.gs[19, :])
        self.slider_image = Slider(
            self.ax_slider_image,
            "Index",
            0,
            len(self.data) - 1,
            valinit=0,
            valfmt="%i",
        )

        # Initialize images
        stuff = np.zeros(self.dataset_utils.config.output_dims)
        stuff[0][0] = 1
        stuff_patch = np.zeros(self.dataset_utils.config.cell_dims)
        stuff_patch[0][0] = 1

        self.images = {}
        for subplot in subplot_configs:
            name = subplot["name"]
            self.images[f"im_ax_{name}_patches"] = self.axes[f"ax_{name}_patches"].imshow(
                u_image.convert_yuyv_to_rgb(self.data[0]["image"])
            )
            self.images[f"im_ax_{name}"] = self.axes[f"ax_{name}"].imshow(stuff)
            self.images[f"im_ax_{name}_gt"] = self.axes[f"ax_{name}_gt"].imshow(stuff)
            if name not in [
                u_dataset.CategoryNames.INTERSECTIONS.value,
                u_dataset.CategoryNames.ROBOT_BASE.value,
            ]:  # Multi-Class categories don't have a results patch axis.
                self.images[f"im_ax_{name}_result"] = self.axes[f"ax_{name}_result"].imshow(
                    stuff_patch
                )

        # Connect slider events
        for category in self.categories:
            self.sliders[f"{category}_cpn_slider"].on_changed(
                lambda val, category=category: self.update_threshold(True, category, val)
            )
            self.sliders[f"{category}_classifier_slider"].on_changed(
                lambda val, category=category: self.update_threshold(False, category, val)
            )
        self.slider_image.on_changed(lambda val: self.image_slider_changed(val))

        self.fig.canvas.mpl_disconnect(self.fig.canvas.manager.key_press_handler_id)
        self.fig.canvas.mpl_connect("key_release_event", lambda event: self.key_released(event))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="This script shows the results of a model.")
    parser.add_argument("data_path")
    parser.add_argument("log_path")
    parser.add_argument("--distance", type=int, default=9)
    args = parser.parse_args()

    app = EvaluateApplication(args.log_path, args.data_path, args.distance)
    app.run()
