"""evaluate.py
This script provides an interactive visualization tool for evaluating the predictions of a trained TensorFlow model on a labeled image dataset. It loads a model and dataset, displays images and their corresponding predictions in a heatmap, and allows navigation through the dataset using a slider or keyboard keys.

Usage:
    Run this script from the command line with the following arguments:
        python evaluate.py <data_path> <model_path>
    where <data_path> is the path to the dataset and <model_path> is the path to the trained model.
Arguments:
    data_path (str): Path to the .tfrecords file with the data that the model should be evaluated on.
    model_path (str): Path to the trained TensorFlow model.
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
    def __init__(self, model_path, data_path):
        path_to_model = "/".join(model_path.split("/")[:-2])
        model_name = model_path.split("/")[-1]
        resolution_string = model_path.split("/")[-3]
        if "checkpoints" in path_to_model:
            model_timestamp = path_to_model.split("/")[-1]
        else:
            model_timestamp = model_name.split(".")[0]

        config = self.load_config(f"logs/fit/{resolution_string}/{model_timestamp}/config.yaml")

        input_dims = config["model"]["encoder"]["input_dims"]
        cell_dims = config["model"]["encoder"]["cell_dims"]
        self.dataset_utils = u_dataset.DatasetUtils(
            u_dataset.DatasetConfig(input_dims, cell_dims=cell_dims)
        )
        self.data = list(
            u_dataset_io.get_dataset(data_path, self.dataset_utils).as_numpy_iterator()
        )
        self.model = self.load_model(config, path_to_model, model_name)
        self.categories = config["categories"]

        assert len(self.model.encoder.input_shape) == 4
        self.image_format = (
            u_image.ImageFormat.GRAYSCALE
            if self.model.encoder.input_shape[3] == 1
            else (
                u_image.ImageFormat.YUYV
                if self.model.encoder.input_shape[3] == 2
                else u_image.ImageFormat.YUV
            )
        )

        self.index = 0
        self.thresholds = {
            "encoder": {
                "ball": 0.1,
                "penaltyMark": 0.1,
                "intersections": 0.1,
            },
            "classifier": {
                "ball": 0.9,
                "penaltyMark": 0.9,
                "intersections": 0.9,
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

    def update_threshold(self, encoder: bool, object_name: str, val: float):
        self.thresholds["encoder" if encoder else "classifier"][object_name] = val

        print(
            f"Updated threshold for {'encoder' if encoder else 'classifier'} for {object_name} with new value {val}"
        )
        self.update_predictions()

    def run(self):
        plt.show()

    def select_image(self):
        self.update_predictions()
        self.fig.canvas.draw()

    def update_predictions(self):
        self.remove_artists()

        image = self.data[self.index]["image"]  # (H_in, W_in / 2, 4)

        output = self.model(
            (
                image[np.newaxis, ...],
                self.data[self.index]["camera"][np.newaxis, ...],
                self.data[self.index]["intrinsics"][np.newaxis, ...],
            ),
            training=False,
        )
        image_rgb = u_image.convert_yuyv_to_rgb(image)

        # Set prediction figures
        for category in self.categories:
            self.images[f"im_ax_{category}_patches"] = self.axes[f"ax_{category}_patches"].imshow(
                image_rgb
            )

            output_logits = output["results"][category]["logits"][0].numpy()
            self.images[f"im_ax_{category}"].set_data(
                np.reshape(output_logits, self.dataset_utils.config.output_dims)
            )
            iou_threshold = 0.35
            processed_predictions = u_metrics.handle_predictions(
                output["results"][category],
                self.thresholds["encoder"][category],
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
            self.images[f"im_ax_{category}_gt"].set_data(
                self.data[self.index][category]["object_mask"]
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
            The axes with the prediction. Or a zeros array if no object has been found that exceeds the combined threshold of encoder and classifier confidence.
        """
        if not processed_predictions["valid_samples"]:
            return axes.imshow(np.zeros(self.dataset_utils.config.cell_dims))

        best_score_index = processed_predictions["best_candidate_indices"][0]

        # Groundtruth coords
        coords_true = self.dataset_utils.get_coords_from_offsets(
            self.data[self.index][object_name]["offset_mask"]
        )[0]
        # Coords predicted by the encoder
        encoder_coords_pred = output["coords"][0][best_score_index]
        # Coords corrected by the classifier
        position_pred = output["positions"][0][best_score_index]

        abs_error = np.linalg.norm(coords_true - position_pred)
        best_box = output["boxes"][0][best_score_index]

        # We only need the width because the patch is a square.
        best_width = (best_box[3] - best_box[1]) * (self.dataset_utils.config.input_dims[1] / 2 - 1)
        # Used to scale the box with variable size to the fixed patch size
        patch_to_box_ratio = self.dataset_utils.config.cell_dims[0] / best_width

        box_coords = (
            best_box[1] * (self.data[self.index]["image"].shape[1] * 2 - 1),
            best_box[0] * (self.data[self.index]["image"].shape[0] - 1),
        )

        if ~tf.reduce_all(coords_true == -1.0):
            axes.plot(*(tf.squeeze(coords_true) - box_coords) * patch_to_box_ratio, "gx")
        axes.plot(*(encoder_coords_pred - box_coords) * patch_to_box_ratio, "rx")
        axes.plot(*(position_pred - box_coords) * patch_to_box_ratio, "bx")

        axes.text(0, 2, f"cand.: {best_score_index + 1}", color="lime")
        axes.text(
            0,
            4,
            f"enc.: {processed_predictions['encoder_confidences'][0].numpy():.3f}",
            color="lime",
        )
        axes.text(
            0,
            6,
            f"cla.: {processed_predictions['classifier_confidences'][0].numpy():.3f}",
            color="lime",
        )
        axes.text(0, 8, f"err.: {abs_error:.3f}", color="lime")
        return axes.imshow(output["patches"][0][best_score_index][..., 0], cmap="gray")

    def remove_artists(self):
        """Remove all the Artists (texts, patches and lines) for all the axes."""
        for ax in self.axes.values():
            for artist_type in ["lines", "texts", "patches"]:
                artists = getattr(ax, artist_type, [])
                for artist in artists:
                    artist.remove()

    def draw_patch_candidates(self, image, axes, output, processed_predictions, object_name):
        suppressed_indices = []
        if object_name == u_dataset.CategoryNames.INTERSECTIONS.value:
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
                logit < self.thresholds["encoder"][object_name]
                or tf.reduce_max(output["classification"][0][i], -1)
                < self.thresholds["classifier"][object_name]
            ):
                continue

            # Apply nms for intersections
            if (
                object_name == u_dataset.CategoryNames.INTERSECTIONS.value
                and i not in suppressed_indices
            ):
                continue

            # Coordinates for each box are y1, x1, y2, x2
            # Upscale the normalized coordinates
            box_coords = (box[1] * (image.shape[1] - 1), box[0] * (image.shape[0] - 1))
            width = (box[3] - box[1]) * (image.shape[1] - 1)
            height = (box[2] - box[0]) * (image.shape[0] - 1)

            rect = patches.Rectangle(
                box_coords,
                width,
                height,
                linewidth=1,
                edgecolor="lime",
                facecolor=(255 / 255, 123 / 255, 0 / 255, 0 / 255),
            )

            # Each patch has a number to identify the ordering
            # axes.text(x=(box_coords[0] + 4.0), y=box_coords[1] + 17.0, s=i + 1, color="lime")
            if object_name == u_dataset.CategoryNames.INTERSECTIONS.value:
                pred_patch_class = processed_predictions["classes_of_candidates"][0][i]

                axes.text(
                    x=(box_coords[0] + 4.0),
                    y=box_coords[1] + 17.0,
                    s=list(u_dataset.IntersectionType)[pred_patch_class.numpy()].value,
                    color="red",
                )

            axes.add_patch(rect)
            axes.plot(*coords_pred, "rx")
            axes.plot(*position_pred, "bx")

            coords_true = self.dataset_utils.get_coords_from_offsets(
                self.data[self.index][object_name]["offset_mask"]
            )[0]
            for c_true in coords_true:
                if tf.reduce_all(c_true == -1.0):
                    continue
                axes.plot(*c_true, "gx")

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
            encoder_architecture=config["model"]["encoder"]["architecture"],
            classifier_architecture=config["model"]["classifier"]["architecture"],
            input_dims=config["model"]["encoder"]["input_dims"],
            cell_dims=config["model"]["encoder"]["input_dims"],
            filepath=path_to_model,
            filename=model_name,
            n_context=config["model"]["encoder"]["n_context"],
            only_train_encoder=config["model"]["encoder"]["only_train_encoder"],
            classifier_offsets=config["model"]["classifier"]["with_offsets"],
            encoder_only=False,
            verbose=True,
            n_meta=config["model"]["classifier"]["n_meta"],
            encoder_use_batch_norm=config["model"]["encoder"]["use_batch_norm"],
            classifier_use_batch_norm=config["model"]["classifier"]["use_batch_norm"],
            categories_config=config["categories"],
        )
        model.compile(optimizer=tf.keras.optimizers.Adam(), jit_compile=False)
        return model

    def initialize_figures(self):
        print("Initializing Figures...")
        self.fig = plt.figure(figsize=(15, 8))
        self.gs = GridSpec(16, 18, figure=self.fig)

        # Define subplot configurations
        subplot_configs = [
            {"name": "ball", "rows": [0, 4], "cols": [[0, 5], [5, 10], [10, 15], [15, 18]]},
            {"name": "penaltyMark", "rows": [5, 9], "cols": [[0, 5], [5, 10], [10, 15], [15, 18]]},
            {
                "name": "intersections",
                "rows": [10, 14],
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
                "type": "encoder",
                "name": "ball",
                "pos": [0.1, 0.71, 0.0225, 0.16],
                "label": "enc",
            },
            {
                "type": "encoder",
                "name": "penaltyMark",
                "pos": [0.1, 0.465, 0.0225, 0.16],
                "label": "enc",
            },
            {
                "type": "encoder",
                "name": "intersections",
                "pos": [0.1, 0.22, 0.0225, 0.16],
                "label": "enc",
            },
            {
                "type": "classifier",
                "name": "ball",
                "pos": [0.075, 0.71, 0.0225, 0.16],
                "label": "cla",
            },
            {
                "type": "classifier",
                "name": "penaltyMark",
                "pos": [0.075, 0.465, 0.0225, 0.16],
                "label": "cla",
            },
            {
                "type": "classifier",
                "name": "intersections",
                "pos": [0.075, 0.22, 0.0225, 0.16],
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
        self.ax_slider_image = self.fig.add_subplot(self.gs[15, :])
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
            if (
                name != u_dataset.CategoryNames.INTERSECTIONS.value
            ):  # Intersectios don't have a results patch axis.
                self.images[f"im_ax_{name}_result"] = self.axes[f"ax_{name}_result"].imshow(
                    stuff_patch
                )

        # Connect slider events
        for category in self.categories:
            self.sliders[f"{category}_encoder_slider"].on_changed(
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
    parser.add_argument("model_path")
    args = parser.parse_args()

    app = EvaluateApplication(args.model_path, args.data_path)
    app.run()
