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

import yaml

os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from matplotlib import patches
from matplotlib.gridspec import GridSpec
from matplotlib.widgets import Slider

from train.models import FullModel
from util import dataset as u_dataset
from util import image as u_image


class EvaluateApplication:
    def __init__(self, model_path, data_path):
        self.data = list(u_dataset.get_dataset(data_path).as_numpy_iterator())

        config = self.load_config(f"logs/fit/{model_path.split('/')[-1].split('.')[0]}/config.yaml")

        # self.model = tf.keras.models.load_model(model_path, compile=False)
        self.model = self.load_model(config, model_path)

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

        self.fig = plt.figure(figsize=(15, 8))
        self.gs = GridSpec(11, 18, figure=self.fig)

        self.ax_ball_patches = self.fig.add_subplot(self.gs[0:4, 0:5])
        self.ax_ball_patches.axis("off")
        self.ax_ball_patches.set_title("Ball Patches")

        self.ax_ball = self.fig.add_subplot(self.gs[0:4, 5:10])
        self.ax_ball.axis("off")
        self.ax_ball.set_title("Ball")

        self.ax_ball_gt = self.fig.add_subplot(self.gs[0:4, 10:15])
        self.ax_ball_gt.axis("off")
        self.ax_ball_gt.set_title("Ball Groundtruth")

        self.ax_ball_result = self.fig.add_subplot(self.gs[0:4, 15:18])
        self.ax_ball_result.axis("off")
        self.ax_ball_result.set_title("Ball Result")

        self.ax_penalty_mark_patches = self.fig.add_subplot(self.gs[5:9, 0:5])
        self.ax_penalty_mark_patches.axis("off")
        self.ax_penalty_mark_patches.set_title("PenaltyMark Patches")

        self.ax_penalty_mark = self.fig.add_subplot(self.gs[5:9, 5:10])
        self.ax_penalty_mark.axis("off")
        self.ax_penalty_mark.set_title("PenaltyMark")

        self.ax_penalty_mark_gt = self.fig.add_subplot(self.gs[5:9, 10:15])
        self.ax_penalty_mark_gt.axis("off")
        self.ax_penalty_mark_gt.set_title("PenaltyMark Groundtruth")

        self.ax_penalty_mark_result = self.fig.add_subplot(self.gs[5:9, 15:18])
        self.ax_penalty_mark_result.axis("off")
        self.ax_penalty_mark_result.set_title("Ball Result")

        self.penalty_mark_threshold = 0.8
        self.ball_threshold = 0.8

        self.index = 0
        self.thresholds = {
            "encoder": {
                "ball": 0.8,
                "penaltyMark": 0.8,
            },
            "classifier": {
                "ball": 0.6,
                "penaltyMark": 0.6,
            },
        }
        self.select_image()

    def update_threshold(self, encoder: bool, object_name: str, val: float):
        self.thresholds["encoder" if encoder else "classifier"][object_name] = val
        self.update_predictions()

    def run(self):
        plt.show()

    def select_image(self):
        self.im_ax_ball_patches.set_data(
            u_image.convert_yuyv_to_rgb(self.data[self.index]["image"])
        )
        self.im_ax_penalty_mark_patches.set_data(
            u_image.convert_yuyv_to_rgb(self.data[self.index]["image"])
        )
        self.update_predictions()
        self.fig.canvas.draw()

    def update_predictions(self):
        self.remove_artists()

        image = self.data[self.index]["image"]
        image_rgb = u_image.convert_yuyv_to_rgb(image)
        output = self.model(
            (
                image[np.newaxis, ...],
                self.data[self.index]["camera"][np.newaxis, ...],
                self.data[self.index]["intrinsics"][np.newaxis, ...],
            ),
            training=False,
        )
        print(output["results"]["penaltyMark"]["classification"][0].numpy())
        output_penaltyMark = output["results"]["penaltyMark"]["logits"][
            0
        ].numpy()  # remove batch dimension

        # Set prediction figures
        self.im_ax_penalty_mark.set_data(np.reshape(output_penaltyMark, (15, 20)))

        # Set best patches
        self.im_ax_penalty_mark_result = self.get_best_patch(
            self.ax_penalty_mark_result, output, "penaltyMark"
        )

        # Set groundtruth figures
        self.im_ax_ball_gt.set_data(self.data[self.index]["ball"]["object_mask"])
        self.im_ax_penalty_mark_gt.set_data(self.data[self.index]["penaltyMark"]["object_mask"])
        self.im_ax_penalty_mark_patches.set_data(image_rgb)
        # Set patch figures
        self.draw_patch_candidates(image_rgb, self.ax_penalty_mark_patches, output, "penaltyMark")

    def get_best_patch(self, axes, output, object_name):
        """Find the best candidate and draw the patch with the predicted object position in the gives pyplot axes.

        Args:
            axes: Axes that will contain the patch and the predicted coordinates.
            output: The output of the classifier.
            object_name: The object name for which the best patch should be drawn

        Returns:
            The axes with the prediction. Or a zeros array if no object has been found that exceeds the combined threshold of encoder and classifier confidence.
        """
        patch_indices = output["results"][object_name]["patch_indices"][0]
        best_logits = [output["results"][object_name]["logits"][0][i] for i in patch_indices]
        # The sum of the encoder's and classifier's prediction values
        combined_predictions = best_logits + output["results"][object_name]["classification"][0]

        best_score_index = np.argmax(combined_predictions)
        # Get the offset predicted by the classifier. This works because (position = coords + classifier_offset).
        best_classifier_offset = (
            output["results"][object_name]["positions"][0][best_score_index]
            - output["results"][object_name]["coords"][0][best_score_index]
        )
        best_box = output["results"][object_name]["boxes"][0][best_score_index]

        # We only need the width because the patch is a square.
        best_width = (best_box[3] - best_box[1]) * (640 - 1)

        # Used to scale the box with variable size to the fixed patch size
        patch_to_box_ratio = 32 / best_width

        # The classifier_offset need to be added the center coordinates of the patch.
        best_position = (best_width / 2 + best_classifier_offset) * patch_to_box_ratio

        if (
            combined_predictions[best_score_index]
            >= self.thresholds["encoder"][object_name] + self.thresholds["classifier"][object_name]
        ):
            axes.plot(*best_position, "bx")
            axes.text(x=0.0, y=2.0, s=best_score_index + 1, color="lime")
            return axes.imshow(
                output["results"][object_name]["patches"][0][best_score_index][..., 0], cmap="gray"
            )
        else:
            return axes.imshow(np.zeros((32, 32)))

    def remove_artists(self):
        """Remove all the Artists (texts, patches and lines) for all the axes."""
        # Remove texts and patches
        for text in self.ax_ball_patches.texts:
            text.remove()
        for patch in self.ax_ball_patches.patches:
            patch.remove()
        for line in self.ax_ball_patches.lines:
            line.remove()
        for line in self.ax_ball_result.lines:
            line.remove()
        for text in self.ax_penalty_mark_patches.texts:
            text.remove()
        for patch in self.ax_penalty_mark_patches.patches:
            patch.remove()
        for line in self.ax_penalty_mark_patches.lines:
            line.remove()
        for line in self.ax_penalty_mark_result.lines:
            line.remove()
        for test in self.ax_penalty_mark_result.texts:
            test.remove()

    def draw_patch_candidates(self, image, axes, output, object_name):
        for i, box in enumerate(
            output["results"][object_name]["boxes"][0]
        ):  # take index 0 to remove batch dimension
            patch_index = output["results"][object_name]["patch_indices"][0][i]
            logit = output["results"][object_name]["logits"][0][patch_index]
            coords_pred = output["results"][object_name]["coords"][0][i]
            coords_true = u_dataset.get_coords_from_offsets(self.data[self.index][object_name]["offset_mask"])
            position_pred = output["results"][object_name]["positions"][0][i]

            # dont draw patch if its prediction is under the threshold
            if logit < self.thresholds["encoder"][object_name]:
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
            axes.text(x=(box_coords[0] + 4.0), y=box_coords[1] + 17.0, s=i + 1, color="lime")
            axes.add_patch(rect)
            axes.plot(coords_pred[0], coords_pred[1], "rx")
            axes.plot(position_pred[0], position_pred[1], "bx")
            axes.plot(coords_true[0], coords_true[1], "gx")
            
            print(np.linalg.norm(position_pred - coords_true))

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
        with open(config_path) as f:
            config = yaml.safe_load(f)
        return config

    def load_model(self, config, model_path):
        model = FullModel.load(
            encoder_architecture=config["model"]["encoder"]["architecture"],
            classifier_architecture=config["model"]["classifier"]["architecture"],
            input_dims=config["model"]["encoder"]["input_dims"],
            filepath="/".join(model_path.split("/")[:-2]),
            filename=model_path.split("/")[-1],
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

    def get_threshold(self, object_name):
        if object_name == "ball":
            return self.ball_threshold
        elif object_name == "penaltyMark":
            return self.penalty_mark_threshold

    def set_threshold(self, object_name, value):
        if object_name == "ball":
            self.ball_threshold = value
        if object_name == "penaltyMark":
            self.penalty_mark_threshold = value


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="This script shows the results of a model.")
    parser.add_argument("data_path")
    parser.add_argument("model_path")
    args = parser.parse_args()

    app = EvaluateApplication(args.model_path, args.data_path)
    app.run()
