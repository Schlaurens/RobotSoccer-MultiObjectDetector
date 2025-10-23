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

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import matplotlib.widgets as widgets
import numpy as np
import tensorflow as tf
from matplotlib import patches

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

        self.fig = plt.figure(figsize=(12, 8))
        self.gs = gridspec.GridSpec(11, 16, figure=self.fig)

        self.ax_ball_patches = self.fig.add_subplot(self.gs[0:4, 0:5])
        self.ax_ball_patches.axis("off")
        self.ax_ball_patches.set_title("Ball Patches")
        self.ax_ball = self.fig.add_subplot(self.gs[0:4, 5:10])
        self.ax_ball.axis("off")
        self.ax_ball.set_title("Ball")
        self.ax_ball_gt = self.fig.add_subplot(self.gs[0:4, 10:15])
        self.ax_ball_gt.axis("off")
        self.ax_ball_gt.set_title("Ball Groundtruth")
        self.ax_penalty_mark_patches = self.fig.add_subplot(self.gs[5:9, 0:5])
        self.ax_penalty_mark_patches.axis("off")
        self.ax_penalty_mark_patches.set_title("PenaltyMark Patches")
        self.ax_penalty_mark = self.fig.add_subplot(self.gs[5:9, 5:10])
        self.ax_penalty_mark.axis("off")
        self.ax_penalty_mark.set_title("PenaltyMark")
        self.ax_penalty_mark_gt = self.fig.add_subplot(self.gs[5:9, 10:15])
        self.ax_penalty_mark_gt.axis("off")
        self.ax_penalty_mark_gt.set_title("PenaltyMark Groundtruth")

        self.ax_slider_image = self.fig.add_subplot(self.gs[10, :])
        self.slider_image = widgets.Slider(
            self.ax_slider_image,
            "Index",
            0,
            len(self.data) - 1,
            valinit=0,
            valfmt="%i",
        )

        self.im_ax_ball_patches = self.ax_ball_patches.imshow(
            u_image.convert_yuyv_to_rgb(self.data[0]["image"])
        )
        self.im_ax_penalty_mark_patches = self.ax_penalty_mark_patches.imshow(
            u_image.convert_yuyv_to_rgb(self.data[0]["image"])
        )
        stuff = np.zeros((15, 20))
        stuff[0][0] = 1
        self.im_ax_ball = self.ax_ball.imshow(stuff)
        self.im_ax_ball_gt = self.ax_ball_gt.imshow(stuff)
        self.im_ax_penalty_mark = self.ax_penalty_mark.imshow(stuff)
        self.im_ax_penalty_mark_gt = self.ax_penalty_mark_gt.imshow(stuff)

        self.slider_image.on_changed(lambda val: self.image_slider_changed(val))
        self.fig.canvas.mpl_disconnect(self.fig.canvas.manager.key_press_handler_id)
        self.fig.canvas.mpl_connect("key_release_event", lambda event: self.key_released(event))

        self.select_image(0)

    def run(self):
        plt.show()

    def select_image(self, index):
        self.im_ax_ball_patches.set_data(u_image.convert_yuyv_to_rgb(self.data[index]["image"]))
        self.im_ax_penalty_mark_patches.set_data(
            u_image.convert_yuyv_to_rgb(self.data[index]["image"])
        )
        self.update_predictions(index)
        self.fig.canvas.draw()

        self.remove_artists()
        image_rgb = u_image.convert_yuyv_to_rgb(image)
            (
                self.data[index]["image"][np.newaxis, ...],
                self.data[index]["camera"][np.newaxis, ...],
                self.data[index]["intrinsics"][np.newaxis, ...],
            ),
            training=False,
        )

        output_penaltyMark = output["results"]["penaltyMark"]["logits"][
            0
        ].numpy()  # remove batch dimension

        # Set prediction figures
        self.im_ax_penalty_mark.set_data(np.reshape(output_penaltyMark, (15, 20)))

        # Set groundtruth figures
        self.im_ax_ball_gt.set_data(self.data[self.index]["ball"]["object_mask"])
        self.im_ax_penalty_mark_gt.set_data(self.data[self.index]["penaltyMark"]["object_mask"])
        self.im_ax_penalty_mark_patches.set_data(image_rgb)
        # Set patch figures
        self.draw_patches(image_rgb, self.ax_penalty_mark_patches, output, "penaltyMark")

    def remove_artists(self):
        """Remove all the Artists (texts and patches) for all the axes."""
        # Remove texts and patches
        for text in self.ax_ball_patches.texts:
            text.remove()
        for patch in self.ax_ball_patches.patches:
            patch.remove()
        for text in self.ax_penalty_mark_patches.texts:
            text.remove()
        for patch in self.ax_penalty_mark_patches.patches:
            patch.remove()

    def draw_patches(self, image, axes, output, object_name):
        for i, box in enumerate(
            output["results"][object_name]["boxes"][0]
        ):  # take index 0 to remove batch dimension
            patch_index = output["results"][object_name]["patch_indices"][0][i]

            if output["results"][object_name]["logits"][0][patch_index] < self.get_threshold(
                object_name
            ):
                return
            # Coordinates for each box are y1, x1, y2, x2
            # Upscale the normalized coordinates
            coords = (box[1] * (image.shape[1] - 1), box[0] * (image.shape[0] - 1))
            width = (box[3] - box[1]) * (image.shape[1] - 1)
            height = (box[2] - box[0]) * (image.shape[0] - 1)

            rect = patches.Rectangle(
                coords,
                width,
                height,
                linewidth=1,
                edgecolor="lime",
                facecolor=(255 / 255, 123 / 255, 0 / 255, 0 / 255),
            )

            # Each patch has a number to identify the ordering
            axes.text(x=(coords[0] + 4.0), y=coords[1] + 17.0, s=i + 1, color="lime")
            axes.add_patch(rect)

    def image_slider_changed(self, val):
        self.select_image(int(val))

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


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="This script shows the results of a model.")
    parser.add_argument("data_path")
    parser.add_argument("model_path")
    args = parser.parse_args()

    app = EvaluateApplication(args.model_path, args.data_path)
    app.run()
