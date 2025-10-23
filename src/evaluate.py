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

os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import matplotlib.widgets as widgets
import numpy as np
import tensorflow as tf

from util import dataset as u_dataset
from util import image as u_image


class EvaluateApplication:
    def __init__(self, model_path, data_path):
        self.data = list(u_dataset.get_dataset(data_path).as_numpy_iterator())
        self.model = tf.keras.models.load_model(model_path, compile=False)
        assert len(self.model.input_shape) == 4
        self.image_format = (
            u_image.ImageFormat.GRAYSCALE
            if self.model.input_shape[3] == 1
            else (
                u_image.ImageFormat.YUYV
                if self.model.input_shape[3] == 2
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

    def update_predictions(self, index):
        image = self.data[index]["image"]
        predictions = self.model(image[np.newaxis, ...], training=False)

        output_penaltyMark = predictions[0].numpy()  # remove batch dimension

        # Set prediction figures
        self.im_ax_penalty_mark.set_data(output_penaltyMark[..., 2])

        # Set groundtruth figures
        self.im_ax_ball_gt.set_data(self.data[index]["ball"]["object_mask"])
        self.im_ax_penalty_mark_gt.set_data(self.data[index]["penaltyMark"]["object_mask"])

    def normalize_array(self, arr):
        arr_min = arr.min()
        arr_max = arr.max()
        if arr_max == arr_min:
            return np.zeros_like(arr)

        return (arr - arr_min) / (arr_max - arr_min)

    def image_slider_changed(self, val):
        self.select_image(int(val))

    def key_released(self, event):
        if event.key in ["left", "right"]:
            current = int(self.slider_image.val)
            sign = 1 if event.key == "right" else -1
            current += sign
            self.slider_image.set_val(max(0, min(current, len(self.labels) - 1)))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="This script shows the results of a model.")
    parser.add_argument("data_path")
    parser.add_argument("model_path")
    args = parser.parse_args()

    app = EvaluateApplication(args.model_path, args.data_path)
    app.run()
