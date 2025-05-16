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
    def __init__(self, model_path, directory):
        self.directory = directory
        self.labels = u_dataset.load_labels(directory)
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
        self.gs = gridspec.GridSpec(10, 8, figure=self.fig)

        self.ax_img = self.fig.add_subplot(self.gs[0:4, 0:5])
        self.ax_img.axis("off")
        self.ax_ball = self.fig.add_subplot(self.gs[0:4, 5:10])
        self.ax_ball.axis("off")
        self.ax_obstacles = self.fig.add_subplot(self.gs[4:8, 0:5])
        self.ax_obstacles.axis("off")
        self.ax_penalty_mark = self.fig.add_subplot(self.gs[4:8, 5:10])
        self.ax_penalty_mark.axis("off")

        self.ax_slider_image = self.fig.add_subplot(self.gs[9, :])
        self.slider_image = widgets.Slider(
            self.ax_slider_image,
            "Index",
            0,
            len(self.labels) - 1,
            valinit=0,
            valfmt="%i",
        )

        self.im_ax_img = self.ax_img.imshow(
            u_dataset.load_image(
                self.directory, self.labels[0], image_format=u_image.ImageFormat.RGB
            )
        )
        stuff = np.zeros((15, 20))
        stuff[0][0] = 1
        self.im_ax_ball = self.ax_ball.imshow(stuff)
        self.im_ax_obstacles = self.ax_obstacles.imshow(stuff)
        self.im_ax_penalty_mark = self.ax_penalty_mark.imshow(stuff)

        self.slider_image.on_changed(lambda val: self.image_slider_changed(val))
        self.fig.canvas.mpl_disconnect(self.fig.canvas.manager.key_press_handler_id)
        self.fig.canvas.mpl_connect("key_release_event", lambda event: self.key_released(event))

        self.select_image(0)

    def run(self):
        plt.show()

    def select_image(self, index):
        self.im_ax_img.set_data(
            u_dataset.load_image(
                self.directory,
                self.labels[index],
                image_format=u_image.ImageFormat.RGB,
            )
        )
        self.update_predictions(index)
        self.fig.canvas.draw()

    def update_predictions(self, index):
        image = u_dataset.load_image(
            self.directory, self.labels[index], image_format=self.image_format
        )
        predictions = 1.0 / (
            1.0 + np.exp(-self.model(image[np.newaxis, ...], training=False)[0].numpy())
        )
        self.im_ax_ball.set_data(predictions[:, :, 0])
        self.im_ax_penalty_mark.set_data(predictions[:, :, 1])
        self.im_ax_obstacles.set_data(predictions[:, :, 2])

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
    parser.add_argument("directory")
    parser.add_argument("model_path")
    args = parser.parse_args()

    app = EvaluateApplication(args.model_path, args.directory)
    app.run()
