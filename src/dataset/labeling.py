import os
import sys
from enum import Enum

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import gridspec, widgets

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from util import augmentation as u_augmentation
from util import camera as u_camera
from util import dataset_io as u_dataset_io
from util import image as u_image
from util import labels as u_labels

# TODO: the "verification" modes should display a grid of multiple images at the same time


class LabelMode(Enum):
    BALL = 1
    OBSTACLES = 2
    PENALTY_MARK = 3
    INTERSECTION_X = 4
    INTERSECTION_T = 5
    INTERSECTION_L = 6


class LabelApplication:
    def __init__(self, directory):
        self.img_dims = (448, 544)  # y, x
        self.directory = directory
        self.labels = u_dataset_io.load_labels(args.directory)
        self.label_mode = LabelMode.BALL
        self.augmentation = False

        self.fig = plt.figure(figsize=(12, 8), num="Label Tool")
        self.gs = gridspec.GridSpec(10, 4, figure=self.fig)

        self.ax_img = self.fig.add_subplot(self.gs[0:8, :])
        self.ax_img.axis("off")
        self.ax_slider_image = self.fig.add_subplot(self.gs[9, :])

        self.slider_image = widgets.Slider(
            self.ax_slider_image,
            "Index",
            0,
            len(self.labels) - 1,
            valinit=0,
            valfmt="%i",
        )

        self.im_ax_img = self.ax_img.imshow(np.zeros(self.img_dims))

        self.slider_image.on_changed(lambda val: self.image_slider_changed(val))
        self.fig.canvas.mpl_disconnect(self.fig.canvas.manager.key_press_handler_id)
        self.fig.canvas.mpl_connect("key_release_event", lambda event: self.key_released(event))
        self.fig.canvas.mpl_connect(
            "button_press_event", lambda event: self.image_button_pressed(event)
        )
        self.fig.canvas.mpl_connect(
            "button_release_event", lambda event: self.image_button_released(event)
        )
        self.fig.canvas.mpl_connect("motion_notify_event", lambda event: self.on_motion(event))

        self.is_dragging = False
        self.drag_start_pos = None
        self.patches = []

        self.select_image(0)

    def run(self):
        plt.show()

    def select_image(self, index):
        label = self.labels[index]
        image = u_dataset_io.load_image(self.directory, label, image_format=u_image.ImageFormat.RGB)
        if self.augmentation:
            image, label = u_augmentation.apply(np.asarray(image), label)
        self.im_ax_img.set_data(image)
        self.redraw_labels(label)

    def redraw_labels(self, labels):
        for patch in self.patches:
            patch.remove()
        self.patches = []
        if u_labels.has_ball(labels):
            x, y, radius = u_labels.get_ball(labels)
            self.patches.append(
                self.ax_img.add_patch(
                    plt.Rectangle(
                        [x-radius, y-radius],
                        radius * 2, radius * 2,
                        color="r",
                        fill=False,
                    )
                )
            )
            self.patches.append(
                self.ax_img.add_patch(
                    plt.Rectangle(
                        (0, 0),
                        width=self.img_dims[1],
                        height=self.img_dims[0],
                        color="r",
                        fill=False,
                        linewidth=6,
                    )
                )
            )

            self.patches.append(self.ax_img.add_patch(plt.Circle((x, y), 1, color="r", fill=True)))
        if u_labels.has_penalty_mark(labels):
            x, y = u_labels.get_penalty_mark(labels)
            self.patches.append(
                self.ax_img.add_patch(plt.Circle((x, y), 32, color="b", fill=False))
            )
            self.patches.append(self.ax_img.add_patch(plt.Circle((x, y), 2, color="b", fill=True)))
        if u_labels.has_intersections(labels):
            intersections = u_labels.get_intersections(labels)
            # self.patches.append(
            #     self.ax_img.add_patch(
            #         plt.Rectangle(
            #             (0, 0),
            #             width=self.img_dims[1],
            #             height=self.img_dims[0],
            #             color="r" if intersections["ignore_sample"] else "b",
            #             fill=False,
            #             linewidth=4,
            #         )
            #     )
            # )
            for type in u_labels.IntersectionType:
                for intersection in intersections[type.value]:
                    self.patches.append(
                        self.ax_img.add_patch(
                            plt.Circle(
                                (intersection["x"], intersection["y"]), 2, color="m", fill=True
                            )
                        )
                    )
                    self.patches.append(
                        self.ax_img.text(
                            intersection["x"] - 7,
                            intersection["y"] - 7,
                            type.value,
                            color="m",
                        )
                    )
        if u_labels.has_obstacles(labels):
            mask = u_labels.get_obstacles(labels)
            for y, row in enumerate(mask):
                for x, value in enumerate(row):
                    if value > 0:
                        self.patches.append(
                            self.ax_img.add_patch(
                                plt.Rectangle(
                                    [x * 16, y * 16],
                                    16,
                                    16,
                                    alpha=0.5 * value,
                                    color="y",
                                )
                            )  # TODO: 16
                        )
        self.fig.canvas.draw()

    def image_slider_changed(self, val):
        self.select_image(int(val))

    def key_released(self, event):
        if event.key in ["left", "right"]:
            current = int(self.slider_image.val)
            sign = 1 if event.key == "right" else -1
            current += sign
            # If no intersection has been set, ignore this sample for intersections.
            if not u_labels.has_intersections(self.labels[current]):
                u_labels.set_ignore_intersection_sample_flag(self.labels[current], True)
            self.slider_image.set_val(max(0, min(current, len(self.labels) - 1)))
        elif event.key in ["up", "down"]:
            current = int(self.slider_image.val)
            if self.label_mode == LabelMode.BALL and u_labels.has_ball(self.labels[current]):
                x, y, r = u_labels.get_ball(self.labels[current])
                r += 1 if event.key == "up" else -1
                u_labels.set_ball(self.labels[current], x, y, r)
                self.redraw_labels(self.labels[current])
        elif event.key == "b":
            self.label_mode = LabelMode.BALL
        elif event.key == "o":
            self.label_mode = LabelMode.OBSTACLES
        elif event.key == "p":
            self.label_mode = LabelMode.PENALTY_MARK
        elif event.key == ",":
            self.label_mode = LabelMode.INTERSECTION_L
        elif event.key == ".":
            self.label_mode = LabelMode.INTERSECTION_T
        elif event.key == "-":
            self.label_mode = LabelMode.INTERSECTION_X
        elif event.key == "ö":  # Unignore the (non-existing) intersection labels in this sample
            current = int(self.slider_image.val)
            u_labels.set_ignore_intersection_sample_flag(self.labels[current], False)
            self.redraw_labels(self.labels[current])
        elif event.key == "ä":  # Ignore the (non-existing) intersection labels in this sample
            current = int(self.slider_image.val)
            u_labels.set_ignore_intersection_sample_flag(self.labels[current], True)
            self.redraw_labels(self.labels[current])
        elif event.key == "alt":
            self.unset_current_label()
        elif event.key == "cmd":
            plt.gcf().canvas.manager.toolbar.home()
        elif event.key == "s":
            u_dataset_io.save_labels(self.directory, self.labels)
            print("Labels saved!")
        elif event.key == "a":
            self.augmentation = not self.augmentation
            self.select_image(int(self.slider_image.val))
        else:
            print(event.key)

    def image_button_pressed(self, event):
        if event.inaxes != self.ax_img:
            return
        self.drag_start_pos = (event.xdata, event.ydata)

    def image_button_released(self, event):
        # If the cursor was dragged no new label will be set at this button release event.
        if self.is_dragging:
            self.is_dragging = False
            return

        # TODO: change augmentations and enable labelling accordingly.
        if event.inaxes != self.ax_img:  # or self.augmentation:
            return
        current = int(self.slider_image.val)
        if self.label_mode == LabelMode.BALL:
            camera_intr = u_dataset_io.intrinsics_from_label(self.labels[current])
            camera = u_dataset_io.camera_from_label(self.labels[current])
            ball_size = self.labels[current]["ball_size"]

            # Transform camera coords to world coords
            data_in_world = u_camera.image_to_world(
                camera, camera_intr, (event.xdata, event.ydata), object_height=ball_size
            )
            ballbbox = u_camera.project_sphere_bbox_square(
                data_in_world, ball_size / 2, camera, camera_intr, (event.xdata, event.ydata)
            )

            radius = float(
                max(ballbbox[..., 3] - ballbbox[..., 2], ballbbox[..., 1] - ballbbox[..., 0]) / 2
            )

            u_labels.set_ball(self.labels[current], event.xdata, event.ydata, radius)
        elif self.label_mode == LabelMode.OBSTACLES:
            x, y = int(event.xdata / 16), int(event.ydata / 16)  # TODO: 16
            x_start, y_start = (
                (int(self.drag_start_pos[0] / 16), int(self.drag_start_pos[1] / 16))
                if self.drag_start_pos
                else (x, y)
            )  # TODO: 16
            u_labels.set_obstacles(
                self.labels[current],
                min(x, x_start),
                min(y, y_start),
                max(x, x_start),
                max(y, y_start),
                op=u_labels.ObstaclesOp.SET
                if event.key == "control"
                else u_labels.ObstaclesOp.INVERT,
            )
        elif self.label_mode == LabelMode.PENALTY_MARK:
            u_labels.set_penalty_mark(self.labels[current], event.xdata, event.ydata)
        elif self.label_mode == LabelMode.INTERSECTION_L:
            u_labels.set_intersections(
                self.labels[current], event.xdata, event.ydata, u_labels.IntersectionType.L
            )
        elif self.label_mode == LabelMode.INTERSECTION_T:
            u_labels.set_intersections(
                self.labels[current], event.xdata, event.ydata, u_labels.IntersectionType.T
            )
        elif self.label_mode == LabelMode.INTERSECTION_X:
            u_labels.set_intersections(
                self.labels[current], event.xdata, event.ydata, u_labels.IntersectionType.X
            )
        self.redraw_labels(self.labels[current])
        self.drag_start_pos = None

    def unset_current_label(self):
        current = int(self.slider_image.val)
        if self.label_mode == LabelMode.BALL:
            u_labels.unset_ball(self.labels[current])
        elif self.label_mode == LabelMode.PENALTY_MARK:
            u_labels.unset_penalty_mark(self.labels[current])
        elif self.label_mode == LabelMode.INTERSECTION_L:
            u_labels.unset_intersection(self.labels[current], u_labels.IntersectionType.L)
        elif self.label_mode == LabelMode.INTERSECTION_T:
            u_labels.unset_intersection(self.labels[current], u_labels.IntersectionType.T)
        elif self.label_mode == LabelMode.INTERSECTION_X:
            u_labels.unset_intersection(self.labels[current], u_labels.IntersectionType.X)
        self.redraw_labels(self.labels[current])

    def on_motion(self, event):
        """Check whether the cursor has been dragged over a certain distance. If the cursor has been dragged more than 2 pixels the drag_distance variable gets set to True.

        Args:
            event: The motion_notify_event
        """
        if not event.button:
            return

        drag_distance = np.sqrt(
            (event.xdata - self.drag_start_pos[0]) ** 2
            + (event.ydata - self.drag_start_pos[1]) ** 2
        )

        if drag_distance > 3:
            self.is_dragging = True
            print(f"Dragging distance: {drag_distance}")
        else:
            self.is_dragging = False


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="This script manages labeled images.")
    parser.add_argument("directory")
    args = parser.parse_args()

    app = LabelApplication(args.directory)
    app.run()
