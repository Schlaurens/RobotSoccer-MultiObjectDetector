import json
import os

from . import image as u_image


def get_label_path(directory):
    return os.path.join(directory, "labels.json")


def get_image_path(directory, name):
    return os.path.join(directory, f"{name}.bin")


def save_labels(directory, labels):
    with open(get_label_path(directory), "w") as f:
        json.dump(labels, f, indent=0)


def load_labels(directory):
    with open(get_label_path(directory), "r") as f:
        labels = json.load(f)
    return labels


def load_image(directory, label, **kwargs):
    with open(get_image_path(directory, label["name"]), "rb") as f:
        image = u_image.load_bhuman_jpeg_image(f.read(), **kwargs)
    return image


def load_image_direct(path, **kwargs):
    with open(path, "rb") as f:
        image = u_image.load_bhuman_jpeg_image(f.read(), **kwargs)
    return image
