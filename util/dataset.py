import json
import os

import tensorflow as tf
import numpy as np

from . import image as u_image


def get_label_path(directory):
    """Just a helper function to get the label path.

    Args:
        directory: directory of the dataset

    Returns:
        path to the JSON file with the labels
    """
    return os.path.join(directory, "labels.json")


def get_image_path(directory, name):
    """Get the path to an image with a given name from a given directory.

    Args:
        directory: directory of the dataset
        name: name of the image

    Returns:
        path to the image
    """
    return os.path.join(directory, f"{name}.jpg")


def save_labels(directory, labels):
    with open(get_label_path(directory), "w") as f:
        json.dump(labels, f, indent=0)


def load_labels(directory):
    with open(get_label_path(directory), "r") as f:
        labels = json.load(f)
    return labels


def load_image(directory, label, **kwargs):
    """Load image from a given directory and label.

    Args:
        directory: directory of the dataset
        label: corresponding label of the image

    Returns:
        the image
    """
    with open(get_image_path(directory, label["name"]), "rb") as f:
        image = u_image.load_bhuman_jpeg_image(f.read(), **kwargs)
    return image


def load_image_direct(path, **kwargs):
    with open(path, "rb") as f:
        image = u_image.load_bhuman_jpeg_image(f.read(), **kwargs)
    return image

def get_masks(label, object_name, input_dims= (480, 640), output_dims = (15, 20)):
    """Generate an offset mask that converts the image coordinates of the object
    into offsets relative to given cell dimensions. 
    And and objectsness mask that marks the cell where the center of the object is in.
    
    Args:
        label: label of the image
        input_dims: the full dimensions of the camera image.
        output_dims: the number of cells. Should be the same dimensions of encoder output
        
    Returns:
        an array of shape [output_dims_x, output_dims_y, 2]. Where the offset for each cell is
        portrayed in x and y coordinates.
    """

    if(object_name not in label):

        # If there are no objects of interest in the image all the offset get an arbitrary value. 
        # In the loss function all offsets of cell without an object are ignored anyway.
        offsets = tf.cast(tf.fill((*output_dims, 2), -1), dtype=tf.float32)

        # All cell are marked as false, as there are no object in the whole image.
        objectness_mask = tf.fill(output_dims, False)

        return offsets, objectness_mask

    coordinates = list(label[object_name].values())[:-1] #Only take x and y coordinates (ignore radius)

    # Make sure that input_dims are divisible by output_dims
    cell_dims = np.array(input_dims) // np.array(output_dims)
    scale = np.array(output_dims) / np.array(input_dims)

    # Generate the cell grid in the full image scale (values point to upper left corner of each cell)
    cells = tf.cast(tf.stack(tf.meshgrid(range(input_dims[1])[::cell_dims[1]], range(input_dims[0])[::cell_dims[0]]), axis=-1), dtype=tf.float32)

    offsets = coordinates - cells

    # Scale offsets to the output size
    offsets_scaled = offsets * scale

    # Mark all cells with true, where the value is between 0 and 1 (object is in that cell)
    objectness_mask = [[all(n>=0 and n<1 for n in x) for x in row ] for row in offsets_scaled]

    return offsets_scaled, objectness_mask

