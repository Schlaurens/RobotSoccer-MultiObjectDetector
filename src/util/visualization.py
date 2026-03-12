import cv2
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import patches

from util import dataset as u_dataset
from util import dataset_io as u_dataset_io
from util import image as u_image


def show_masks_on_image(
    dataset_utils: u_dataset.DatasetUtils,
    directory=None,
    label=None,
    image=None,
    coordinates=None,
    object_name=None,
    mask_name=None,
):
    """Show the given image with an illustrated cell grid of given dimension.

    If an object_name is given and that object is present in the label the mask with the given mask_name is drawn.

    Input can be either (directory and label) or (coordinates and image).

    Args:
        directory: Directory of the image
        label: label of the image
        image: image in RGB format
        coordinates: image coordinates of the object
        object_name: name of the object. E. g. "ball". Defaults to None
        mask_name: Name of the mask that should be drawn. None := no mask, 'object' := Object mask, 'loss' := Loss mask
        grid_dims: The dimensions of the cell_grid. Defaults to (15,20).

    """

    dataset_config = dataset_utils.config

    if coordinates is not None and image is not None:
        masks = dataset_utils.get_masks(coordinates=coordinates)
    elif directory is not None and label is not None:
        image = u_dataset_io.load_image(directory, label, image_format=u_image.ImageFormat.RGB)
        image = cv2.resize(image, dataset_config.input_dims[::-1], cv2.INTER_AREA)

        masks = dataset_utils.get_masks(label, object_name)
    else:
        raise ValueError(
            "Either (directory and label) or (coordinates and image) must be provided."
        )

    cell_dims = dataset_config.cell_dims
    _, ax = plt.subplots()
    ax.imshow(image)
    ax.set_title(
        f"image_res={dataset_config.input_dims}, grid_dims={dataset_config.output_dims}, cell_size={cell_dims}, mask={mask_name}"
    )

    # Draw cell grid with the given grid dimensions
    for i in range(image.shape[1])[:: cell_dims[1]]:
        ax.axvline(x=i, color="black")
    for i in range(image.shape[0])[:: cell_dims[0]]:
        ax.axhline(y=i, color="black")

    if mask_name == None:
        plt.show()
        return

    if mask_name == "object":
        ax = _add_object_mask_axes(dataset_utils, masks, ax)
    elif mask_name == "loss":
        ax = _add_loss_mask_axes(dataset_config, masks, ax)
    elif mask_name == "classification":
        ax = _add_classification_mask_axes(dataset_config, masks, ax)
    else:
        print("Error: Unknown Mask requested.")
        return

    plt.show()


def _add_object_mask_axes(dataset_utils, masks, ax):
    dataset_config = dataset_utils.config

    coords = dataset_utils.get_coords_from_offsets(masks["offsets"]).numpy()
    for c in coords[0]:
        if -1.0 not in c:
            ax.plot(c[0], c[1], "rx")

            # Without this the plot expands in x and y axis.
            ax.set_xlim(0, dataset_config.input_dims[1])
            ax.set_ylim(dataset_config.input_dims[0], 0)

    object_mask = np.array(masks["object_mask"])

    # Get the indices with the highest/lowest values. Scaled to the cell_grid
    scaled_object_mask_indices_pos = np.dstack(np.where(object_mask == True))[0] * np.array(
        dataset_config.cell_dims
    )
    scaled_object_mask_indices_neg = np.dstack(np.where(object_mask == False))[0] * np.array(
        dataset_config.cell_dims
    )

    # Make sure that the indices are 2D arrays to make iteration possible in the next step.
    if len(scaled_object_mask_indices_pos.shape) == 1:
        scaled_object_mask_indices_pos = np.expand_dims(scaled_object_mask_indices_pos, axis=0)

    if len(scaled_object_mask_indices_neg.shape) == 1:
        scaled_object_mask_indices_neg = np.expand_dims(scaled_object_mask_indices_neg, axis=0)

    # Draw a rectangle on the cells
    for i in scaled_object_mask_indices_pos:
        rect_pos = patches.Rectangle(
            i[::-1],  # Flip x and y coordinates for matplotlib
            dataset_config.cell_dims[0],
            dataset_config.cell_dims[1],
            linewidth=1,
            edgecolor="black",
            facecolor=(63 / 255, 255 / 255, 0 / 255, 75 / 255),  # lime color
        )
        ax.add_patch(rect_pos)

    for i in scaled_object_mask_indices_neg:
        rect_pos = patches.Rectangle(
            i[::-1],  # Flip x and y coordinates for matplotlib
            dataset_config.cell_dims[0],
            dataset_config.cell_dims[1],
            linewidth=1,
            edgecolor="black",
            facecolor=(255 / 255, 0 / 255, 0 / 255, 75 / 255),  # red color
        )
        ax.add_patch(rect_pos)

    return ax


def _add_loss_mask_axes(dataset_config, masks, ax):
    loss_mask = np.array(masks["loss_mask"])

    scaled_loss_mask_indices_pos = np.dstack(np.where(loss_mask == True))[0] * np.array(
        dataset_config.cell_dims
    )
    scaled_loss_mask_indices_neg = np.dstack(np.where(loss_mask == False))[0] * np.array(
        dataset_config.cell_dims
    )

    if len(scaled_loss_mask_indices_pos.shape) == 1:
        scaled_loss_mask_indices_pos = np.expand_dims(scaled_loss_mask_indices_pos, axis=0)

    if len(scaled_loss_mask_indices_neg.shape) == 1:
        scaled_loss_mask_indices_neg = np.expand_dims(scaled_loss_mask_indices_neg, axis=0)

    for i in scaled_loss_mask_indices_pos:
        rect_loss_mask = patches.Rectangle(
            i[::-1],  # Flip x and y coordinates for matplotlib
            dataset_config.cell_dims[0],
            dataset_config.cell_dims[1],
            linewidth=1,
            edgecolor="black",
            facecolor=(63 / 255, 255 / 255, 0 / 255, 75 / 255),  # lime color
        )
        ax.add_patch(rect_loss_mask)

    for i in scaled_loss_mask_indices_neg:
        rect_loss_mask = patches.Rectangle(
            i[::-1],  # Flip x and y coordinates for matplotlib
            dataset_config.cell_dims[0],
            dataset_config.cell_dims[1],
            linewidth=1,
            edgecolor="black",
            facecolor=(255 / 255, 0 / 255, 0 / 255, 75 / 255),  # red color
        )
        ax.add_patch(rect_loss_mask)

    return ax


def _add_classification_mask_axes(dataset_config, masks, ax):
    class_mask = masks["classification_mask"]

    scaled_indices_no_intersections = np.dstack(
        np.where(class_mask == u_dataset.IntersectionType.NONE.value)
    )[0] * np.array(dataset_config.cell_dims)
    scaled_indices_l_intersections = np.dstack(
        np.where(class_mask == u_dataset.IntersectionType.L.value)
    )[0] * np.array(dataset_config.cell_dims)
    scaled_indices_t_intersections = np.dstack(
        np.where(class_mask == u_dataset.IntersectionType.T.value)
    )[0] * np.array(dataset_config.cell_dims)
    scaled_indices_x_intersections = np.dstack(
        np.where(class_mask == u_dataset.IntersectionType.X.value)
    )[0] * np.array(dataset_config.cell_dims)

    for i in scaled_indices_no_intersections:
        rect_loss_mask = patches.Rectangle(
            i[::-1],  # Flip x and y coordinates for matplotlib
            dataset_config.cell_dims[0],
            dataset_config.cell_dims[1],
            linewidth=1,
            edgecolor="black",
            facecolor=(255 / 255, 0 / 255, 0 / 255, 75 / 255),  # red color
        )
        ax.add_patch(rect_loss_mask)

    for i in scaled_indices_l_intersections:
        rect_loss_mask = patches.Rectangle(
            i[::-1],  # Flip x and y coordinates for matplotlib
            dataset_config.cell_dims[0],
            dataset_config.cell_dims[1],
            linewidth=1,
            edgecolor="black",
            facecolor=(227 / 255, 14 / 255, 236 / 255, 150 / 255),  # purple color
        )
        ax.add_patch(rect_loss_mask)

    for i in scaled_indices_t_intersections:
        rect_loss_mask = patches.Rectangle(
            i[::-1],  # Flip x and y coordinates for matplotlib
            dataset_config.cell_dims[0],
            dataset_config.cell_dims[1],
            linewidth=1,
            edgecolor="black",
            facecolor=(18 / 255, 18 / 255, 221 / 255, 120 / 255),  # blue color
        )
        ax.add_patch(rect_loss_mask)

    for i in scaled_indices_x_intersections:
        rect_loss_mask = patches.Rectangle(
            i[::-1],  # Flip x and y coordinates for matplotlib
            dataset_config.cell_dims[0],
            dataset_config.cell_dims[1],
            linewidth=1,
            edgecolor="black",
            facecolor=(63 / 255, 255 / 255, 0 / 255, 70 / 255),  # lime color
        )
        ax.add_patch(rect_loss_mask)

    return ax


def show_patches_on_image(image, label, results):
    """Draw the given image with rectangles that indicate the position of the extracted patches. And the patches in separate plots

    Args:
        image: the image in RGB format [480, 640, 3]
        label: the label of the object
        results: the results from the patch extractor. Contains for each detected object
            a number of patch candidates,
            masks that indicate for each patch whether the center could be projected to the plane,
            and the normalized corner coordinates of each patch (y1, x1, y2, x2)

    """
    image_res = image.shape[0:-1]
    num_candidates = results[label]["patches"].shape[1]

    # Draw image with patches on top of it
    _, axes = plt.subplots()
    axes.imshow(image[..., 0] / 255, cmap="gray")

    for i, box in enumerate(results[label]["boxes"][0]):  # take index 0 to remove batch dimension
        # Coordinates for each box are y1, x1, y2, x2
        # Upscale the normalized coordinates

        coords = (box[1] * (image_res[1] - 1), box[0] * (image_res[0] - 1))
        width = (box[3] - box[1]) * (image_res[1] - 1)
        height = (box[2] - box[0]) * (image_res[0] - 1)

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

    plt.title("Predicted Patches in Image.")

    # Draw the patch candidates in separate plots
    _, axes = plt.subplots(num_candidates)
    axes[0].imshow(results[label]["patches"][0, 0, ..., 0].numpy() / 255, cmap="gray")
    axes[1].imshow(results[label]["patches"][0, 1, ..., 0].numpy() / 255, cmap="gray")
    axes[2].imshow(results[label]["patches"][0, 2, ..., 0].numpy() / 255, cmap="gray")
    axes[3].imshow(results[label]["patches"][0, 3, ..., 0].numpy() / 255, cmap="gray")
    axes[4].imshow(results[label]["patches"][0, 4, ..., 0].numpy() / 255, cmap="gray")

    plt.show()
