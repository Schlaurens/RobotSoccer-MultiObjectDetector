import numpy as np
import tensorflow as tf

from train.models import FullModel
from util import dataset as u_dataset
from util import image as u_image


def camera_from_label(label):
    """Calculate the camera roll pitch and height from the camera pose in the data.

    Args:
        label: the label with the camera pose

    Returns:
        A tuple of roll, pitch and height.
    """
    alpha = np.arccos(label["cpose"]["z"][2])
    if np.abs(alpha) < 0.01:
        roll = pitch = 0
    else:
        sin_alpha = np.sqrt(1 - label["cpose"]["z"][2] * label["cpose"]["z"][2])
        roll = label["cpose"]["z"][1] / sin_alpha * alpha
        pitch = -label["cpose"]["z"][0] / sin_alpha * alpha
    height = label["cpose"]["h"] * 0.001
    return (roll, pitch, height)


def intrinsics_from_label(label):
    """
    Get the camera intrinsics from the label.

    Args:
        label: A label from the dataset

    Returns:
        The camera intrinsics as a tuple (cx, cy, fx, fy).
    """

    return (label["cintr"]["cx"], label["cintr"]["cy"], label["cintr"]["fx"], label["cintr"]["fy"])


def get_dataset(directory):
    # Load the dataset
    # TODO: must be divisible by 32
    labels = u_dataset.load_labels(directory)[:736]

    images = []
    cameras = []
    intrinsics = []
    objectness_mask = []
    offsets = []
    loss_mask = []

    for label in labels:
        # Load the image for the label in YUYV format
        images.append(u_dataset.load_image(directory, label, image_format=u_image.ImageFormat.YUYV))
        # Load the camera pose for the label (roll, pitch, height)
        cameras.append(camera_from_label(label))
        # Load the camera intrinsics for the label
        intrinsics.append(intrinsics_from_label(label))
        
        masks = u_dataset.get_masks(label, "ball")
        # Load the offsets for the label
        offsets.append(masks[0])
        # Load the objectsness mask for the label
        objectness_mask.append(masks[1])
        # Load the loss mask for the label
        loss_mask.append(masks[2])
        

    # Combine the images, cameras and intrinsics into a single tensorflow dataset
    return tf.data.Dataset.from_tensor_slices(
        {
            "image": tf.reshape(tf.constant(images), (-1, 480, 320, 4)),
            "camera": tf.constant(cameras, dtype=tf.float32),
            "intrinsics": tf.constant(intrinsics, dtype=tf.float32),
            "objectness_mask": tf.reshape(tf.cast(objectness_mask, dtype=tf.float32), (-1, 15, 20)),
            "offsets": tf.reshape(offsets, (-1, 15, 20, 2)),
            "loss_mask": tf.reshape(tf.cast(loss_mask, dtype=tf.float32), (-1, 15, 20))
        }
    )


def main():
    train_ds = get_dataset(
        "/home/laurens/var/git/MA_LabelingTool/data/Joerg_Joerg_CompetitionWalk_GO2025__HULKs_2ndHalf_5"
    )
    train_ds = train_ds.shuffle(32)
    train_ds = train_ds.batch(32, drop_remainder=False)

    # Upper camera dimensions. Width is halved because of YUYV format
    model = FullModel(480, 320)
    model.compile(optimizer=tf.keras.optimizers.Adam())
    model.fit(x=train_ds, epochs=10)

    """
    results = model(image, camera, intrinsics)

    fig, axes = plt.subplots(5)
    axes[0].imshow(results["ball"][0][0, 0, ...].numpy() / 255)
    axes[1].imshow(results["ball"][0][0, 1, ...].numpy() / 255)
    axes[2].imshow(results["ball"][0][0, 2, ...].numpy() / 255)
    axes[3].imshow(results["ball"][0][0, 3, ...].numpy() / 255)
    axes[4].imshow(results["ball"][0][0, 4, ...].numpy() / 255)
    
    plt.show()
    print(results)
    """


if __name__ == "__main__":
    main()
