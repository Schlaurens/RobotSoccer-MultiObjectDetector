import numpy as np
import scipy
import tensorflow as tf

from . import camera as u_camera
from . import dataset as u_dataset
from . import keypoint as u_keypoint


class Error_Metric(tf.keras.metrics.Metric):
    """
    A base class for custom error metrics.

    This class provides common functionality for calculating error between predicted
    and actual values. It handles state updates, resetting state, and matching keypoints.

    Attributes:
        err_threshold (float): The maximum error threshold for considering a match (in meters). Defaults to 0.2 m
        object_name (str): The name of the object being tracked.
        abs_error (tf.Variable): The accumulated absolute error.
        num_samples (tf.Variable): Number of samples processed.
    """

    def __init__(self, object_name, err_threshold=0.2, name="error_metric", **kwargs):
        """
        Initialize the ErrorMetric instance.

        Args:
            object_name (str): Name of the object being tracked.
            err_threshold (float): Threshold for error to consider a valid match. (in m)
            name (str): Name of the metric instance.
            **kwargs: Additional keyword arguments passed to the parent class.
        """
        super().__init__(name=name, **kwargs)
        self.err_threshold = err_threshold
        self.object_name = object_name
        self.abs_error = self.add_weight(name="abs_error", initializer="zeros")
        self.num_samples = self.add_weight(name="num_samples", initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):
        """
        Update the metric state given y_true and y_pred.

        Args:
            y_true (dict): Ground truth values.
            y_pred (dict): Predicted values.
            sample_weight: Optional weighting of the samples. Defaults to None.
        """
        threshold = self.get_threshold()

        match = self.match_keypoints(y_pred, y_true, threshold)

        # If there were false predictions add the threshold distance as a penalty for every false prediction
        if match["num_false_predictions"] > 0:
            # self.abs_error.assign_add(match["num_false_predictions"] * threshold)
            self.add_error(error=match["num_false_predictions"], multiplier=threshold)
            self.num_samples.assign_add(match["num_false_predictions"])

        for pair in match["matches"]:
            distance = 0

            # If all element is in pair are 0.0, there are no objects in the image
            # do not count them towards the metric
            if (pair == 0).all():
                continue

            # If a valid pair was found
            distance += tf.norm(pair[0] - pair[1])
            # self.abs_error.assign_add(distance)
            self.add_error(error=distance)
            self.num_samples.assign_add(1.0)

        # if sample_weight is not None:
        #     sample_weight = tf.cast(sample_weight, "float32")
        #     values = tf.multiply(values, sample_weight)

    def add_error(self, error, multiplier=1):
        raise NotImplementedError

    def get_threshold(self):
        """
        Return the error threshold.

        Returns:
            float: The error threshold.
        """
        # TODO: make dependent on distance from object to camera.
        return self.err_threshold

    def result(self):
        return self.abs_error / self.num_samples

    def reset_state(self):
        # The state of the metric will be reset at the start of each epoch.
        self.abs_error.assign(0.0)
        self.num_samples.assign(0.0)

    def match_keypoints(self, y_pred, y_true, threshold):
        """
        Matches predicted and labeled points optimally. Points are only matched as
        long as their distance is below or equal to threshold. It does not match two
        points from kps to the same point in pts or vice versa. Correspondingly it can
        happen that some points remain unmatched.

        A matrix is constructed that contains a score for each pair of detected and
        labeled point. The score is 0 for pairs that are at least threshold pixels apart
        and 1 for points with distance 0 (cf. the error metric for point detectors from
        lecture 11a).

        Then scipy.optimize.linear_sum_assignment calculates the assignment that maximizes
        the overall score. Dummy assignments (with score 0) are filtered out before the
        matched points are stacked together to obtain the result.

        Args:
            y_pred (dict): Predicted values. Used to calculate predicted world coordinates of object.
            y_true (dict): Groundtruth values. Used to calculate groundtruth world coordinates of object.
            threshold (float): Max distance (in m) to consider a pair a match.

        Returns:
            dict: Contains matched points and the number of false predictions.
        """

        img_coords_true = u_dataset.get_coords_from_offsets(
            y_true[self.object_name]["offset_mask"]
        )  # (x, y)
        img_coords_pred = u_dataset.get_coords_from_offsets(
            y_pred[self.object_name]["offset_mask"]
        )  # (x, y)
        pts = tf.expand_dims(
            u_camera.image_to_world(y_true["camera"], y_true["intrinsics"], img_coords_true), axis=0
        ).numpy()  # (1, 3)
        kps = tf.expand_dims(
            u_camera.image_to_world(y_pred["camera"], y_pred["intrinsics"], img_coords_pred), axis=0
        ).numpy()  # (1, 3)

        if kps.size == 0 or pts.size == 0:
            return np.zeros((0, 2, 2), dtype=kps.dtype)
        diffs = kps[:, np.newaxis] - pts[np.newaxis]
        score_matrix = np.maximum(1 - np.linalg.norm(diffs, axis=-1) / threshold, 0)
        row_ind, col_ind = scipy.optimize.linear_sum_assignment(score_matrix, maximize=True)
        assigned = score_matrix[row_ind, col_ind] > 0

        # Number of points for which no match could be found, because the threshold was reached (false positive/negatives)
        num_of_unassigned = assigned.size - np.count_nonzero(assigned)

        # Number of points that were sorted out in linear_sum_assignment (in case of non-square score matrix)
        num_discarded_points = np.abs(score_matrix.shape[0] - score_matrix.shape[1])

        num_false_predictions = num_of_unassigned + num_discarded_points

        return {
            "matches": np.stack([kps[row_ind[assigned]], pts[col_ind[assigned]]], axis=1),
            "num_false_predictions": num_false_predictions,
        }


class MAE(Error_Metric):
    """
    A custom metric class for calculating Mean Absolute Error.

    This class extends the ErrorMetric base class for calculating the MAE
    """

    def __init__(self, object_name="ball", err_threshold=0.2, name="mae", **kwargs):
        super().__init__(object_name=object_name, err_threshold=err_threshold, name=name, **kwargs)

    def add_error(self, error, multiplier=1):
        self.abs_error.assign_add(error * multiplier)


class RMSE(Error_Metric):
    """
    A custom metric class for calculating Root Mean Squared Error.

    This class extends the ErrorMetric base class for calculating the RMSE,
    which squares the error to emphasize outliers.

    Attributes:
        scaling_factor (int): Used to internally scale the error unit between meters and cm.
    """

    def __init__(self, object_name="ball", err_threshold=0.2, name="mae", **kwargs):
        super().__init__(object_name=object_name, err_threshold=err_threshold, name=name, **kwargs)
        self.scaling_factor = 100

    def add_error(self, error, multiplier=1):
        # Scale the error from m to cm, so that the error of outlier is bigger.
        self.abs_error.assign_add(((error * self.scaling_factor) ** 2) * multiplier)

    def result(self):
        # Scale the RMSE back to m.
        return tf.sqrt(self.abs_error / self.num_samples) / self.scaling_factor


def calculate_binary_metrics(
    predictions,
    groundtruth,
    encoder_threshold,
    classifier_threshold,
    include_encoder_logits=False,
):
    """Calculate y_pred. A binary tensor which is True if an object was detected in the sample and False if no object was detected.

    A prediction counts as positive if:
        - the combined prediction (encoder prediction + classifier prediction) is greater-equal than the combined threshold (encoder_threshold + classifier_threshold)
        - The predicted patch coordinates could be projected on the ground
        - If the predicted patch is actually over the object.

    Args:
        predictions: The model predictions.
        groundtruth: The corresponding groundtruth data.
        encoder_threshold: The threshold of the encoder.
        classifier_threshold: The threshold of the classifier.
        include_encoder_logits: Whether the predicted probably of the encoder should be considered when getting the patch with the highest probility. Defaults to False

    Returns:
        dict() containing confusion matrix, precision, recall, indices of false_positives and false_negatives, false_positive rate and false_negative rate
    """

    # Get the encoder logits where a patch was drawn (those are the one with the highest predicted probability)
    best_logits = tf.gather(
        predictions["logits"], predictions["patch_indices"], batch_dims=1
    )  # (B, N)

    # Add the encoder prediction and the classifier prediction to a combined prediction
    combined_predictions = (
        best_logits + predictions["classification"]
        if include_encoder_logits
        else predictions["classification"]
    )  # (B, N)

    # The groundtruth coordinates of the object
    coords_true = u_keypoint.get_coords_from_offsets(groundtruth["offset_mask"])  # (B, 2)
    object_in_image = tf.math.reduce_any(
        tf.cast(groundtruth["object_mask"], tf.bool), axis=[1, 2]
    )  # (B, )

    # The patch_indices with the best combined prediction score
    best_score_index = tf.argmax(combined_predictions, axis=-1)  # (B, )

    # The best prediction of each sample.
    best_predictions = tf.gather(combined_predictions, best_score_index, batch_dims=1)  # (B, )

    # The best box of each sample
    best_boxes = tf.gather(predictions["boxes"], best_score_index, batch_dims=1)  # (B, 4)

    coords_true_normalized = coords_true / [640, 480]  # [B, 2]
    # Is True if the best predicted box is actually on the object.
    valid_boxes = u_keypoint.are_coords_in_patch(coords_true_normalized, best_boxes)  # (B, )
    # print(best_boxes)

    # Is True if the prediction of the best patch is greater-equal the combined threshold
    over_threshold = best_predictions >= (encoder_threshold + classifier_threshold)  # (B, )

    fp = over_threshold & tf.math.logical_not(valid_boxes)  # (B, )
    tp = over_threshold & valid_boxes  # (B, )
    fn = tf.math.logical_not(over_threshold) & object_in_image  # (B, )
    tn = tf.math.logical_not(over_threshold) & tf.logical_not(object_in_image)  # (B, )

    fp_indices = tf.where(fp).numpy()
    fn_indices = tf.where(fn).numpy()

    fp_count = tf.math.count_nonzero(fp).numpy()
    tp_count = tf.math.count_nonzero(tp).numpy()
    fn_count = tf.math.count_nonzero(fn).numpy()
    tn_count = tf.math.count_nonzero(tn).numpy()

    precision = tp_count / (tp_count + fp_count)
    recall = tp_count / (tp_count + fn_count)

    fp_rate = fp_count / (fp_count + tp_count)
    fn_rate = fn_count / (fn_count + tn_count)

    return {
        "confusion_matrix": np.array([[tn_count, fp_count], [fn_count, tp_count]]),
        "precision": precision,
        "recall": recall,
        "fp_indices": fp_indices,
        "fn_indices": fn_indices,
        "fp_rate": fp_rate,
        "fn_rate": fn_rate,
    }
