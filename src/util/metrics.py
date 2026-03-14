import json
import os

import numpy as np
import scipy
import tensorflow as tf

from . import camera as u_camera
from . import dataset as u_dataset
from . import keypoint as u_keypoint

dataset_utils = u_dataset.DatasetUtils(u_dataset.DatasetConfig())


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

        img_coords_true = dataset_utils.get_coords_from_offsets(
            y_true[self.object_name]["offset_mask"]
        )  # (x, y)
        img_coords_pred = dataset_utils.get_coords_from_offsets(
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
    classifier_threshold,
    encoder_threshold,
    padding,
    camera,
    intrinsics,
    max_distance,
    threshold_mode,
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
        include_encoder_logits: Whether the predicted probability of the encoder should be considered when getting the patch with the highest probility. Defaults to False

    Returns:
        dict() containing confusion matrix, precision, recall, indices of false_positives and false_negatives, false_positive rate and false_negative rate
    """

    # The groundtruth coordinates of the object. Assumes there is only ONE instance of the object in the image.
    coords_true = dataset_utils.get_coordinate_mask(groundtruth["offset_mask"])[
        :, 0, 0, :
    ]  # (B, 2)
    coords_true_normalized = coords_true / dataset_utils.config.input_dims[::-1]  # (B, 2)

    object_in_image = tf.reduce_any(tf.cast(groundtruth["object_mask"], tf.bool), [1, 2])  # (B, )

    # coords_true that are not (-1.0, -1.0)
    valid_coords = ~tf.reduce_all(coords_true == -1.0, axis=-1)

    coords_true_distances = tf.linalg.norm(
        u_camera.image_to_world(camera, intrinsics, coords_true), axis=-1, keepdims=True
    )
    # The distances of the coords_true that are not (-1.0, -1.0)
    coords_true_distances_valid = tf.where(
        valid_coords, tf.squeeze(coords_true_distances, axis=-1), np.inf
    )
    # Binary mask of the coords_true that are valid and inside the max_distance threshold.
    coords_true_distance_mask = coords_true_distances_valid <= max_distance

    best_predictions = handle_predictions_binary(
        predictions, encoder_threshold, classifier_threshold, threshold_mode
    )

    # The best box of each sample
    best_box = tf.gather(
        predictions["boxes"], best_predictions["best_candidate_indices"], batch_dims=1
    )  # (B, 4)

    # Is True if the best predicted box is actually on the object.
    is_best_box_valid = u_keypoint.are_coords_in_patch(
        coords_true_normalized, best_box, padding
    )  # (B, )

    fp = best_predictions["valid_samples"] & (
        (~is_best_box_valid & coords_true_distance_mask) | ~object_in_image
    )  # (B, )
    tp = (
        best_predictions["valid_samples"]
        & is_best_box_valid
        & coords_true_distance_mask
        & object_in_image
    )  # (B, )
    fn = ~best_predictions["valid_samples"] & coords_true_distance_mask & object_in_image  # (B, )
    tn = ~best_predictions["valid_samples"] & (
        ~object_in_image | ~coords_true_distance_mask
    )  # (B, )

    fp_count = tf.math.count_nonzero(fp).numpy()
    tp_count = tf.math.count_nonzero(tp).numpy()
    fn_count = tf.math.count_nonzero(fn).numpy()
    tn_count = tf.math.count_nonzero(tn).numpy()

    confusion_matrix = np.array([[tp_count, fn_count], [fp_count, tn_count]])

    precision = tp_count / (tp_count + fp_count) if (tp_count + fp_count) > 0 else 1.0
    recall = tp_count / (tp_count + fn_count) if (tp_count + fn_count) > 0 else 0.0

    fp_rate = fp_count / np.sum(confusion_matrix)
    fn_rate = fn_count / np.sum(confusion_matrix)

    return {
        "confusion_matrix": confusion_matrix,
        "precision": precision,
        "recall": recall,
        "fp_rate": fp_rate,
        "fn_rate": fn_rate,
    }


def calculate_multiclass_metrics(
    predictions: dict,
    groundtruth: dict,
    classifier_threshold: float,
    encoder_threshold: float,
    camera,
    intrinsics,
    max_distance,
    iou_threshold: float = None,
):
    """
    Calculate metrics for multi-class predictions.

    A prediction counts as positive for a class if:
        - The combined prediction (encoder + classifier) for that class is >= combined threshold
        - The predicted patch coordinates could be projected on the ground
        - The predicted patch is actually over the object.

    Args:
        predictions: The model predictions (logits and classification scores for each class).
        groundtruth: The corresponding groundtruth data (classification_mask to be converted to one-hot).        classifier_threshold: The threshold of the classifier.
        encoder_threshold: The threshold of the encoder. Defaults to 0.1.
        pooled: Whether the metrics should be calculated for each class or whether they should be pooled together. Defaults to False.

    Returns:
        dict containing per-class confusion matrices, precision, recall, and error indices or the pooled metrics.
    """

    predicted_probabilities = predictions["classification"]  # (B, N, num_classes)
    num_classes = tf.shape(predicted_probabilities)[-1]
    num_candidates = tf.shape(predicted_probabilities)[1]

    processed_predictions = handle_predictions_multiclass(
        predictions, encoder_threshold, classifier_threshold, iou_threshold
    )
    # y_true labels of extracted patches
    y_true_labels = tf.cast(
        dataset_utils.get_groundtruth_class_of_patches(
            predictions, groundtruth, padding=0.2, batch_dims=1
        ),
        tf.int32,
    )  # (B, N)
    y_pred_labels = processed_predictions["classes_of_candidates"]  # (B, N)
    tf.assert_equal(tf.shape(y_true_labels), tf.shape(y_pred_labels))

    # True for every sample that should be used. False else.
    use_sample = tf.reduce_any(tf.cast(groundtruth["loss_mask"], tf.bool), axis=[1, 2])  # (B, )
    use_sample_tiled = tf.reshape(
        tf.tile(use_sample[:, None], [1, num_candidates]), [-1]
    )  # (B * N)

    coord_mask = tf.reshape(
        dataset_utils.get_coordinate_mask(groundtruth["offset_mask"]),
        (-1, tf.reduce_prod(dataset_utils.config.output_dims), 2),
    )  # (B, H_out * W_out, 2)

    # Groundtruth coords of objects in predicted patches
    coords_true = tf.gather(coord_mask, predictions["patch_indices"], batch_dims=1)  # (B, N, 2)

    # ==== Handle Non-Maximum-Suppression ====
    if iou_threshold is not None:
        # Binary mask that is True where the selected indices by the nms are NOT padded.
        nms_sequence_mask = tf.reshape(
            tf.sequence_mask(processed_predictions["nms_num_valid"], maxlen=num_candidates), [-1]
        )  # (B * N)

        y_true_out = tf.reshape(
            tf.gather(y_true_labels, processed_predictions["nms_selected_indices"], batch_dims=1),
            [-1],
        )  # (B * N)
        y_pred_out = tf.reshape(
            tf.gather(
                y_pred_labels,
                processed_predictions["nms_selected_indices"],
                batch_dims=1,
            ),
            [-1],
        )  # (B * N)

    else:
        # No NMS — use all candidates, sequence mask is all True
        nms_sequence_mask = tf.ones(tf.shape(use_sample_tiled), dtype=tf.bool)  # (B * N)
        y_true_out = tf.reshape(y_true_labels, [-1])  # (B * N)
        y_pred_out = tf.reshape(y_pred_labels, [-1])  # (B * N)

    # =============================
    # = Handle Distance Filtering =
    # =============================
    # TODO: This assumes that the object_height is 0.0, which is currently true for all multiclass categories.
    distance_mask = tf.reshape(
        dataset_utils.get_distance_mask_from_offsets(
            groundtruth["offset_mask"], camera, intrinsics, 0.0
        ),
        (-1, tf.reduce_prod(dataset_utils.config.output_dims)),
    )  # (B, H_out, W_out)
    distances_of_coords_true = tf.gather(
        distance_mask, predictions["patch_indices"], batch_dims=1
    )  # (B, N)
    coords_true_distance_mask = tf.reshape(distances_of_coords_true <= max_distance, [-1])  # (B, N)

    y_true_labels_filtered = tf.boolean_mask(
        y_true_out,
        use_sample_tiled & nms_sequence_mask & coords_true_distance_mask,
    )
    y_pred_labels_filtered = tf.boolean_mask(
        y_pred_out,
        use_sample_tiled & nms_sequence_mask & coords_true_distance_mask,
    )

    # ================================================================
    # = Calculate the Objects that are not covered by any candidates =
    # ================================================================

    # Encode the 2D coordinate into a 1D key that is unique for every possible coordinate up 100000.
    scale = 1e5
    invalid_key = -1.0 * scale + (-1.0)  # (-1.0, -1.0) is marked as invalid and later filtered out.

    mask_gt_coords = use_sample[:, tf.newaxis] & (distance_mask <= max_distance)
    # use_sample and distance filtering
    coord_mask_filtered = tf.where(
        mask_gt_coords[..., None], coord_mask, tf.fill([2], -1.0, tf.float32)
    )

    # round the coordinates to the 4th decimal to account for rounding error that occured when calculate the coordinate mask.
    coord_mask_rounded = tf.round(coord_mask_filtered * 1e4) / 1e4  # (B, H_out * W_out, 2)
    keys_gt = tf.sort(
        coord_mask_rounded[:, :, 0] * scale + coord_mask_rounded[:, :, 1], axis=-1
    )  # (B, H_out * W_out)

    mask_coords_true = use_sample[:, tf.newaxis] & (distances_of_coords_true <= max_distance)
    coords_true_filtered = tf.where(
        mask_coords_true[..., None], coords_true, tf.fill([2], -1.0, tf.float32)
    )
    coords_true_rounded = tf.round(coords_true_filtered * 1e4) / 1e4  # (B, N, 2)
    keys_covered = tf.sort(
        coords_true_rounded[:, :, 0] * scale + coords_true_rounded[:, :, 1], axis=-1
    )  # (B, N)

    num_of_covered_coords = tf.reduce_sum(count_unique(keys_covered, invalid_key))  # ( )
    num_of_gt_coords = tf.reduce_sum(count_unique(keys_gt, invalid_key))  # ( )

    num_of_uncovered_gt_coords = num_of_gt_coords - num_of_covered_coords  # ( )

    # ======================
    # = Evaluation Metrics =
    # ======================
    # The (num_classes, num_classes) confusion matrix
    confusion_matrix = tf.math.confusion_matrix(
        y_true_labels_filtered, y_pred_labels_filtered, num_classes
    )

    # Calculate precision and recall for every class.
    precisions = tf.where(
        tf.reduce_sum(confusion_matrix, axis=0) == 0,
        tf.ones_like(
            tf.linalg.diag_part(confusion_matrix), dtype=tf.float32
        ),  # 1.0 when no predictions made
        tf.cast(tf.linalg.diag_part(confusion_matrix), tf.float32)
        / tf.cast(tf.reduce_sum(confusion_matrix, axis=0), tf.float32),
    )  # (num_classes, )
    recalls = tf.math.divide_no_nan(
        tf.linalg.diag_part(confusion_matrix), tf.reduce_sum(confusion_matrix, axis=1)
    )  # (num_classes, )

    # =========================================
    # = Pooled metrics (class 0 = background) =
    # =========================================
    tp_count_pooled = tf.reduce_sum(tf.linalg.diag_part(confusion_matrix)[1:])
    tn_count_pooled = confusion_matrix[0, 0]
    fp_count_pooled = tf.reduce_sum(confusion_matrix[0, 1:])
    # Add the number of gt_coords that were not covered by any candidates to fn and total_samples
    fn_count_pooled = tf.reduce_sum(confusion_matrix[1:, 0]).numpy() + num_of_uncovered_gt_coords
    total_samples = tf.reduce_sum(confusion_matrix).numpy() + num_of_uncovered_gt_coords

    fp_rate_pooled = fp_count_pooled / total_samples
    fn_rate_pooled = fn_count_pooled / total_samples

    # print(
    #     f"Expected ceiling: {1 - num_of_uncovered_gt_coords / tf.reduce_sum(num_of_gt_coords).numpy():.3f}"
    # )

    # Pooled metrics
    pooled_confusion_matrix = np.array(
        [
            [tp_count_pooled, fn_count_pooled],
            [fp_count_pooled, tn_count_pooled],
        ]
    )

    # Precision = TP / TP + FP
    pooled_precision = (
        pooled_confusion_matrix[0][0]
        / (pooled_confusion_matrix[0][0] + pooled_confusion_matrix[1][0])
        if pooled_confusion_matrix[0][0] + pooled_confusion_matrix[1][0] > 0
        else 1.0
    )
    # Recall = TP / TP + FN
    pooled_recall = (
        pooled_confusion_matrix[0][0]
        / (pooled_confusion_matrix[0][0] + pooled_confusion_matrix[0][1])
        if pooled_confusion_matrix[0][0] + pooled_confusion_matrix[0][1] > 0
        else 0.0
    )

    return {
        "confusion_matrix": confusion_matrix.numpy(),
        "precisions": precisions,
        "recalls": recalls,
        "confusion_matrix_pooled": pooled_confusion_matrix,
        "precision_pooled": pooled_precision,
        "recall_pooled": pooled_recall,
        "fp_rate": fp_rate_pooled,
        "fn_rate": fn_rate_pooled,
    }


def count_unique(x: tf.Tensor, invalid_val: float = None) -> tf.Tensor:
    """Counts the number of unique values per row in a 2D tensor.

    Unique counting is done by sorting each row and counting adjacent differences.
    If an invalid value is specified, those entries are excluded from the count.

    Args:
        x: A 2D tensor of shape (B, N) containing the values to count unique
            entries for. Each row is counted independently.
        invalid_val: A float value to exclude from the unique count, e.g. a
            sentinel value for padding or missing entries. If None, all values
            are counted. Defaults to None.

    Returns:
        A 1D tensor of shape (B,) containing the number of unique valid values
        per row.

    Example:
        >>> x = tf.constant([[1.0, 2.0, 2.0, 3.0], [1.0, -1.0, -1.0, 1.0]])
        >>> count_unique(x, invalid_val=-1.0)
        <tf.Tensor: shape=(2,), dtype=int32, numpy=array([3, 1])>
    """
    if invalid_val is not None:
        valid_mask = x != invalid_val  # (B, N)
        sentinel = tf.reduce_max(x) + 1
        x = tf.where(valid_mask, x, sentinel)

    sorted_x = tf.sort(x, axis=-1)
    diff = sorted_x[:, 1:] != sorted_x[:, :-1]

    if invalid_val is not None:
        valid_sorted = tf.sort(
            tf.cast(valid_mask, tf.int32), axis=-1, direction="DESCENDING"
        )  # (B, N)
        diff = tf.cast(valid_sorted[:, 1:], tf.bool) & diff
        any_valid = tf.reduce_any(valid_mask, axis=-1)  # (B,)
        counts = tf.reduce_sum(tf.cast(diff, tf.int32), axis=-1) + tf.cast(any_valid, tf.int32)
    else:
        counts = tf.reduce_sum(tf.cast(diff, tf.int32), axis=-1) + 1

    return counts


def calculate_metrics(
    predictions: dict,
    groundtruth: dict,
    num_classes: int,
    classifier_threshold: float,
    encoder_threshold: float,
    treshold_mode: str,
    camera,
    intrinsics,
    max_distance,
    padding: float = None,
    iou_threshold: float = None,
):
    if num_classes > 1:
        return calculate_multiclass_metrics(
            predictions,
            groundtruth,
            classifier_threshold,
            encoder_threshold,
            camera,
            intrinsics,
            max_distance,
            iou_threshold,
        )

    else:
        binary_metrics = calculate_binary_metrics(
            predictions,
            groundtruth,
            classifier_threshold,
            encoder_threshold,
            padding,
            camera,
            intrinsics,
            max_distance,
            treshold_mode,
        )
        return {
            "confusion_matrix": binary_metrics["confusion_matrix"],
            "precisions": np.array([binary_metrics["precision"], binary_metrics["precision"]]),
            "recalls": np.array([binary_metrics["recall"], binary_metrics["recall"]]),
            "confusion_matrix_pooled": binary_metrics["confusion_matrix"],
            "precision_pooled": binary_metrics["precision"],
            "recall_pooled": binary_metrics["recall"],
            "fp_rate": binary_metrics["fp_rate"],
            "fn_rate": binary_metrics["fn_rate"],
        }


def get_thresholding_mask(
    classifier_preds: tf.Tensor,
    classifier_threshold: float,
    encoder_preds: tf.Tensor = None,
    encoder_threshold: float = None,
    mode: str = "additive",
):
    """Generates a binary mask that is True everwhere the prediction is within the specified theshold. This function assumes that `classifier_preds` and `encoder_preds` are of the same shape. `encoder_preds` and `encoder_threshold` are optional.

    Args:
        classifier_preds: The tf.Tensor of the classifier preditions.
        classifier_threshold: The classifier threshold.
        encoder_preds: The tf.Tensor containing the encoder thresholds. Defaults to None.
        encoder_threshold: The encoder threshold. Default to None.

    Returns:
    The thresholding mask.
    """
    if mode == "additive":
        if encoder_preds is not None:
            combined_score = classifier_preds + encoder_preds
            return combined_score >= (classifier_threshold + encoder_threshold)
        else:
            return classifier_preds >= classifier_threshold

    elif mode == "logical_and":
        classifier_preds_thresholded = classifier_preds >= classifier_threshold  # (...)

        if encoder_preds is not None and encoder_threshold is not None:
            encoder_preds_thresholded = encoder_preds >= encoder_threshold  # (...)
        else:
            encoder_preds_thresholded = tf.ones_like(classifier_preds_thresholded)

        tf.assert_equal(tf.shape(classifier_preds_thresholded), tf.shape(encoder_preds_thresholded))

        combined_thresholds = tf.logical_and(
            classifier_preds_thresholded, encoder_preds_thresholded
        )

        return combined_thresholds


def match_keypoints_image(y_pred, y_true, threshold: float, batch_dims: int = 1):
    """
    Matches predicted and labeled points optimally. Points are only matched as
    long as their distance is below or equal to threshold. It does not match two
    points from y_pred to the same point in y_true or vice versa. Correspondingly it can
    happen that some points remain unmatched. This is way it is recommended to filter out
    any coordinates that can be considered the same. Otherwise this might ruin the fn/fp
    metric.

    A matrix is constructed that contains a score for each pair of detected and
    labeled point. The score is 0 for pairs that are at least threshold pixels apart
    and 1 for points with distance 0.

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

    if batch_dims == 0:
        y_true = tf.expand_dims(y_true, axis=0)
        y_pred = tf.expand_dims(y_pred, axis=0)
    else:
        y_true = y_true
        y_pred = y_pred

    number_of_pts = tf.shape(y_true)[-2] if len(tf.shape(y_true)) > 1 else 0
    number_of_kps = tf.shape(y_pred)[-2] if len(tf.shape(y_pred)) > 1 else 0

    if number_of_pts == 0 or number_of_kps == 0:
        return {
            "matches": tf.constant([], shape=(0, 2, 2), dtype=y_pred.dtype),
            "true_positives": 0,
            "false_negatives": number_of_pts,
            "false_positives": number_of_kps,
            "fn_tensor": y_true,
            "fp_tensor": y_pred,
        }

    diffs = y_pred[:, tf.newaxis] - y_true[tf.newaxis]
    score_matrix = tf.linalg.norm(diffs, axis=-1) <= threshold

    row_ind, col_ind = scipy.optimize.linear_sum_assignment(score_matrix.numpy(), maximize=True)
    assigned = score_matrix.numpy()[row_ind, col_ind] > 0
    matches = tf.stack(
        [tf.gather(y_pred, row_ind[assigned]), tf.gather(y_true, col_ind[assigned])], axis=1
    )

    # Get the coords from y_true/y_pred that have been matched.
    y_true_in_matches = matches[:, 1]
    y_pred_in_matches = matches[:, 0]

    # Compare each element of y_true with each element of matched y_true coords.
    fn_equal_elements = tf.equal(y_true[:, tf.newaxis, :], y_true_in_matches)

    # Check if all elements in each row of y_true match any row in matched_y_true
    fn_equal_rows = tf.reduce_all(fn_equal_elements, axis=2)

    # Check if any row in y_true matches any row in matched_y_true
    fn_tensor_mask = tf.reduce_any(fn_equal_rows, axis=1)

    # Select the correct elements from y_true
    fn_tensor = tf.boolean_mask(y_true, tf.logical_not(fn_tensor_mask))

    # Same procedure for y_pred...
    fp_equal_elements = tf.equal(y_pred[:, tf.newaxis, :], y_pred_in_matches)
    fp_equal_rows = tf.reduce_all(fp_equal_elements, axis=2)
    fp_tensor_mask = tf.reduce_any(fp_equal_rows, axis=1)
    fp_tensor = tf.boolean_mask(y_pred, tf.logical_not(fp_tensor_mask))

    num_assigned = tf.reduce_sum(tf.cast(assigned, tf.int32))
    true_positives = num_assigned
    false_positives = number_of_kps - num_assigned
    false_negatives = number_of_pts - num_assigned

    return {
        "matches": matches,
        "true_positives": true_positives,
        "false_negatives": false_negatives,
        "false_positives": false_positives,
        "fn_tensor": fn_tensor,
        "fp_tensor": fp_tensor,
    }


def batch_nms(
    boxes: tf.Tensor,
    scores: tf.Tensor,
    max_output_size_per_class: int = 7,
    iou_threshold: float = 0.35,
    score_threshold: float = float("-inf"),
):
    """
    Apply NMS to each batch element.

    Args:
        boxes: Tensor of shape (B, N, 4)
        scores: Tensor of shape (B, N)
        max_output_size_per_class: Maximum number of boxes to keep per batch element
        iou_threshold: IoU threshold for NMS
        score_threshold: Minimum score threshold for keeping a box

    Returns:
        selected_indices: List of tensors, each containing the indices of selected boxes for a batch element
    """
    selected_indices = []
    for i in range(boxes.shape[0]):
        # Extract boxes and scores for the i-th batch element
        boxes_i = boxes[i]
        scores_i = scores[i]

        # Apply NMS
        selected = tf.image.non_max_suppression(
            boxes_i,
            scores_i,
            max_output_size_per_class,
            iou_threshold=iou_threshold,
            score_threshold=score_threshold,
        )
        selected_indices.append(selected)
    return selected_indices


def save_predictions(
    predictions: dict[tf.Tensor],
    groundtruth: dict[tf.Tensor],
    object_name: str,
    save_directory: str,
    classifier_threshold: float,
    encoder_threshold: float,
    nms_iou_threshold: float,
) -> None:
    """Saves the predictions for each object category into a .json file.

    Args:
        predictions: The predictions for the given object
        groundtruth: The groundtruth for the given object
        object_name: The object name
        save_directory: The directory where the .json file should be saved to
        classifier_threshold: The classifier threshold
        encoder_threshold: The encoder_threshold
        nms_iou_threshold: The IoU Threshold used for the non-maximum-suppression
        nms_max_output_size: The max output size for the non-maximum-suppression. The nms used here is padded.
    """

    def coords_tensor_to_dict_list(tensor):
        return [
            {"x": float(x), "y": float(y), "confidence": float(conf)}
            for x, y, conf in tensor.numpy()
        ]

    positions = predictions["positions"]  # (B, N, 2)
    classifier_preds = tf.reduce_max(predictions["classification"], axis=-1)  # (B, N)

    processed_predictions = handle_predictions(
        predictions, encoder_threshold, classifier_threshold, nms_iou_threshold
    )

    preds = []
    for idx, name in enumerate(groundtruth["name"]):
        frame_time = groundtruth["frame_time"][idx]

        sample = {
            "name": name.numpy().decode("utf-8"),
            "frame_time": int(frame_time.numpy()),
        }

        if object_name == u_dataset.CategoryNames.INTERSECTIONS.value:
            nms_valid_indices = tf.slice(
                processed_predictions["nms_selected_indices"][idx],
                tf.constant([0]),
                processed_predictions["nms_num_valid"][idx, tf.newaxis],
            )

            intersections_thresholded_suppressed = tf.gather(
                processed_predictions["classes_of_candidates"][idx],
                nms_valid_indices,
            )
            positions_supressed = tf.gather(positions[idx], nms_valid_indices)
            classifier_preds_supressed = tf.gather(classifier_preds[idx], nms_valid_indices)

            l_intersections = intersections_thresholded_suppressed == 1
            l_positions = tf.boolean_mask(positions_supressed, l_intersections)
            l_confidence = tf.boolean_mask(classifier_preds_supressed, l_intersections)
            l_tensor = tf.concat([l_positions, tf.expand_dims(l_confidence, axis=-1)], axis=-1)

            t_intersections = intersections_thresholded_suppressed == 2
            t_positions = tf.boolean_mask(positions_supressed, t_intersections)
            t_confidence = tf.boolean_mask(classifier_preds_supressed, t_intersections)
            t_tensor = tf.concat([t_positions, tf.expand_dims(t_confidence, axis=-1)], axis=-1)

            x_intersections = intersections_thresholded_suppressed == 3
            x_positions = tf.boolean_mask(positions_supressed, x_intersections)
            x_confidence = tf.boolean_mask(classifier_preds_supressed, x_intersections)
            x_tensor = tf.concat([x_positions, tf.expand_dims(x_confidence, axis=-1)], axis=-1)

            sample[object_name] = {
                "L": coords_tensor_to_dict_list(l_tensor),
                "T": coords_tensor_to_dict_list(t_tensor),
                "X": coords_tensor_to_dict_list(x_tensor),
            }

        elif object_name in [
            u_dataset.CategoryNames.BALL.value,
            u_dataset.CategoryNames.PENALTYMARK.value,
        ]:
            if not processed_predictions["valid_samples"][idx]:
                sample[object_name] = []
                preds.append(sample)
                continue

            best_position = tf.gather(
                positions[idx], processed_predictions["best_candidate_indices"][idx]
            )  # (1, 2)
            best_classifier_preds = processed_predictions["classifier_confidences"][
                idx
            ]  # Shape: ( )

            tensor = tf.concat(
                [best_position, tf.expand_dims(best_classifier_preds, axis=-1)], axis=-1
            )

            sample[object_name] = coords_tensor_to_dict_list(tf.expand_dims(tensor, 0))
        else:
            raise ValueError("Invalid object_name.")

        preds.append(sample)

    os.makedirs(save_directory, exist_ok=True)
    with open(f"{save_directory}/{object_name}.json", "w") as f:
        json.dump(preds, f, indent=4)


def handle_predictions_binary(
    predictions: dict,
    encoder_threshold: float,
    classifier_threshold: float,
    threshold_mode: str = "logical_and",
) -> dict:
    """Processes binary predictions by applying thresholding to filter and classify candidates.

    This function takes raw predictions from a model and applies thresholding based on classifier and encoder
    confidence scores. It returns information about valid samples, the indices of the best candidates, and their
    corresponding confidence scores.

    Args:
        predictions (dict): A dictionary containing model predictions with the following keys:
            - "classification": Tensor of shape (B, N) containing classification scores for each candidate.
            - "logits": Tensor containing logits for each candidate.
            - "patch_indices": Tensor containing indices of patches corresponding to each candidate.
        encoder_threshold (float): Threshold for the encoder's confidence scores. Candidates with scores below this threshold are filtered out.
        classifier_threshold (float): Threshold for the classifier's confidence scores. Candidates with scores below this threshold are filtered out.

    Returns:
        dict: A dictionary containing processed prediction information with the following keys:
            - "valid_samples": Tensor of shape (B,) indicating whether each sample contains at least one valid candidate.
            - "threshold_mask": Tensor of shape (B,) indicating whether the best candidate for each sample passed the thresholding criteria.
            - "best_candidate_indices": Tensor of shape (B,) containing the indices of the best candidates.
            - "encoder_confidences": Tensor of shape (B,) containing the encoder confidence scores for the best candidates.
            - "classifier_confidences": Tensor of shape (B,) containing the classifier confidence scores for the best candidates.
    """
    best_logits = tf.gather(
        predictions["logits"], predictions["patch_indices"], batch_dims=1
    )  # (B, N)
    classification_scores = tf.squeeze(predictions["classification"], axis=-1)  # (B, N)

    # Candidates that pass the threshold(s)
    combined_threshold_mask = get_thresholding_mask(
        classification_scores, classifier_threshold, best_logits, encoder_threshold, threshold_mode
    )  # (B, N)

    # If the classification_scores are invalid because of the threshold, they are tf.float32.min !
    masked_scores = tf.where(
        combined_threshold_mask, classification_scores, tf.float32.min
    )  # (B, N)

    # True if the sample contains a valid candidate, else False.
    valid_samples = ~tf.reduce_all(masked_scores == tf.float32.min, axis=-1)  # (B, )

    # The indices of the candidate with the highest confidence and where the candidate is inside the threshold.
    best_score_index = tf.argmax(
        masked_scores,
        axis=-1,
    )  # (B, )

    threshold_mask = tf.gather(combined_threshold_mask, best_score_index, batch_dims=1)  # (B, )
    encoder_confidences = tf.gather(best_logits, best_score_index, batch_dims=1)  # (B, )
    classifier_confidences = tf.gather(
        classification_scores, best_score_index, batch_dims=1
    )  # (B, )

    return {
        "valid_samples": valid_samples,
        "threshold_mask": threshold_mask,
        "best_candidate_indices": best_score_index,
        "encoder_confidences": encoder_confidences,
        "classifier_confidences": classifier_confidences,
    }


def handle_predictions_multiclass(
    predictions: dict,
    encoder_threshold: float,
    classifier_threshold: float,
    iou_threshold: float = None,
) -> dict:
    """Processes multiclass predictions by applying thresholding and non-maximum suppression to filter and classify candidates.

    This function takes raw predictions from a model and applies thresholding based on classifier and encoder
    confidence scores. If specified, it also applies non-maximum suppression to limit the number of output candidates
    per batch based on their intersection-over-union (IoU) overlap and confidence scores.

    Args:
        predictions (dict): A dictionary containing model predictions with the following keys:
            - "classification": Tensor of shape (B, N, num_classes) containing classification logits for each candidate.
            - "boxes": Tensor of shape (B, N, 4) containing bounding box coordinates for each candidate.
            - "logits": Tensor containing encoder logits for each candidate.
            - "patch_indices": Tensor containing encoder indices of patches corresponding to each candidate.
        encoder_threshold (float): Threshold for the encoder's confidence scores. Candidates with scores below this threshold are filtered out.
        classifier_threshold (float): Threshold for the classifier's confidence scores. Candidates with scores below this threshold are filtered out.
        iou_threshold (float, optional): Intersection-over-union (IoU) threshold for non-maximum suppression. Candidates with an IoU overlap greater than this threshold are suppressed. If None, non-maximum suppression is not applied.

    Returns:
        dict: A dictionary containing processed prediction information with the following keys:
            - "classes_of_candidates": Tensor of shape (B, N) containing the predicted class labels for each candidate, with candidates below the threshold classified as 0 (negative class).
            - "threshold_mask": Tensor of shape (B, N) indicating which candidates passed the thresholding criteria.
            - "nms_selected_indices": Tensor of shape (B, N) containing the indices of candidates selected by non-maximum suppression, or None if non-maximum suppression was not applied.
            - "nms_num_valid": Tensor of shape (B,) containing the number of valid candidates per batch after non-maximum suppression, or None if non-maximum suppression was not applied.
    """
    classification_scores = predictions["classification"]  # (B, N, num_classes)
    num_candidates = tf.shape(classification_scores)[1]

    nms_selected_indices = None
    nms_num_valid = None

    if iou_threshold is not None:
        nms_selected_indices, nms_num_valid = tf.image.non_max_suppression_padded(
            predictions["boxes"],  # (B, N, 4)
            tf.reduce_max(classification_scores, axis=-1),  # (B, N)
            num_candidates,
            iou_threshold,
            pad_to_max_output_size=True,
        )  # (B, N), (B, )

    best_logits = tf.gather(
        predictions["logits"], predictions["patch_indices"], batch_dims=1
    )  # (B, N)

    # Candidates that pass the threshold(s)
    combined_threshold_mask = get_thresholding_mask(
        tf.reduce_max(classification_scores, axis=-1),
        classifier_threshold,
        best_logits,
        encoder_threshold,
    )  # (B, N)

    y_pred_labels = tf.argmax(classification_scores, axis=-1)  # (B, N)

    # Classify all samples that are under the threshold as 0 (negative class).
    masked_scores = tf.where(combined_threshold_mask, y_pred_labels, 0)  # (B, N)

    return {
        "classes_of_candidates": masked_scores,
        "threshold_mask": combined_threshold_mask,
        "nms_selected_indices": nms_selected_indices,
        "nms_num_valid": nms_num_valid,
    }


def handle_predictions(
    predictions: dict, encoder_threshold: float, classifier_threshold: float, iou_threshold: float
):
    """A wrapper function that processes predictions based on the number of classes in the classification tensor.

    This function acts as a dispatcher, directing the handling of predictions to either a multiclass or binary classification
    handler based on the number of classes detected in the predictions. It ensures that the appropriate
    processing is applied according to the model's output structure.

    Args:
        predictions (dict): A dictionary containing model predictions
        encoder_threshold (float): Threshold for the encoder's confidence scores.
            Candidates with scores below this threshold are filtered out.
        classifier_threshold (float): Threshold for the classifier's confidence scores.
            Candidates with scores below this threshold are filtered out.
        iou_threshold (float): Intersection-over-union (IoU) threshold for non-maximum suppression.
            Used only in multiclass predictions. Candidates with an IoU overlap greater than this
            threshold are suppressed.

    Raises:
        ValueError: If the number of classes in the classification tensor is not 1 or greater than 1.
            This indicates an unexpected shape of the classification tensor.

    Returns:
        dict: Processed prediction information. The structure of the returned dictionary depends on
            whether the predictions are multiclass or binary:
            - For multiclass predictions, refer to the return value of `handle_predictions_multiclass`.
            - For binary predictions, refer to the return value of `handle_predictions_binary`.

    Example:
        For multiclass predictions, the returned dictionary will contain:
            - "classes_of_candidates": Tensor of shape (B, N) containing the predicted class labels.
            - "threshold_mask": Tensor of shape (B, N) indicating which candidates passed the thresholding criteria.
            - "nms_selected_indices": Tensor of shape (B, N) containing the indices of candidates selected by non-maximum suppression.
            - "nms_num_valid": Tensor of shape (B,) containing the number of valid candidates per batch after non-maximum suppression.

        For binary predictions, the returned dictionary will contain:
            - "valid_samples": Tensor indicating whether each sample contains at least one valid candidate.
            - "threshold_mask": Tensor indicating whether the best candidate for each sample passed the thresholding criteria.
            - "best_candidate_indices": Tensor containing the indices of the best candidates.
            - "encoder_confidences": Tensor containing the encoder confidence scores for the best candidates.
            - "classifier_confidences": Tensor containing the classifier confidence scores for the best candidates.
    """
    num_classes = tf.shape(predictions["classification"])[-1]

    if num_classes > 1:
        return handle_predictions_multiclass(
            predictions, encoder_threshold, classifier_threshold, iou_threshold
        )
    elif num_classes == 1:
        return handle_predictions_binary(predictions, encoder_threshold, classifier_threshold)
    else:
        raise ValueError(
            "Unknown number of classes. Classification Tensor in predictions probably has no num_classes dimension."
        )
