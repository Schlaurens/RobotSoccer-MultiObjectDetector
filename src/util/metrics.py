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
        include_encoder_logits: Whether the predicted probability of the encoder should be considered when getting the patch with the highest probility. Defaults to False

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


def calculate_multiclass_metrics(
    predictions: dict,
    groundtruth: dict,
    classifier_threshold: float,
    encoder_threshold: float = 0.1,
    # pooled: bool = False,
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

    # True for every sample that should be used. False else.
    use_sample = tf.cast(
        tf.reduce_any(tf.cast(groundtruth["loss_mask"], tf.bool), axis=[1, 2]),
        tf.float32,
    )  # (B, )

    predicted_probabilities = predictions["classification"]  # (B, N, num_classes)
    num_classes = tf.shape(predicted_probabilities)[-1]

    best_logits = tf.gather(
        predictions["logits"], predictions["patch_indices"], batch_dims=1
    )  # (B, N)

    combined_threshold_mask = get_thresholding_mask(
        tf.reduce_max(predicted_probabilities, axis=-1),
        classifier_threshold,
        best_logits,
        encoder_threshold,
    )  # (B, N)

    groundtruth_probabilities = tf.one_hot(
        tf.cast(
            dataset_utils.get_groundtruth_class_of_patches(
                predictions, groundtruth, padding=0.2, batch_dims=1
            ),
            tf.int32,
        ),
        num_classes,
        axis=-1,
    )  # (B, N, num_classes)

    tf.assert_equal(tf.shape(predicted_probabilities), tf.shape(groundtruth_probabilities))

    # Filter out the ignored samples.
    combined_threshold_mask_filtered = tf.boolean_mask(
        combined_threshold_mask, use_sample
    )  # (#use_sample, N)
    y_pred_filtered = tf.boolean_mask(
        predicted_probabilities, use_sample
    )  # (#use_samples, N, num_classes)
    y_true_filtered = tf.boolean_mask(
        groundtruth_probabilities, use_sample
    )  # (#use_samples, N, num_classes)

    tf.assert_equal(tf.shape(y_pred_filtered), tf.shape(y_true_filtered))
    tf.assert_equal(tf.shape(combined_threshold_mask_filtered), tf.shape(y_true_filtered)[:-1])

    combined_threshold_mask_flat = tf.reshape(combined_threshold_mask_filtered, [-1])
    y_pred_flat = tf.reshape(y_pred_filtered, (-1, num_classes))  # (B * N, num_classes)
    y_true_flat = tf.reshape(y_true_filtered, (-1, num_classes))  # (B * N, num_classes)

    y_pred_labels = tf.argmax(y_pred_flat, axis=-1)  # (B * N, )
    y_true_labels = tf.argmax(y_true_flat, axis=-1)  # (B * N, )

    # Throw away all samples that are under the threshold.
    # y_pred_thresholded = tf.boolean_mask(y_pred_labels, combined_threshold_mask_flat)
    y_pred_thresholded = tf.where(combined_threshold_mask_flat, y_pred_labels, 0)
    # y_true_labels = tf.boolean_mask(y_true_labels, combined_threshold_mask_flat)

    tf.assert_equal(tf.shape(y_pred_thresholded), tf.shape(y_true_labels))

    # ===== Evaluation Metrics ======

    # The (num_classes, num_classes) confusion matrix
    confusion_matrix = tf.math.confusion_matrix(y_true_labels, y_pred_thresholded, num_classes)

    # Calculate precision and recall for every class.
    precisions = tf.linalg.diag_part(confusion_matrix) / tf.reduce_sum(
        confusion_matrix, axis=0
    )  # (num_classes, )
    recalls = tf.linalg.diag_part(confusion_matrix) / tf.reduce_sum(
        confusion_matrix, axis=1
    )  # (num_classes, )

    total_samples = tf.reduce_sum(confusion_matrix)

    # Calculate fp, tp, fn, fp for every class.

    # tp_count = tf.linalg.diag_part(confusion_matrix)
    # fp_count = tf.reduce_sum(confusion_matrix, axis=0) - tp_count  # (num_classes, )
    # fn_count = tf.reduce_sum(confusion_matrix, axis=1) - fp_count  # (num_classes, )
    # true_negatives = total_samples - (
    #     true_positives + false_positives + false_negatives
    # )  # (num_classes, )

    # fp_rate = fp_count / total_samples
    # fn_rate = fn_count / total_samples

    # The (2, 2, num_classes) confusion matrix
    # confusion_matrices = tf.reshape(
    #     tf.stack([true_positives, false_negatives, false_positives, true_negatives], -1),
    #     (4, num_classes),
    # )

    tp_count_pooled = tf.reduce_sum(tf.linalg.diag_part(confusion_matrix)[1:])
    tn_count_pooled = confusion_matrix[0, 0]
    fn_count_pooled = tf.reduce_sum(tf.experimental.numpy.tril(confusion_matrix, k=-1))
    fp_count_pooled = tf.reduce_sum(tf.experimental.numpy.triu(confusion_matrix, k=1))

    fp_rate_pooled = fp_count_pooled / total_samples
    fn_rate_pooled = fn_count_pooled / total_samples

    # Pooled metrics
    # pooled_confusion_matrix = tf.reshape(tf.reduce_sum(confusion_matrices, axis=0), (2, 2))
    pooled_confusion_matrix = np.array(
        [
            [tp_count_pooled, fn_count_pooled],
            [fp_count_pooled, tn_count_pooled],
        ]
    )

    # Precision = TP / TP + FP
    pooled_precision = pooled_confusion_matrix[0][0] / (
        pooled_confusion_matrix[0][0] + pooled_confusion_matrix[1][0]
    )
    # Recall = TP / TP + FN
    pooled_recall = pooled_confusion_matrix[0][0] / (
        pooled_confusion_matrix[0][0] + pooled_confusion_matrix[0][1]
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


def get_thresholding_mask(
    classifier_preds: tf.Tensor,
    classifier_threshold: float,
    encoder_preds: tf.Tensor = None,
    encoder_threshold: float = None,
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

    classifier_preds_thresholded = classifier_preds >= classifier_threshold  # (...)

    if encoder_preds is not None and encoder_threshold is not None:
        encoder_preds_thresholded = encoder_preds >= encoder_threshold  # (...)
    else:
        encoder_preds_thresholded = tf.ones_like(classifier_preds_thresholded)

    tf.assert_equal(tf.shape(classifier_preds_thresholded), tf.shape(encoder_preds_thresholded))

    combined_thresholds = tf.logical_and(classifier_preds_thresholded, encoder_preds_thresholded)

    return combined_thresholds


def match_keypoints_image(y_pred, y_true, threshold: float, batch_dims: int = 1):
    """
    Matches predicted and labeled points optimally. Points are only matched as
    long as their distance is below or equal to threshold. It does not match two
    points from kps to the same point in pts or vice versa. Correspondingly it can
    happen that some points remain unmatched.

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
        pts = tf.expand_dims(y_true, axis=0)
        kps = tf.expand_dims(y_pred, axis=0)
    else:
        pts = y_true
        kps = y_pred

    number_of_pts = tf.shape(pts)[-2] if len(tf.shape(pts)) > 1 else 0
    number_of_kps = tf.shape(kps)[-2] if len(tf.shape(kps)) > 1 else 0

    if number_of_pts == 0 or number_of_kps == 0:
        return {
            "matches": tf.constant([], shape=(0, 2, 2), dtype=kps.dtype),
            "true_positives": 0,
            "false_negatives": number_of_pts,
            "false_positives": number_of_kps,
        }

    diffs = kps[:, tf.newaxis] - pts[tf.newaxis]
    score_matrix = tf.linalg.norm(diffs, axis=-1) <= threshold

    row_ind, col_ind = scipy.optimize.linear_sum_assignment(score_matrix.numpy(), maximize=True)
    assigned = score_matrix.numpy()[row_ind, col_ind] > 0
    matches = tf.stack(
        [tf.gather(kps, row_ind[assigned]), tf.gather(pts, col_ind[assigned])], axis=1
    )

    num_assigned = tf.reduce_sum(tf.cast(assigned, tf.int32))
    true_positives = num_assigned
    false_positives = number_of_kps - num_assigned
    false_negatives = number_of_pts - num_assigned

    return {
        "matches": matches,
        "true_positives": true_positives,
        "false_negatives": false_negatives,
        "false_positives": false_positives,
    }
def save_predictions(
    predictions: dict[tf.Tensor],
    groundtruth: dict[tf.Tensor],
    object_name: str,
    save_directory: str,
    classifier_threshold: float,
    encoder_threshold: float,
    nms_iou_threshold: float,
    nms_max_output_size: float,
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

    selected_indices = batch_nms(
        predictions["boxes"],
        tf.reduce_max(predictions["classification"], axis=-1),
        nms_max_output_size,
        nms_iou_threshold,
    )

    best_logits = tf.gather(
        predictions["logits"],
        predictions["patch_indices"],
        batch_dims=1,
    )  # (B, N)
    classifier_preds = tf.reduce_max(predictions["classification"], axis=-1)  # (B, N)
    positions = predictions["positions"]  # (B, N, 2)

    tresholding_mask = get_thresholding_mask(
        classifier_preds,
        classifier_threshold,
        best_logits,
        encoder_threshold,
    )  # (B, N)

    intersections_thresholded = tf.reshape(
        tf.where(
            tf.reshape(tresholding_mask, [-1]),
            tf.reshape(tf.argmax(predictions["classification"], axis=-1), [-1]),
            0,
        ),
        tf.shape(tresholding_mask),
    )  # (B, N)

    preds = []
    for idx, name in enumerate(groundtruth["name"]):
        frame_time = groundtruth["frame_time"][idx]

        sample = {
            "name": name.numpy().decode("utf-8"),
            "frame_time": int(frame_time.numpy()),
        }

        intersections_thresholded_supressed = tf.gather(
            intersections_thresholded[idx], selected_indices[idx]
        )
        positions_supressed = tf.gather(positions[idx], selected_indices[idx])
        classifier_preds_supressed = tf.gather(classifier_preds[idx], selected_indices[idx])

        if object_name == u_dataset.CategoryNames.INTERSECTIONS.value:
            l_intersections = intersections_thresholded_supressed == 1
            l_positions = tf.boolean_mask(positions_supressed, l_intersections)
            l_confidence = tf.boolean_mask(classifier_preds_supressed, l_intersections)
            l_tensor = tf.concat([l_positions, tf.expand_dims(l_confidence, axis=-1)], axis=-1)

            t_intersections = intersections_thresholded_supressed == 2
            t_positions = tf.boolean_mask(positions_supressed, t_intersections)
            t_confidence = tf.boolean_mask(classifier_preds_supressed, t_intersections)
            t_tensor = tf.concat([t_positions, tf.expand_dims(t_confidence, axis=-1)], axis=-1)

            x_intersections = intersections_thresholded_supressed == 3
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
            tensor = tf.concat(
                [positions_supressed, tf.expand_dims(classifier_preds_supressed, axis=-1)], axis=-1
            )

            sample[object_name] = coords_tensor_to_dict_list(tensor)
        else:
            raise ValueError("Invalid object_name.")

        preds.append(sample)

    os.makedirs(save_directory, exist_ok=True)
    with open(f"{save_directory}/{object_name}.json", "w") as f:
        json.dump(preds, f, indent=4)
