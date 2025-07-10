import numpy as np
import scipy
import tensorflow as tf

from . import camera as u_camera
from . import dataset as u_dataset


class MAE(tf.keras.metrics.Metric):
    def __init__(self, object_name="ball", err_threshold=0.2, name="custom_mae", **kwargs):
        super().__init__(name=name, **kwargs)
        self.err_threshold = err_threshold
        self.object_name = object_name
        self.abs_error = self.add_weight(name="abs_error", initializer="zeros")
        self.num_samples = self.add_weight(name="num_samples", initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):
        threshold = self.get_threshold()

        img_coords_true = u_dataset.get_coords_from_offsets(y_true[f"offsets_{self.object_name}"])
        img_coords_pred = u_dataset.get_coords_from_offsets(y_pred[f"offsets_{self.object_name}"])
        wrld_coords_true = tf.expand_dims(
            u_camera.image_to_world(y_true["camera"], y_true["intrinsics"], img_coords_true), axis=0
        ).numpy()
        wrld_coords_pred = tf.expand_dims(
            u_camera.image_to_world(y_pred["camera"], y_pred["intrinsics"], img_coords_pred), axis=0
        ).numpy()

        # print("Img coords pred: ", img_coords_pred)
        # print("Img coords true: ", img_coords_true)
        # print("wrld coords pred: ", wrld_coords_pred)
        # print("wrld coords true: ", wrld_coords_true)

        match = self.match_keypoints(wrld_coords_pred, wrld_coords_true, e_max=threshold)

        # If there were false predictions add the threshold distance as a penalty for every false prediction
        if match["num_false_predictions"] > 0:
            self.abs_error.assign_add(match["num_false_predictions"] * threshold)
            self.num_samples.assign_add(match["num_false_predictions"])

        for pair in match["matches"]:
            distance = 0

            # If all element is in pair are 0.0, there are no objects in the image
            # do not count them towards the metric
            if (pair == 0).all():
                continue

            # If a valid pair was found
            distance += tf.norm(pair[0] - pair[1])
            self.abs_error.assign_add(distance)
            self.num_samples.assign_add(1.0)

        # if sample_weight is not None:
        #     sample_weight = tf.cast(sample_weight, "float32")
        #     values = tf.multiply(values, sample_weight)

    def get_threshold(self):
        return 0.2

    def result(self):
        print(self.abs_error)
        print(self.num_samples)
        return self.abs_error / self.num_samples

    def reset_state(self):
        # The state of the metric will be reset at the start of each epoch.
        self.abs_error.assign(0.0)
        self.num_samples.assign(0.0)

    def match_keypoints(self, kps, pts, e_max):
        """Matches predicted and labeled points optimally. Points are only matched as
        long as their distance is below or equal to e_max. It does not match two
        points from kps to the same point in pts or vice versa. Correspondingly it can
        happen that some points remain unmatched.

        A matrix is constructed that contains a score for each pair of detected and
        labeled point. The score is 0 for pairs that are at least e_max pixels apart
        and 1 for points with distance 0 (cf. the error metric for point detectors from
        lecture 11a).

        Then scipy.optimize.linear_sum_assignment calculates the assignment that maximizes
        the overall score. Dummy assignments (with score 0) are filtered out before the
        matched points are stacked together to obtain the result.

        :param kps: numpy array of predicted keypoints (Mx2)
        :param pts: numpy array of labeled points (Nx2)
        :param e_max: max pixel distance to consider a match
        :return: numpy array of matches (k, p), where k is from kps and p the matched point from pts (Kx2x2)
        """
        if kps.size == 0 or pts.size == 0:
            return np.zeros((0, 2, 2), dtype=kps.dtype)
        diffs = kps[:, np.newaxis] - pts[np.newaxis]
        score_matrix = np.maximum(1 - np.linalg.norm(diffs, axis=-1) / e_max, 0)
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
