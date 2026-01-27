import tensorflow as tf

from src.util import metrics as u_metrics


class TestGetThresholdingMask:
    def testBaseCase(self):
        classifier_threshold = 0.8
        encoder_threshold = 0.1

        encoder_preds = tf.constant([[0.1, 0.4, 0.04], [0.9, 1, 0]], tf.float32)  # (2, 3)
        classifier_preds = tf.constant([[0.9, 0.8, 1], [0.3, 0, 0]], tf.float32)  # (2, 3)

        expected = tf.cast(tf.constant([[1, 1, 0], [0, 0, 0]]), tf.bool)
        results = u_metrics.get_thresholding_mask(
            classifier_preds, classifier_threshold, encoder_preds, encoder_threshold
        )

        assert tf.reduce_all(expected == results)

    def testFlatInput(self):
        classifier_threshold = 0.8
        encoder_threshold = 0.1

        encoder_preds = tf.constant([0.1, 0.4, 0.04, 0.9, 1, 0], tf.float32)  # (6)
        classifier_preds = tf.constant([0.9, 0.8, 1, 0.3, 0, 0], tf.float32)  # (6)

        expected = tf.cast(tf.constant([1, 1, 0, 0, 0, 0]), tf.bool)
        results = u_metrics.get_thresholding_mask(
            classifier_preds, classifier_threshold, encoder_preds, encoder_threshold
        )

        assert tf.reduce_all(expected == results)

    def testHighDimensions(self):
        classifier_threshold = 0.8
        encoder_threshold = 0.1

        encoder_preds = tf.constant([[[[[0.1], [0.4], [0.04]]], [[[0.9], [1], [0]]]]], tf.float32)
        classifier_preds = tf.constant([[[[[0.9], [0.8], [1]]], [[[0.3], [0], [0]]]]], tf.float32)

        expected = tf.cast(tf.constant([[[[[1], [1], [0]]], [[[0], [0], [0]]]]]), tf.bool)
        results = u_metrics.get_thresholding_mask(
            classifier_preds, classifier_threshold, encoder_preds, encoder_threshold
        )

        assert tf.reduce_all(expected == results)

    def testNoEncoderThresholds(self):
        classifier_threshold = 0.8

        classifier_preds = tf.constant([[0.9, 0.8, 1], [0.3, 0, 0]], tf.float32)  # (2, 3)

        expected = tf.cast(tf.constant([[1, 1, 1], [0, 0, 0]]), tf.bool)
        results = u_metrics.get_thresholding_mask(classifier_preds, classifier_threshold)

        assert tf.reduce_all(expected == results)


class TestMatchKeypointsImage:
    def testPerfectMatch(self):
        y_pred = tf.constant([[5, 5], [1, 0]], tf.float32)
        y_true = tf.constant([[5, 4], [2, 0]], tf.float32)
        threshold = 1

        expected = tf.constant([[[5, 5], [5, 4]], [[1, 0], [2, 0]]], tf.float32)
        result = u_metrics.match_keypoints_image(y_pred, y_true, threshold)

        assert result["true_positives"] == 2
        assert result["false_negatives"] == 0
        assert result["false_positives"] == 0
        assert tf.reduce_all(result["matches"] == expected)

    def testNoMatch(self):
        y_pred = tf.constant([[5, 5], [1, 0]], tf.float32)
        y_true = tf.constant([[6.1, 5], [70, 0]], tf.float32)
        threshold = 1

        expected = tf.constant([], shape=(0, 2, 2), dtype=tf.float32)
        result = u_metrics.match_keypoints_image(y_pred, y_true, threshold)

        assert result["true_positives"] == 0
        assert result["false_negatives"] == 2
        assert result["false_positives"] == 2
        assert tf.reduce_all(result["matches"] == expected)

    def testShapeMismatch(self):
        y_pred = tf.constant([[5, 5], [1, 0]], tf.float32)
        y_true = tf.constant([[6.1, 5]], tf.float32)
        threshold = 2

        expected = tf.constant([[5, 5], [6.1, 5]], tf.float32)
        result = u_metrics.match_keypoints_image(y_pred, y_true, threshold)

        assert result["true_positives"] == 1
        assert result["false_negatives"] == 0
        assert result["false_positives"] == 1
        assert tf.reduce_all(result["matches"] == expected)

    def testNegativeValues(self):
        y_pred = tf.constant([[-1, -5], [-4, 4]], tf.float32)
        y_true = tf.constant([[-2, -5], [-4, 10], [-10, -10]], tf.float32)
        threshold = 2

        expected = tf.constant([[-1, -5], [-2, -5]], tf.float32)
        result = u_metrics.match_keypoints_image(y_pred, y_true, threshold)

        assert result["true_positives"] == 1
        assert result["false_negatives"] == 2
        assert result["false_positives"] == 1
        assert tf.reduce_all(result["matches"] == expected)

    def testMultipleToOneFalsePositives(self):
        y_pred = tf.constant([[6, 5], [8, 4], [7, 9], [7, 10]], tf.float32)
        y_true = tf.constant([[7, 10]], tf.float32)
        threshold = 1.5

        expected = tf.constant([[7, 9], [7, 10]], tf.float32)
        result = u_metrics.match_keypoints_image(y_pred, y_true, threshold)

        assert result["true_positives"] == 1
        assert result["false_negatives"] == 0
        assert result["false_positives"] == 3
        assert tf.reduce_all(result["matches"] == expected)

    def testMultipleToOneFalseNegatives(self):
        y_pred = tf.constant([[7, 10]], tf.float32)
        y_true = tf.constant([[6, 5], [8, 4], [7, 9], [7, 10]], tf.float32)

        threshold = 1.5

        expected = tf.constant([[7, 10], [7, 9]], tf.float32)
        result = u_metrics.match_keypoints_image(y_pred, y_true, threshold)
        tf.print(result)
        assert result["true_positives"] == 1
        assert result["false_negatives"] == 3
        assert result["false_positives"] == 0
        assert tf.reduce_all(result["matches"] == expected)

    def testNoInput(self):
        y_pred = tf.constant([], tf.float32)
        y_true = tf.constant([], tf.float32)
        threshold = 1.5

        expected = tf.constant([], tf.float32, (0, 2, 2))
        result = u_metrics.match_keypoints_image(y_pred, y_true, threshold)

        assert result["true_positives"] == 0
        assert result["false_negatives"] == 0
        assert result["false_positives"] == 0
        assert tf.reduce_all(result["matches"] == expected)

    def testSingleCoord(self):
        y_pred = tf.constant([3, 4], tf.float32)
        y_true = tf.constant([3, 4], tf.float32)
        threshold = 1.5

        expected = tf.constant(([3,4], [3,4]), tf.float32)
        result = u_metrics.match_keypoints_image(y_pred, y_true, threshold, batch_dims=0)

        assert result["true_positives"] == 1
        assert result["false_negatives"] == 0
        assert result["false_positives"] == 0
        assert tf.reduce_all(result["matches"] == expected)
        
    def testNoTrues(self):
        y_pred = tf.constant([[6, 5], [8, 4], [7, 9], [7, 10]], tf.float32)
        y_true = tf.constant([], tf.float32)
        threshold = 1.5

        expected = tf.constant([], tf.float32)
        result = u_metrics.match_keypoints_image(y_pred, y_true, threshold)

        assert result["true_positives"] == 0
        assert result["false_negatives"] == 0
        assert result["false_positives"] == 4
        # tf.print(result["matches"] == expected)
        # tf.print(expected)
        # tf.print(tf.reduce_all(result["matches"] == expected))
        assert tf.size(result["matches"]) == tf.size(expected)
        
    def testNoPreds(self):
        y_pred = tf.constant([], tf.float32)
        y_true = tf.constant([[6, 5], [8, 4], [7, 9], [7, 10]], tf.float32)
        threshold = 1.5

        expected = tf.constant([], tf.float32)
        result = u_metrics.match_keypoints_image(y_pred, y_true, threshold)

        assert result["true_positives"] == 0
        assert result["false_negatives"] == 4
        assert result["false_positives"] == 0
        assert tf.size(result["matches"]) == tf.size(expected)
