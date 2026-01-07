import tensorflow as tf

from src.util import keypoint as u_keypoint


class TestAreCoordsInPatch:
    def test_single_patch(self):
        coordinates_normalized = tf.constant([[0.343, 0.89]], tf.float32)
        box = tf.constant([[0.1, 0.24, 0.7, 0.5]], tf.float32)

        expected = tf.constant([False])

        result = u_keypoint.are_coords_in_patch(coordinates_normalized, box, padding=0.0)

        assert tf.reduce_all(result == expected)

    def test_multiple_patches(self):
        coordinates_normalized = tf.constant([[0.8, 0.1], [0.343, 0.89]], tf.float32)
        boxes = tf.constant([[0.01, 0.7, 0.3, 0.9], [0.1, 0.24, 0.7, 0.5]], tf.float32)

        expected = tf.constant([True, False])

        result = u_keypoint.are_coords_in_patch(coordinates_normalized, boxes, padding=0.0)

        assert tf.reduce_all(result == expected)

    def test_coord_not_in_margin_of_patch(self):
        coordinates_normalized = tf.constant([[0.75, 0.75]], tf.float32)
        box = tf.constant([0.5, 0.5, 1.0, 1.0], tf.float32)

        expected = tf.constant([True])

        result = u_keypoint.are_coords_in_patch(coordinates_normalized, box, padding=0.0)

        assert tf.reduce_all(result == expected)

    def test_coord_in_left_margin_of_patch(self):
        coordinates_normalized = tf.constant([[0.6, 0.75]], tf.float32)
        box = tf.constant([0.5, 0.5, 1.0, 1.0], tf.float32)

        expected = tf.constant([False])

        result = u_keypoint.are_coords_in_patch(coordinates_normalized, box, padding=0.20)

        assert tf.reduce_all(result == expected)

    def test_coord_in_right_margin_of_patch(self):
        coordinates_normalized = tf.constant([[0.9, 0.75]], tf.float32)
        box = tf.constant([0.5, 0.5, 1.0, 1.0], tf.float32)

        expected = tf.constant([False])

        result = u_keypoint.are_coords_in_patch(coordinates_normalized, box, padding=0.20)

        assert tf.reduce_all(result == expected)

    def test_coord_in_top_margin_of_patch(self):
        coordinates_normalized = tf.constant([[0.75, 0.6]], tf.float32)
        box = tf.constant([0.5, 0.5, 1.0, 1.0], tf.float32)

        expected = tf.constant([False])

        result = u_keypoint.are_coords_in_patch(coordinates_normalized, box, padding=0.20)

        assert tf.reduce_all(result == expected)

    def test_coord_in_bottom_margin_of_patch(self):
        coordinates_normalized = tf.constant([[0.75, 0.9]], tf.float32)
        box = tf.constant([0.5, 0.5, 1.0, 1.0], tf.float32)

        expected = tf.constant([False])

        result = u_keypoint.are_coords_in_patch(coordinates_normalized, box, padding=0.20)

        assert tf.reduce_all(result == expected)
