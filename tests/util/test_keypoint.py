import tensorflow as tf

from src.util import keypoint as u_keypoint


class TestAreCoordsInPatch:
    def test_single_patch(self):
        coordinates_normalized = tf.constant([[0.343, 0.89]], tf.float32)
        boxes = tf.constant([[0.1, 0.24, 0.7, 0.5]], tf.float32)
        
        expected = tf.constant([False])
        
        result = u_keypoint.are_coords_in_patch(coordinates_normalized, boxes)
        
        assert tf.reduce_all(result == expected)
    def test_multiple_patches(self):
        coordinates_normalized = tf.constant([[0.8, 0.1], [0.343, 0.89]], tf.float32)
        boxes = tf.constant([[0.01, 0.7, 0.3, 0.9], [0.1, 0.24, 0.7, 0.5]], tf.float32)
        
        expected = tf.constant([True, False])
        
        result = u_keypoint.are_coords_in_patch(coordinates_normalized, boxes)
        
        assert tf.reduce_all(result == expected)
