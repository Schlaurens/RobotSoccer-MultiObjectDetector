import tensorflow as tf

from src.util import dataset as u_dataset

dataset_config = u_dataset.DatasetConfig()
dataset_utils = u_dataset.DatasetUtils(dataset_config)


class TestFilterCoordinates:
    def test_single_coordinate(self):
        coordinates = tf.constant([[32, 54]], tf.float32)
        expected = coordinates

        result = dataset_utils.filter_coordinates(coordinates)

        # assert result.shape == expected.shape
        assert tf.reduce_all(result == expected)
    
    def test_unique_coordinates(self):
        # No need for filtering.
        coordinates = tf.constant([[32, 54], [100, 102]], tf.float32)
        expected = coordinates

        result = dataset_utils.filter_coordinates(coordinates)

        # assert result.shape == expected.shape
        assert tf.reduce_all(result == expected)

    def test_duplicate_coordinates(self):
        coordinates = tf.constant([[32, 54], [32, 54]], tf.float32)
        expected = tf.constant([[32, 54]], tf.float32)

        result = dataset_utils.filter_coordinates(coordinates)

        # assert result.shape == expected.shape
        assert tf.reduce_all(result == expected)

    def test_same_cell_both_axes(self):
        # One coordinate pair is on both axes bigger than the other one.
        coordinates = tf.constant([[33, 40], [34, 43]], tf.float32)
        expected = tf.constant([[34, 43]], tf.float32)

        result = dataset_utils.filter_coordinates(coordinates)
        assert tf.reduce_all(result == expected)

    def test_same_cell_bigger_x_axis(self):
        # One coordinate pair is only on the x axis bigger than the other. On the y axis they are equal.
        coordinates = tf.constant([[33, 43], [34, 43]], tf.float32)
        expected = tf.constant([[34, 43]], tf.float32)

        result = dataset_utils.filter_coordinates(coordinates)
        assert tf.reduce_all(result == expected)

    def test_same_cell_bigger_y_axis(self):
        # One coordinate pair is only on the y axis bigger than the other. On the x axis they are equal.
        coordinates = tf.constant([[33, 43], [33, 45]], tf.float32)
        expected = tf.constant([[33, 45]], tf.float32)

        result = dataset_utils.filter_coordinates(coordinates)
        assert tf.reduce_all(result == expected)

    def test_negative_values(self):
        coordinates = tf.constant([[-1, -1], [-1, -1], [35, 30]], tf.float32)
        expected = tf.constant([[-1, -1], [35, 30]], tf.float32)

        result = dataset_utils.filter_coordinates(coordinates)
        assert tf.reduce_all(result == expected)

    def test_close_floating_values(self):
        # floats that are very close to each other
        coordinates = tf.constant(
            [[33, 34], [33.0000002, 34.000003], [67.03125, 78.436558]], tf.float32
        )
        expected = tf.constant([[33.0000002, 34.000003], [67.03125, 78.436558]], tf.float32)

        result = dataset_utils.filter_coordinates(coordinates)
        assert tf.reduce_all(result == expected)

    def test_values_at_cell_edge(self):
        grid_size = dataset_config.cell_dims[0]
        coordinates = tf.constant(
            [
                [grid_size, grid_size],
                [0, 0],
                [0, 0],
                [grid_size, grid_size * 2],
                [grid_size * 2, grid_size],
                [grid_size * 2, grid_size * 2],
            ],
            tf.float32,
        )
        expected = tf.constant(
            [
                [grid_size, grid_size],
                [0, 0],
                [grid_size, grid_size * 2],
                [grid_size * 2, grid_size],
                [grid_size * 2, grid_size * 2],
            ],
            tf.float32,
        )

        result = dataset_utils.filter_coordinates(coordinates)
        assert tf.reduce_all(result == expected)

    def test_close_floating_valuesdd(self):
        # floats that are very close to each other
        coordinates = tf.constant(
            [[10, 10], [33, 33], [32, 34], [300, 324], [310, 321]], tf.float32
        )
        expected = tf.constant([[10, 10], [32, 34], [300, 324]], tf.float32)

        result = dataset_utils.filter_coordinates(coordinates)
        tf.print(result)
        assert tf.reduce_all(result == expected)


class TestCoordsInSameCell:
    def test_zeros(self):
        coords_zero = tf.constant([0, 0], tf.float32)
        assert dataset_utils.are_coords_in_same_cell(coords_zero, coords_zero) == True

    def test_same_cell(self):
        coords_a = tf.constant([32, 32], tf.float32)
        coords_b = tf.constant([35, 35], tf.float32)
        assert dataset_utils.are_coords_in_same_cell(coords_a, coords_b) == True

    def test_same_cell_close_to_edge(self):
        coords_a = tf.constant([300, 324], tf.float32)
        coords_b = tf.constant([310, 321], tf.float32)
        assert dataset_utils.are_coords_in_same_cell(coords_a, coords_b) == True

        coords_c = tf.constant([33, 33], tf.float32)
        coords_d = tf.constant([32, 34], tf.float32)
        assert dataset_utils.are_coords_in_same_cell(coords_c, coords_d) == True

    def test_different_cell(self):
        coords_a = tf.constant([100, 35], tf.float32)
        coords_b = tf.constant([32, 35], tf.float32)
        assert dataset_utils.are_coords_in_same_cell(coords_a, coords_b) == False


class TestGetCoordsFromOffset:
    # Assumes that the generation of offset_masks works!
    def test_single_coordinate_pair(self):
        # Only one object in the sample (all cells point to one cell)
        coordinates = tf.constant([[34.0534, 67.432]], tf.float32)
        offset_mask = dataset_utils._generate_offset_mask(coordinates)

        result = dataset_utils.get_coords_from_offsets(offset_mask)
        assert tf.reduce_all(
            tf.keras.ops.isclose(tf.sort(result, axis=0), tf.sort(coordinates, axis=0))
        )

    def test_multiple_coordinate_pairs(self):
        # Multiple objects in the sample (each in their own cell)
        coordinates = tf.constant(
            [[34.0534, 67.432], [24.7644, 67.954], [340.0534, 500.652]], tf.float32
        )
        offset_mask = dataset_utils._generate_offset_mask(coordinates)

        result = dataset_utils.get_coords_from_offsets(offset_mask)
        assert tf.reduce_all(
            tf.keras.ops.isclose(tf.sort(result, axis=0), tf.sort(coordinates, axis=0))
        )

    def test_multiple_coordinate_pairs_with_negative(self):
        coordinates = tf.constant(
            [[-1, -1], [34.0534, 67.432], [24.7644, 67.954], [340.0534, 500.652]], tf.float32
        )
        offset_mask = dataset_utils._generate_offset_mask(coordinates)

        result = dataset_utils.get_coords_from_offsets(offset_mask)
        assert tf.reduce_all(
            tf.keras.ops.isclose(tf.sort(result, axis=0), tf.sort(coordinates, axis=0))
        )

    def test_no_coordinates(self):
        # The offset_mask of an empty sample points to [-1.0, -1.0]
        coordinates = tf.constant([[]], tf.float32)
        offset_mask = dataset_utils.get_masks(coordinates=coordinates)["offsets"]
        expected = tf.constant([[-1, -1]], tf.float32)

        result = dataset_utils.get_coords_from_offsets(offset_mask)
        assert tf.reduce_all(
            tf.keras.ops.isclose(tf.sort(result, axis=0), tf.sort(expected, axis=0))
        )

    def test_coords_at_cell_edge(self):
        grid_size = dataset_config.cell_dims[0]

        coordinates = tf.constant(
            [
                [grid_size, grid_size],
                [0, 0],
                [0, 0],
                [grid_size, grid_size * 2],
                [grid_size * 2, grid_size],
                [grid_size * 2, grid_size * 2],
            ],
            tf.float32,
        )
        expected = tf.constant(
            [
                [grid_size, grid_size],
                [0, 0],
                [grid_size, grid_size * 2],
                [grid_size * 2, grid_size],
                [grid_size * 2, grid_size * 2],
            ],
            tf.float32,
        )
        offset_mask = dataset_utils._generate_offset_mask(coordinates)

        result = dataset_utils.get_coords_from_offsets(offset_mask)
        # tf.print(result)
        assert tf.reduce_all(
            tf.keras.ops.isclose(tf.sort(result, axis=0), tf.sort(expected, axis=0))
        )
