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

    def test_multiple_zero_coords(self):
        coordinates = tf.constant([[0, 0], [0, 0], [0, 0]], tf.float32)
        expected = tf.constant([[0, 0]], tf.float32)

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
            tf.keras.ops.isclose(
                tf.sort(result, axis=1), tf.sort(tf.expand_dims(coordinates, axis=0), axis=1)
            )
        )

    def test_multiple_coordinate_pairs(self):
        # Multiple objects in the sample (each in their own cell)
        coordinates = tf.constant(
            [[34.0534, 67.432], [24.7644, 67.954], [340.0534, 500.652]], tf.float32
        )
        offset_mask = dataset_utils._generate_offset_mask(coordinates)

        result = dataset_utils.get_coords_from_offsets(offset_mask)
        assert tf.reduce_all(
            tf.keras.ops.isclose(
                tf.sort(result, axis=1), tf.sort(tf.expand_dims(coordinates, axis=0), axis=1)
            )
        )

    def test_multiple_coordinate_pairs_with_negative(self):
        coordinates = tf.constant(
            [[-1, -1], [34.0534, 67.432], [24.7644, 67.954], [340.0534, 500.652]], tf.float32
        )
        offset_mask = dataset_utils._generate_offset_mask(coordinates)

        result = dataset_utils.get_coords_from_offsets(offset_mask)
        assert tf.reduce_all(
            tf.keras.ops.isclose(
                tf.sort(result, axis=1), tf.sort(tf.expand_dims(coordinates, axis=0), axis=1)
            )
        )

    def test_no_coordinates(self):
        # The offset_mask of an empty sample points to [-1.0, -1.0]
        coordinates = tf.constant([[]], tf.float32)
        offset_mask = dataset_utils.get_masks(coordinates=coordinates)["offsets"]
        expected = tf.constant([[-1, -1]], tf.float32)

        result = dataset_utils.get_coords_from_offsets(offset_mask)
        assert tf.reduce_all(
            tf.keras.ops.isclose(
                tf.sort(result, axis=1), tf.sort(tf.expand_dims(expected, axis=0), axis=1)
            )
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
        assert tf.reduce_all(
            tf.keras.ops.isclose(
                tf.sort(result, axis=1), tf.sort(tf.expand_dims(expected, axis=0), axis=1)
            )
        )

    def test_coords_batched(self):
        coordinates = tf.constant(
            [[34.0534, 67.432], [24.7644, 67.954], [340.0534, 500.652]], tf.float32
        )
        offset_mask = tf.expand_dims(
            dataset_utils._generate_offset_mask(coordinates), axis=0
        )  # (H, W, 2)
        coordinates_empty = tf.constant([[]], tf.float32)
        offset_mask_empty = tf.expand_dims(
            dataset_utils.get_masks(coordinates=coordinates_empty)["offsets"], axis=0
        )

        input_mask = tf.concat(
            [offset_mask, offset_mask_empty, offset_mask], axis=0
        )  # (2, H, W, 2)

        result = dataset_utils.get_coords_from_offsets(input_mask)  # (B, N, 2)

        expected = tf.stack(
            [coordinates, [[-1.0, -1.0], [-1.0, -1.0], [-1.0, -1.0]], coordinates], axis=0
        )  # (B, N, 2)

        assert tf.reduce_all(
            tf.keras.ops.isclose(
                tf.sort(result, axis=1),
                tf.sort(expected, axis=1),
            )
        )


class TestObjectMask:
    def test_basic(self):
        coordinates = tf.constant([[0.5, 0.5], [100.3, 106.06]], tf.float32)
        object_mask = tf.cast(
            dataset_utils.get_masks(coordinates=coordinates)["object_mask"], tf.int32
        )

        mask_indices = tf.cast(coordinates // dataset_config.cell_dims, tf.int32)[
            ..., ::-1
        ]  # (N, 2)
        mask_values = tf.gather_nd(object_mask, mask_indices)

        assert tf.reduce_sum(object_mask) == coordinates.shape[0]
        assert tf.reduce_all(mask_values == 1)

    def test_same_cell(self):
        coordinates = tf.constant([[0.5, 0.5], [24, 27.005]], tf.float32)
        object_mask = tf.cast(
            dataset_utils.get_masks(coordinates=coordinates)["object_mask"], tf.int32
        )

        mask_indices = tf.cast(coordinates // dataset_config.cell_dims, tf.int32)[
            ..., ::-1
        ]  # (N, 2)
        mask_values = tf.gather_nd(object_mask, mask_indices)

        assert tf.reduce_sum(object_mask) == coordinates.shape[0] - 1
        assert tf.reduce_all(mask_values == 1)

    def test_out_of_bounds(self):
        coordinates = tf.constant(
            [
                [dataset_config.input_dims[1] + 10, 70],
                [32, dataset_config.input_dims[0] + 10],
                [-5, 80],
                [50, -5],
            ],
            tf.float32,
        )
        object_mask = tf.cast(
            dataset_utils.get_masks(coordinates=coordinates)["object_mask"], tf.int32
        )

        clipped_coords = tf.clip_by_value(
            coordinates,
            clip_value_min=tf.constant([0.0, 0.0], dtype=coordinates.dtype),
            clip_value_max=tf.constant(
                [dataset_config.input_dims[1] - 1, dataset_config.input_dims[0] - 1],
                dtype=coordinates.dtype,
            ),
        )  # (N, 2)

        mask_indices = tf.cast(clipped_coords // dataset_config.cell_dims, tf.int32)[
            ..., ::-1
        ]  # (N, 2)
        mask_values = tf.gather_nd(object_mask, mask_indices)

        assert tf.reduce_sum(object_mask) == coordinates.shape[0]
        assert tf.reduce_all(mask_values == 1)


class TestGetCellOfCoordinates:
    def test_basic_case(self):
        coords_a = tf.constant([15, 15], tf.float32)
        coords_b = tf.constant([50, 70], tf.float32)

        expected_a = tf.constant([0, 0], tf.int32)
        expected_b = tf.constant([1, 2], tf.int32)

        result_a = dataset_utils.get_cell_of_coordinate(coords_a)
        result_b = dataset_utils.get_cell_of_coordinate(coords_b)

        assert tf.reduce_all(expected_a == result_a)
        assert tf.reduce_all(expected_b == result_b)

    def test_small_coords(self):
        coords = tf.constant([[1, 1], [0.5, 0.5], [-1, -1]], tf.float32)

        expected = tf.constant([[0, 0], [0, 0], [-1, -1]], tf.int32)

        results = dataset_utils.get_cell_of_coordinate(coords, clip=False)
        assert tf.reduce_all(expected == results)

    def test_batched_coords(self):
        coords = tf.constant([[[15, 15], [50, 70]], [[80, 15], [50, 300]]], tf.float32)  # (B, N, 2)
        expected = tf.constant([[[0, 0], [1, 2]], [[2, 0], [1, 9]]], tf.int32)

        results = dataset_utils.get_cell_of_coordinate(coords)
        assert tf.reduce_all(expected == results)

    def test_negative_coords(self):
        coords = tf.constant([[-10, -50]], tf.float32)

        expected = tf.constant([[-1, -2]], tf.int32)

        results = dataset_utils.get_cell_of_coordinate(coords)
        assert tf.reduce_all(expected == results)

    def test_coords_at_cell_edge(self):
        coords = tf.constant([[31, 31], [31, 32]], tf.float32)

        expected = tf.constant([[0, 0], [0, 1]], tf.int32)

        results = dataset_utils.get_cell_of_coordinate(coords)
        assert tf.reduce_all(expected == results)

    def test_coords_at_grid_edge(self):
        coords = tf.constant([[31, 639]], tf.float32)

        expected = tf.constant([[0, 19]], tf.int32)

        results = dataset_utils.get_cell_of_coordinate(coords)
        assert tf.reduce_all(expected == results)

    def test_coords_at_grid_edge_with_clip(self):
        coords = tf.constant([[31, 639]], tf.float32)

        expected = tf.constant([[0, 14]], tf.int32)

        results = dataset_utils.get_cell_of_coordinate(coords, clip=True)
        assert tf.reduce_all(expected == results)

    def test_negative_coords_with_clip(self):
        coords = tf.constant([[-1, 40], [3, -50], [-1, -1]], tf.float32)

        expected = tf.constant([[0, 1], [0, 0], [0, 0]], tf.int32)

        results = dataset_utils.get_cell_of_coordinate(coords, clip=True)
        assert tf.reduce_all(expected == results)

    def test_too_large_coords_with_clip(self):
        coords = tf.constant([[480, 40], [3, 640], [700, 1000]], tf.float32)

        expected = tf.constant([[15, 1], [0, 14], [19, 14]], tf.int32)

        results = dataset_utils.get_cell_of_coordinate(coords, clip=True)
        tf.print(results)
        assert tf.reduce_all(expected == results)


class TestFlattenCellIndices:
    # These tests are done with the default cell_grid size.

    def test_index_in_first_row(self):
        indices = tf.constant([0, 5], tf.int32)
        expected = tf.constant([100], tf.int32)
        result = dataset_utils.flatten_cell_indices(indices)

        assert tf.reduce_all(expected == result)

    def test_index_in_second_row(self):
        indices = tf.constant([2, 0], tf.int32)
        expected = tf.constant([2], tf.int32)
        result = dataset_utils.flatten_cell_indices(indices)

        assert tf.reduce_all(expected == result)

    def test_index_in_lower_right_corner_of_grid(self):
        indices = tf.constant([19, 14], tf.int32)
        expected = tf.constant([299], tf.int32)
        result = dataset_utils.flatten_cell_indices(indices)

        assert tf.reduce_all(expected == result)

    def test_batched_index(self):
        indices = tf.constant([[[19, 14], [0, 0]], [[19, 14], [0, 0]]], tf.int32)  # (B, N, 2)
        expected = tf.constant([[299, 0], [299, 0]], tf.int32)  # (B, N)
        result = dataset_utils.flatten_cell_indices(indices)

        assert tf.reduce_all(expected == result)

    def test_negative_index(self):
        indices = tf.constant([[[-1, -1], [0, -4]]], tf.int32)  # (B, N, 2)
        expected = tf.constant([[-21, -80]], tf.int32)  # (B, N)
        result = dataset_utils.flatten_cell_indices(indices)

        assert tf.reduce_all(expected == result)


class TestGetDistanceMaskFromOffsets:
    camera = tf.constant(
        [[-0.0220113918, 0.0786367953, 0.493571222], [0.0550611168, 0.298894078, 0.481570929]],
        tf.float32,
    )  # (B, 3)
    camera_intr = tf.constant(
        [[320, 240, 618.663391, 617.159], [320, 240, 618.663391, 617.159]], tf.float32
    )  # (B, 4)

    empty_offset_mask = tf.expand_dims(tf.fill((*dataset_config.output_dims, 2), -1.0), axis=0)
    non_empty_offset_mask = tf.expand_dims(
        dataset_utils._generate_offset_mask(tf.constant([[100, 400]], tf.float32)), axis=0
    )

    offset_mask_batch = tf.concat([empty_offset_mask, non_empty_offset_mask], axis=0)
    tf.print("offset_masks: ", tf.shape(offset_mask_batch))

    def test_shape(self):
        offset_mask_batched = tf.expand_dims(
            dataset_utils._generate_offset_mask(tf.constant([[100, 400]], tf.float32)), axis=0
        )

        distance_mask = dataset_utils.get_distance_mask_from_offsets(
            offset_mask_batched, self.camera[0:1, :], self.camera_intr[0:1, :], object_height=0.0
        )
        tf.print(tf.unique(tf.reshape(distance_mask, [-1])))
        tf.print("Distances: ", tf.shape(distance_mask))
        assert tf.reduce_all(
            tf.shape(distance_mask) == tf.constant([1, *dataset_config.output_dims, 1])
        )

    def test_mult_coords_one_offset_mask(self):
        # Index 0 and 1 are valid. Index 3 is invalid
        coords = tf.constant([[225, 400], [400, 300], [1, 4]], tf.float32)
        offset_mask_batched = tf.expand_dims(dataset_utils._generate_offset_mask(coords), axis=0)

        number_of_valid_coords = 2

        distance_mask = dataset_utils.get_distance_mask_from_offsets(
            offset_mask_batched, self.camera[0:1, :], self.camera_intr[0:1, :], object_height=0.0
        )

        y, _ = tf.unique(tf.reshape(distance_mask, [-1]))

        _, _, unique_count_distances = tf.unique_with_counts(tf.reshape(distance_mask, [-1]))
        _, _, unique_count_coordinates = tf.unique_with_counts(
            tf.reshape(dataset_utils.get_coordinate_mask(offset_mask_batched)[..., 0], [-1])
        )

        tf.print(y)
        tf.print(number_of_valid_coords)

        # Are distribution of distances and coordinates the same?
        assert tf.reduce_all(unique_count_distances == unique_count_coordinates)
        assert len(y) == number_of_valid_coords + 1

    def test_empty_offset_mask(self):
        distance_mask = dataset_utils.get_distance_mask_from_offsets(
            self.empty_offset_mask, self.camera[0:1, :], self.camera_intr[0:1, :], object_height=0.0
        )

        assert tf.reduce_all(distance_mask == -1.0)

    def test_mult_offset_masks(self):
        empty_offset_mask = tf.expand_dims(tf.fill((*dataset_config.output_dims, 2), -1.0), axis=0)
        non_empty_offset_mask = tf.expand_dims(
            dataset_utils._generate_offset_mask(tf.constant([[300, 400]], tf.float32)), axis=0
        )

        offset_mask_batch = tf.concat([empty_offset_mask, non_empty_offset_mask], axis=0)

        distance_mask = dataset_utils.get_distance_mask_from_offsets(
            offset_mask_batch, self.camera, self.camera_intr, object_height=0.0
        )

        y0, _ = tf.unique(tf.reshape(distance_mask[0], [-1]))
        y1, _ = tf.unique(tf.reshape(distance_mask[1], [-1]))

        # First distance mask should be -1.0
        assert tf.reduce_all(distance_mask[0] == -1.0)
        # Second distance mask should be valid
        assert tf.reduce_all(distance_mask[1] >= 0)
        # Test number of unique elements in distance masks
        assert len(y0) == 1
        assert len(y1) == 1


class TestGetCoordinateMask:
    def test_empty_mask(self):
        empty_offset_mask = tf.expand_dims(tf.fill((*dataset_config.output_dims, 2), -1.0), axis=0)
        result = dataset_utils.get_coordinate_mask(empty_offset_mask)

        assert tf.reduce_all(result == empty_offset_mask)

    def test_single_coordinate_pair(self):
        coord_pair = tf.constant([[400, 300]], tf.float32)  # (1, 2)
        offset_mask = tf.expand_dims(dataset_utils._generate_offset_mask(coord_pair), axis=0)

        expected = tf.tile(
            coord_pair[:, tf.newaxis, tf.newaxis, :],  # (1, 1, 1, 2)
            [1, *dataset_config.output_dims, 1],
        )  # (1, 15, 20, 2)

        tf.print("offset_mask: ", tf.shape(offset_mask))
        tf.print("expected: ", tf.shape(expected))

        result = dataset_utils.get_coordinate_mask(offset_mask)

        assert tf.reduce_all(result == expected)
