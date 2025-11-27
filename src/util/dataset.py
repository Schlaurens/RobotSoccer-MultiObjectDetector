from dataclasses import dataclass
from enum import Enum

import numpy as np
import tensorflow as tf


@dataclass
class DatasetConfig:
    input_dims: tuple = (480, 640)
    output_dims: tuple = (15, 20)
    cell_dims: np.ndarray = None
    scale: float = None
    cell_grid: tf.Tensor = None

    def __post_init__(self):
        self.scale = np.array(self.output_dims) / np.array(self.input_dims)
        self.cell_dims = np.array(self.input_dims) // np.array(self.output_dims)

        # Generate the cell grid in the full image scale
        # (values point to upper left corner of each cell)
        self.cell_grid = tf.cast(
            tf.stack(
                tf.meshgrid(
                    range(self.input_dims[1])[:: self.cell_dims[1]],
                    range(self.input_dims[0])[:: self.cell_dims[0]],
                ),
                axis=-1,
            ),
            dtype=tf.float32,
        )  # (15, 20)


class CategoryNames(Enum):
    BALL = "ball"
    PENALTYMARK = "penaltyMark"
    INTERSECTIONS = "intersections"


class IntersectionType(Enum):
    NONE = 0
    L = 1
    T = 2
    X = 3


class DatasetUtils:
    def __init__(self, config: DatasetConfig):
        self.config = config

    def get_masks(
        self,
        label: dict = None,
        object_name: str = None,
        coordinates: list[list[float]] | tf.Tensor = None,
    ) -> dict[str, tf.Tensor]:
        """Return label masks that are used to train the encoder.

        Generate an offset mask that converts the image coordinates of the object into offsets relative
        to given cell dimensions
        An object mask that marks the cell where the center of
        the object is in.
        And a loss mask that indicted which cell should have an impact on the loss function.

        Input can be either (label and object_name) or coordinates.

        Args:
            label: label of the image
            object_name: name of the object to generate masks for
            coordinates: the image coordinates of the object [B, 2]

        Returns:
            a dictionary with all three masks.

        """

        def _empty_masks(ignore_sample: bool = False):
            """generate default empty masks for when there are no objects in the image.
            The offset_mask will contains only -1.0. This is an arbitrary value, that indicates that no object is in the image.
            The object_mask will only contain False values as there are no objects any of the cells.
            The loss_mask will only contain `True` if `ignore_sample` is set to `False` and only `False` values otherwise.
            The classification_mask will only contain -1.0

            Args:
                ignore_sample: If `ignore_sample` is set to `True` the `loss_mask` consists of only `False` values so that this sample is ignored in the loss function. This is useful for the case that there are objects in the image that are not annotated. If `ignore_sample` is False the `loss_mask` consists of only `True` values.

            Returns:
                The masks in a dictionary
            """
            offsets = tf.cast(tf.fill((*self.config.output_dims, 2), -1), dtype=tf.float32)
            object_mask = tf.fill(self.config.output_dims, value=False)
            loss_mask = tf.fill(self.config.output_dims, value=not ignore_sample)
            classification_mask = tf.cast(
                tf.fill(self.config.output_dims, value=-1), dtype=tf.float32
            )

            return {
                "offsets": offsets,
                "object_mask": object_mask,
                "loss_mask": loss_mask,
                "classification_mask": classification_mask,
            }

        # Case 1: Direct coordinates provided
        # TODO: make possible with a list of cooordinates.
        if coordinates is not None:
            if tf.reduce_all(tf.math.equal(coordinates, -1.0)):
                return _empty_masks()
            coordinate_list = coordinates
        # Case 2: Label and object_name provided
        elif label is not None and object_name is not None:
            if object_name not in label:
                return _empty_masks()

            if object_name == CategoryNames.INTERSECTIONS.value:
                if label[object_name]["ignore_sample"]:
                    return _empty_masks(ignore_sample=True)

                # Intersection coords
                l_coords = (
                    tf.constant(
                        [list(x.values()) for x in label[object_name]["L"]], dtype=tf.float32
                    )
                    if len(label[object_name]["L"]) > 0
                    else tf.constant([], dtype=tf.float32, shape=(0, 2))
                )  # (N_L, 2)
                t_coords = (
                    tf.constant(
                        [list(x.values()) for x in label[object_name]["T"]], dtype=tf.float32
                    )
                    if len(label[object_name]["T"]) > 0
                    else tf.constant([], dtype=tf.float32, shape=(0, 2))
                )  # (N_T, 2)
                x_coords = (
                    tf.constant(
                        [list(x.values()) for x in label[object_name]["X"]], dtype=tf.float32
                    )
                    if len(label[object_name]["X"]) > 0
                    else tf.constant([], dtype=tf.float32, shape=(0, 2))
                )  # (N_X, 2)
                coordinate_list = tf.concat([l_coords, t_coords, x_coords], axis=0)  # (N_O, 2)

                if tf.size(coordinate_list) == 0:
                    return _empty_masks()
            else:
                coordinate_list = [
                    list(label[object_name].values())[:2]
                ]  # Only take x and y coordinates (ignore radius) # has to be list of lists
        else:
            raise ValueError("Either (label and object_name) or coordinates must be provided.")

        offset_mask = self._generate_offset_mask(coordinate_list)

        # Mark all cells with true, where the value is between 0 and 1 (object is in that cell)
        object_mask = self.generate_object_mask(offset_mask)
        loss_mask = self._generate_loss_mask(object_mask)

        if object_name == CategoryNames.INTERSECTIONS.value:
            filtered_coords = self.filter_coordinates(coordinate_list)
            classification_mask = tf.Variable(tf.zeros(self.config.output_dims))
            for c in filtered_coords:
                if tf.reduce_any(c == l_coords):
                    indices = self.get_cell_of_coordinate(c)
                    classification_mask[indices[1], indices[0]].assign(IntersectionType.L.value)
                elif tf.reduce_any(c == t_coords):
                    indices = self.get_cell_of_coordinate(c)
                    classification_mask[indices[1], indices[0]].assign(IntersectionType.T.value)
                elif tf.reduce_any(c == x_coords):
                    indices = self.get_cell_of_coordinate(c)
                    classification_mask[indices[1], indices[0]].assign(IntersectionType.X.value)
        else:
            classification_mask = tf.fill((*self.config.output_dims, 3), -1.0)

        return {
            "offsets": offset_mask,
            "object_mask": object_mask,
            "loss_mask": loss_mask,
            "classification_mask": classification_mask,
        }

    def generate_object_mask(self, offset_mask: tf.Tensor) -> tf.Tensor:
        """Generate a binary object_mask that is 1.0 in every cell where there is an object. And 0.0 in all the other cells.

        Args:
            offset_mask: The offset_mask. Shape (H_out, W_out, 2)

        Returns:
            The object mask. Shape (H_out, W_out)
        """
        return [[all(n >= 0 and n < 1 for n in x) for x in row] for row in offset_mask]

    def _generate_offset_mask(self, coordinates) -> tf.Tensor:
        """Generate the offset_mask for the given list of coordinates.

        Args:
            coordinates: A tf.Tensor that contains the coords that make up the offset_mask.

        Returns:
            The offset_mask
        """

        # Remove duplicates and handle multiple coords in one cell
        filtered_coords = self.filter_coordinates(coordinates)

        # Prepare cells for broadcast
        cells_reshaped = tf.expand_dims(self.config.cell_grid, axis=2)  # (H, W, 1, 2)

        distances = tf.sqrt(
            tf.reduce_sum((filtered_coords - cells_reshaped) ** 2, axis=-1)
        )  # (H, W, N_O)
        closest_indices = tf.argmin(distances, axis=-1)  # (H, W)
        closest_coords = tf.gather(filtered_coords, closest_indices)  # (H, W)

        offsets = closest_coords - self.config.cell_grid

        # Scale offsets to the output size
        return offsets * self.config.scale

    def filter_coordinates(self, coordinates: tf.Tensor) -> tf.Tensor:
        """Filter a Tensor of coordinates so that it does not contain any duplicates. And if two coordinates share the same cell only keep the coordinates pair with the higher y-value. If the y-values are equal keep the pair with the higher x-value.

        Args:
            coordinates: The tensor of coordinates that is to be filtered. Shape (N, 2)

        Returns:
            A filtered tensor. Shape (N, 2).
        """

        unique_coords, _ = tf.raw_ops.UniqueV2(x=coordinates, axis=[0])
        filtered_coords = []

        for coords in unique_coords:
            keep = True
            for other_coords in unique_coords:
                if tf.reduce_all(coords == other_coords):
                    continue

                if self.are_coords_in_same_cell(coords, other_coords):
                    diff = coords > other_coords
                    if not diff[1]:
                        keep = False
                    if coords[1] == other_coords[1] and diff[0] and not diff[1]:
                        # Covers the case if the y-values are the same but coords is bigger on the x-axis.
                        keep = True
            if keep:
                filtered_coords.append(coords)

        # TODO: make output a tf.Tensor
        return filtered_coords

    def are_coords_in_same_cell(
        self,
        coords_a: np.ndarray | list[float] | tf.Tensor,
        coords_b: np.ndarray | list[float] | tf.Tensor,
    ) -> bool:
        """Checks whether two given coordinate pairs are inside the same cell in the cellgrid.

        Args:
            coords_a: The first coordinate pair.
            coords_b: The second coordinate pair.

        Returns:
            True if coords_a and coords_b share the same cell.
        """
        return self.get_cell_of_coordinate(coords_a) == self.get_cell_of_coordinate(coords_b)

    def get_cell_of_coordinate(self, coordinate: tf.Tensor) -> tuple[int, int]:
        """Returns the index of the cell in the cell_grid of the given coordinate pair.

        Args:
            coordinate: The coordinate pair

        Returns:
            the index tuple in the cell_grid where the coordinate points to.
        """

        return (
            int(coordinate[0] // self.config.cell_dims[0]),
            int(coordinate[1] // self.config.cell_dims[1]),
        )

    # def _generate_object_mask(self, object_name, label, cells):
    #     """Generate the binary object_mask using the cell coverage values for each object_category.

    #     ===== Work in Progress =====

    #     If the IoU value of the object and the cell is greater than a specified threshold, that cell is marked with a 1.0. And
    #     0.0 otherwise.

    #     Args:
    #         object_name: _description_
    #         label: _description_
    #     """

    #     def _get_threshold(distance, min_threshold=0.1, max_threshold=0.75):
    #         # Do some linear interpolation for the threshold
    #         pass

    #     # Generate object_mask for ball
    #     if object_name == "ball":
    #         # Geometry object of the ball
    #         ball = Point([label[object_name]["x"], label[object_name]["y"]]).buffer(
    #             label[object_name]["radius"], 128
    #         )

    #         # All cells from the cell grid as shapely polygons
    #         cell_polygons = [
    #             Polygon(
    #                 (
    #                     (coords[0], coords[1]),
    #                     (coords[0], coords[1] + 32),
    #                     (coords[0] + 32, coords[1] + 32),
    #                     (coords[0] + 32, coords[1]),
    #                 )
    #             )
    #             for coords in cells.numpy().reshape(-1, 2)
    #         ]

    #         intersections = np.array([ball.intersection(p).area for p in cell_polygons]).reshape(15, 20)
    #         # unions = np.array([ball.union(p).area for p in polygons]).reshape(15, 20)
    #         cell_areas = np.array([p.area for p in cell_polygons]).reshape(15, 20)

    #         cell_coverage = np.divide(intersections, cell_areas)
    #         print(cell_coverage)

    #         # if the ball is inside any of the cells, then the object_mask is 1.0
    #         return cell_coverage > 0

    #     # Generate object_mask for penaltyMark
    #     if object_name == "penaltyMark":
    #         pass

    def _generate_classification_mask(self, object_name: str, coordinates, object_mask):
        """Generate a mask that has a value in each cell that corresponds to the class type of the object category.
        Example:
        The classification mask for line intersections can have four values: NONE, L, T, X. The values

        Returns:
            A mask like described above.
        """
        pass

    def _generate_loss_mask(self, object_mask: tf.Tensor | np.ndarray):
        """Generate a binary mask that is 0 in each cell where the loss function should be ignored and 1 everywhere else

        The loss function should be ignored when the presence of an object inside a cell in ambiguous. Whether this
        is the case can be determined by the IoU value of the object and the cell. If the object is just a 1 dimensional point
        (e. g. a penalty mark) the cell that contains the object coordinates is marked as one and the 8 cells surrounding it are  marked as 0 (just in case).

        Args:
            object_mask: the object mask

        Returns:
            A binary mask like described above.

        """

        # invert object_mask to even make cells without objects True.
        inverted_obj_mask = np.logical_not(np.array(object_mask))  # (H, W)

        object_indices = np.stack(np.where(object_mask)).T  # (N_O, 2)

        # turn the cells surrounding the index cell to 0
        # TODO: use a more elegant way to set the surrounding cells to 0, like convolution or einsum
        for idx in object_indices:
            for i in range(-1, 2):
                for j in range(-1, 2):
                    if i == 0 and j == 0:
                        continue
                    # Check boundries
                    if (0 <= idx[0] + i < inverted_obj_mask.shape[0]) and (
                        0 <= idx[1] + j < inverted_obj_mask.shape[1]
                    ):
                        # Set the surrounding cells to 0
                        inverted_obj_mask[idx[0] + i, idx[1] + j] = 0.0

        for idx in object_indices:
            inverted_obj_mask[idx[0]][idx[1]] = 1.0

        return inverted_obj_mask

    @tf.function
    def get_coords_from_offsets(
        self,
        offset_mask: tf.Tensor,
    ) -> tf.Tensor:
        """Extract the image coordinates from the offset mask

        Args:
            mask: the offset mask [B, H, W, 2]

        Returns:
            A `tf.Tensor` of shape (N, 2) with N := Number of objects in the sample. The `tf.Tensor` contains the coordinates (x, y) of the objects. (-1.0, -1.0) if the object is not in the image
        """

        # If all the values are -1.0 then there are no objects in mask.
        if tf.reduce_all(offset_mask == -1.0):
            return tf.constant([[-1, -1]], tf.float32)

        # Convert the offset_mask to an absolute coord_mask where all values that are [-1.0, -1.0] are set to 0.
        coord_mask = offset_mask / self.config.scale + self.config.cell_grid

        flat_mask = tf.reshape(
            coord_mask, [self.config.output_dims[0] * self.config.output_dims[1], 2]
        )

        rounded_coords = (
            tf.round(flat_mask * 1e4) / 1e4
        )  # Round to 4 decimal places due to floating point errors.
        unique_coords, _ = tf.raw_ops.UniqueV2(x=rounded_coords, axis=[0])

        return unique_coords
