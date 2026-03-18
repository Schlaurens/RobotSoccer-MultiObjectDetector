import numpy as np
import tensorflow as tf


class Normalization(tf.keras.layers.Layer):
    def __init__(
        self,
        batch_norm: bool = False,
        scale: bool = False,
        groups: int = -1,
        name: str = None,
        **kwargs,
    ):
        """This Layer returns a normalization layer that is either BatchNormalization or GroupNormalization.

        Args:
            batch_norm: If True normalization will be BatchNorm, else GroupNorm . Defaults to False.
            scale: If True, multiply by gamma. If False, gamma is not used. Defaults to False.
            groups: Size of the groups for the GroupNormalization. If -1 the groupsize is the size of the input dimension (InstanceNormalization). Defaults to -1.
            name: The Name of the layer. Defaults to None.
        """
        super().__init__(name=name, **kwargs)
        self.batch_norm = batch_norm
        self.scale = scale
        self.groups = groups

    def build(self, input_shape):
        # Create the normalization layer here, so it gets built with the correct input shape
        if self.batch_norm:
            self.norm_layer = tf.keras.layers.BatchNormalization(scale=self.scale)
        else:
            groups = input_shape[-1] if self.groups == -1 else self.groups
            self.norm_layer = tf.keras.layers.GroupNormalization(
                scale=self.scale, groups=groups, epsilon=1e-3
            )
        # Call build on the sub-layer to ensure weights are created
        self.norm_layer.build(input_shape)
        super().build(input_shape)  # Call the parent build method

    def call(self, inputs):
        return self.norm_layer(inputs)


class IresBlock(tf.keras.layers.Layer):
    def __init__(
        self,
        filters: int,
        use_batch_norm: bool = False,
        stride: int = 1,
        expansion: int = 6,
        name: str = None,
        **kwargs,
    ):
        """Inverted residual block as specified in MobileNetV2

        Args:
            filters: Number of filters
            use_batch_norm: If True use Batch Normalization. Else use Group Normalization.
            stride: The stride. Defaults to 1.
            expansion: Expand the number of filters by multiplying them with this number. Defaults to 6.
            name: The Name of the layer. Defaults to None.
        """
        super().__init__(name=name, **kwargs)
        self.filters = filters
        self.use_batch_norm = use_batch_norm
        self.expansion = expansion
        self.stride = stride

    def build(self, input_shape):
        # Expansion phase: 1x1 convolution to expand channels
        self.conv_expand = tf.keras.layers.Conv2D(
            self.filters * self.expansion,
            1,
            padding="same",
            use_bias=False,
            name=f"{self.name}_conv_expand",
        )
        self.norm_expand = Normalization(
            batch_norm=self.use_batch_norm, scale=False, groups=-1, name=f"{self.name}_norm_expand"
        )

        # Use Depthwise convolution
        self.conv_depthwise = tf.keras.layers.DepthwiseConv2D(
            3,
            strides=self.stride,
            padding="same",
            use_bias=False,
            name=f"{self.name}_conv_depthwise",
        )
        self.norm_depthwise = Normalization(
            batch_norm=self.use_batch_norm,
            scale=False,
            groups=-1,
            name=f"{self.name}_norm_depthwise",
        )

        # Projection phase: 1x1 convolution to project back to original channels
        self.conv_projection = tf.keras.layers.Conv2D(
            self.filters, 1, padding="same", use_bias=False, name=f"{self.name}_conv_projection"
        )
        self.norm_projection = Normalization(
            batch_norm=self.use_batch_norm,
            scale=False,
            groups=-1,
            name=f"{self.name}_norm_projection",
        )

        # Residual projection if dimensions change
        self.conv_residual = tf.keras.layers.Conv2D(
            self.filters,
            1,
            strides=self.stride,
            padding="same",
            use_bias=False,
            name=f"{self.name}_conv_residual",
        )
        self.norm_residual = Normalization(
            batch_norm=self.use_batch_norm,
            scale=False,
            groups=-1,
            name=f"{self.name}_norm_residual",
        )

        # Activation and Add layers
        self.relu = tf.keras.layers.ReLU(6.0)
        self.add = tf.keras.layers.Add()

        super().build(input_shape)

    def call(self, inputs):
        residual = inputs

        # Expansion phase: 1x1 convolution to expand channels
        x = self.conv_expand(inputs)
        x = self.norm_expand(x)
        x = self.relu(x)

        # Use Depthwise convolution
        x = self.conv_depthwise(x)
        x = self.norm_depthwise(x)
        x = self.relu(x)

        # Projection phase: 1x1 convolution to project back to original channels
        x = self.conv_projection(x)
        x = self.norm_projection(x)

        # If dimensions changed, project the residual
        if self.stride != 1 or residual.shape[-1] != self.filters:
            residual = self.conv_residual(residual)
            residual = self.norm_residual(residual)

        # Add residual
        return self.add([x, residual])


class PatchExtractor(tf.keras.layers.Layer):
    def __init__(
        self,
        patch_size: tuple[int] | list[int] = (32, 32),
        object_size: float = 0.2,
        object_height: float = 0,
        interpolation: str = "nearest",
        name: str = "patch_extractor",
        **kwargs,
    ):
        """This layer extracts patches from an image. The center coordinates are given,
        and the size in pixels of the crop in the original image is determined based on
        a desired size in meters, the height of the object over the ground, and the camera pose.
        Thus, the patch will roughly cover the the same "area" regardless of where in the image it is.

        :param patch_size: The size (height, width) in pixels of each extracted patch.
        :param object_size: The size/diameter of an object in meters.
        :param object_height: The height of the object center above the ground in meters.
        :param interpolation: The interpolation method to use when up/downsampling the patch.
        :param name: The name of the layer.
        """
        super().__init__(name=name, **kwargs)
        self.patch_size = tf.constant(patch_size, dtype=tf.int32)
        self.object_size = object_size
        self.object_height = object_height
        self.interpolation = interpolation

    @staticmethod
    def to_rotation_matrix(camera):
        """Converts the (roll, pitch, height) representation to a rotation matrix according to the rodrigues formula.

        :param camera: The extrinsic camera parameters (roll, pitch, height).
            [B, 3]
        :return: The corresponding rotation matrix.
            [B, 3, 3]
        """
        angle = tf.math.reduce_euclidean_norm(camera[..., :2], axis=-1)
        x = tf.math.divide_no_nan(camera[..., 0], angle)
        y = tf.math.divide_no_nan(camera[..., 1], angle)
        c, s = tf.cos(angle), tf.sin(angle)
        return tf.stack(
            [
                tf.stack([x * x * (1 - c) + c, x * y * (1 - c), y * s], axis=-1),
                tf.stack([y * x * (1 - c), y * y * (1 - c) + c, -x * s], axis=-1),
                tf.stack([-y * s, x * s, c], axis=-1),
            ],
            axis=-2,
        )

    @tf.function
    def call(self, image, coords, camera, intrinsics, training=None):
        """Extracts patches of fixed size at given coordinates from an image.

        :param image: The full resolution image from which the patches are extracted.
            [B, H_in, W_in, C]
        :param coords: The center coordinates (x, y) at which patches are extracted.
            [B, N, 2]
        :param camera: The pose of the camera (roll, pitch, height).
            [B, 3]
        :param intrinsics: The intrisics of the camera (cx, cy, fx, fy).
            [B, 4]
        :param training:
        :return: The extracted scaled patches.
            [B, N, H_out, W_out, C]
        :return: Indicates for each patch whether the center could be projected to the plane.
            [B, N]
        """
        # Calculate how big the object would be at each given point.
        # Calculate camera rays. x-Axis points into the image.
        camera_rays = tf.concat(
            [
                tf.ones_like(coords[..., :1]),
                tf.math.divide_no_nan(
                    (intrinsics[..., tf.newaxis, :2] - coords), intrinsics[..., tf.newaxis, 2:]
                ),
            ],
            -1,
        )  # [B, N, 3]

        # get the rotation matrix for the camera rotation.
        camera_rotation = self.to_rotation_matrix(camera)  # [B, 3, 3]

        # Rotate the camera rays with the rotation matrix
        rotated_camera_rays = tf.einsum(
            "...ij,...nj->...ni", camera_rotation, camera_rays
        )  # [B, N, 3]

        # Calculate intersection point with ground
        camera_height = tf.expand_dims(camera[..., 2], -1)  # [B, 1]
        factors = tf.math.divide_no_nan(
            (self.object_height - camera_height), rotated_camera_rays[..., 2]
        )  # [B, N]

        # True if coords could be projected on the plane. Else false.
        masks = factors > 0  # [B, N]
        positions_in_camera = factors[..., tf.newaxis] * rotated_camera_rays  # [B, N, 3]
        distances_in_camera = tf.math.reduce_euclidean_norm(positions_in_camera, axis=-1)  # [B, N]
        pixel_sizes = tf.math.divide_no_nan(
            self.object_size * tf.expand_dims(intrinsics[..., 2], -1), distances_in_camera
        )  # [B, N]

        # Calculate bounding boxes (TODO: margin in pixels).
        boxes = tf.concat(
            [
                coords - 0.5 * pixel_sizes[..., tf.newaxis],
                coords + 0.5 * pixel_sizes[..., tf.newaxis],
            ],
            -1,
        )  # [B, N, 4] (x1, y1, x2, y2)
        maxcoord = tf.cast(tf.shape(image)[-3:-1] - 1, boxes.dtype)
        boxes = tf.stack(
            [
                tf.math.divide_no_nan(boxes[..., 1], maxcoord[0]),
                tf.math.divide_no_nan(boxes[..., 0], maxcoord[1]),
                tf.math.divide_no_nan(boxes[..., 3], maxcoord[0]),
                tf.math.divide_no_nan(boxes[..., 2], maxcoord[1]),
            ],
            axis=-1,
        )  # [B, N, 4]
        boxes = tf.reshape(boxes, (-1, 4))
        box_indices = tf.repeat(tf.range(tf.shape(coords)[0]), tf.shape(coords)[1])

        boxes_no_nan = tf.where(tf.math.is_nan(boxes), 0.0, boxes)

        # if tf.reduce_any(tf.math.is_nan(boxes)):
        #     tf.print("Boxes with nan: ", boxes)
        #     tf.print("Boxes_no_nan: ", boxes_no_nan)

        # Extract patches from image
        patches = tf.image.crop_and_resize(
            image,
            boxes_no_nan,
            box_indices,
            self.patch_size,
            method=self.interpolation,
            extrapolation_value=127.5,
        )  # [B*N, H_out, W_out, C]
        patches = tf.reshape(
            patches, tf.concat([tf.shape(masks), tf.shape(patches)[-3:]], -1)
        )  # [B, N, H_out, W_out, C]

        # TODO: dont return so much unneeded variables
        return (
            patches,
            masks,
            boxes,
            intrinsics,
            distances_in_camera,
        )


class PatchSampler(tf.keras.layers.Layer):
    def __init__(
        self,
        n_sample: int,
        max_distance: float = 5,
        temperature: int = 1,
        generator=tf.random.get_global_generator(),  # noqa: B008
        name: str = "patch_sampler",
        **kwargs,
    ):
        """This layer samples a number of indices from a given distribution.
        The behavior differs by training and test mode: In training mode, the values are sampled randomly (according to the weights),
        while in test mode, the top weighted patches are chosen deterministically.

        :param n_sample: The number of samples to draw.
        :param temperature: At temperature 0, the samples are selected deterministically as the top N.
            At temperature 1, the samples are selected according to the given probability distribution.
            At infinite temperature, samples are selected uniformly.
        :param generator: The generator from which random numbers are sampled.
        :param name: The name of the layer.
        """
        super().__init__(name=name, **kwargs)
        self.n_sample = n_sample
        self.max_distance = max_distance
        self.temperature = temperature
        self.generator = generator

    @tf.function
    def call(self, logits, distances, training=None):
        """Samples indices from a given distribution.

        :param logits: A vector of logits.
            [B, N_in]
        :param training: In training mode, N values are sampled (without replacement)
        :return: The indices that were sampled.
            [B, N_out]
        """
        # All cells that point to an object that is too far away is marked as invalid by assigning it the value -inf.
        valid_cells_mask = tf.logical_and(
            distances <= self.max_distance, distances >= 0
        )  # (B, H_o, W_o)
        masked_logits = tf.where(valid_cells_mask, logits, -np.inf)  # (B, H_o, W_o)

        if training and self.temperature > 0 and False:
            # Apply the so-called Gumbel-max trick:
            # https://github.com/tensorflow/tensorflow/issues/9260
            # https://lips.cs.princeton.edu/the-gumbel-max-trick-for-discrete-distributions/
            # https://timvieira.github.io/blog/post/2014/07/31/gumbel-max-trick/
            # https://timvieira.github.io/blog/post/2019/09/16/algorithms-for-sampling-without-replacement/
            z = -tf.math.log(
                -tf.math.log(
                    self.generator.uniform(tf.shape(logits), minval=0, maxval=1, dtype=logits.dtype)
                )
            )
            logits = logits / self.temperature + z
        _, indices = tf.math.top_k(masked_logits, self.n_sample)
        return indices
