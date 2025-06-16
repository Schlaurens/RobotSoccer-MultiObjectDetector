import tensorflow as tf


class PatchExtractor(tf.keras.layers.Layer):
    """This layer extracts patches from an image. The center coordinates are given,
    and the size in pixels of the crop in the original image is determined based on
    a desired size in meters, the height of the object over the ground, and the camera pose.
    Thus, the patch will roughly cover the the same "area" regardless of where in the image it is.
    """

    def __init__(
        self,
        patch_size=(32, 32),
        object_size=0.2,
        object_height=0,
        interpolation="nearest",
        name="patch_extractor",
        **kwargs,
    ):
        """Constructor.

        :param patch_size: The size (height, width) in pixels of each extracted patch.
        :param object_size: The size/diameter of an object in meters.
        :param object_height: The height of the object center above the ground in meters.
        :param interpolation: The interpolation method to use when up/downsampling the patch.
        :param name: The name of the layer.
        """
        super().__init__(name=name, **kwargs)
        self.patch_size = tf.constant(patch_size, dtype=tf.int32)
        self.object_size = object_size
        self.object_height = 0.5 * object_size
        self.interpolation = interpolation

    @staticmethod
    @tf.function(jit_compile=False)
    def to_rotation_matrix(camera):
        """Converts the (roll, pitch, height) representation to a rotation matrix according to the rodrigues formula.

        :param camera: The extrinsic camera parameters (roll, pitch, height).
            [B, 3]
        :return: The corresponding rotation matrix.
            [B, 3, 3]
        """
        angle = tf.math.reduce_euclidean_norm(camera[..., :2], axis=-1)
        x = camera[..., 0] / angle
        y = camera[..., 1] / angle
        c, s = tf.cos(angle), tf.sin(angle)
        return tf.stack(
            [
                tf.stack([x * x * (1 - c) + c, x * y * (1 - c), y * s], axis=-1),
                tf.stack([y * x * (1 - c), y * y * (1 - c) + c, -x * s], axis=-1),
                tf.stack([-y * s, x * s, c], axis=-1),
            ],
            axis=-2,
        )

    @tf.function(jit_compile=False)
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
                (intrinsics[..., tf.newaxis, :2] - coords) / intrinsics[..., tf.newaxis, 2:],
            ],
            -1,
        )  # [B, N, 3]

        # get the rotation matrix for the camera rotation.
        camera_rotation = self.to_rotation_matrix(camera)  # [B, 3, 3]

        # Rotate the camera rays with the rotation matrix
        rotated_camera_rays = tf.einsum(
            "...ij,...j->...i", camera_rotation, camera_rays
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
        pixel_sizes = (
            self.object_size * tf.expand_dims(intrinsics[..., 2], -1) / distances_in_camera
        )  # [B, N]

        # print("Pixel Sizes", pixel_sizes)
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
                boxes[..., 1] / maxcoord[0],
                boxes[..., 0] / maxcoord[1],
                boxes[..., 3] / maxcoord[0],
                boxes[..., 2] / maxcoord[1],
            ],
            axis=-1,
        )  # [B, N, 4]
        boxes = tf.reshape(boxes, (-1, 4))
        box_indices = tf.repeat(tf.range(tf.shape(coords)[0]), tf.shape(coords)[1])

        # Extract patches from image
        patches = tf.image.crop_and_resize(
            image,
            boxes,
            box_indices,
            self.patch_size,
            method=self.interpolation,
            extrapolation_value=127.5,
        )  # [B*N, H_out, W_out, C]
        patches = tf.reshape(
            patches, tf.concat([tf.shape(masks), tf.shape(patches)[-3:]], -1)
        )  # [B, N, H_out, W_out, C]

        return (
            patches,
            masks,
            boxes,
            camera_rays,
            camera_rotation,
            rotated_camera_rays,
            camera_height,
            factors,
            distances_in_camera,
            positions_in_camera,
            pixel_sizes,
            coords,
            intrinsics,
        )


class PatchSampler(tf.keras.layers.Layer):
    """This layer samples a number of indices from a given distribution.
    The behavior differs by training and test mode: In training mode, the values are sampled randomly (according to the weights),
    while in test mode, the top weighted patches are chosen deterministically.
    """

    def __init__(
        self,
        n_sample,
        temperature=1,
        generator=tf.random.get_global_generator(),  # noqa: B008
        name="patch_sampler",
        **kwargs,
    ):
        """Constructor.

        :param n_sample: The number of samples to draw.
        :param temperature: At temperature 0, the samples are selected deterministically as the top N.
            At temperature 1, the samples are selected according to the given probability distribution.
            At infinite temperature, samples are selected uniformly.
        :param generator: The generator from which random numbers are sampled.
        :param name: The name of the layer.
        """
        super().__init__(name=name, **kwargs)
        self.n_sample = n_sample
        self.temperature = temperature
        self.generator = generator

    def call(self, logits, training=None):
        """Samples indices from a given distribution.

        :param logits: A vector of logits.
            [B, N_in]
        :param training: In training mode, N values are sampled (without replacement)
        :return: The indices that were sampled.
            [B, N_out]
        """
        if training and self.temperature > 0:
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
        _, indices = tf.math.top_k(logits, self.n_sample)
        return indices
