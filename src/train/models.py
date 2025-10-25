import os

import tensorflow as tf

from train import classifier_architectures as u_classifiers
from train import encoder_architectures as u_encoders
from util import dataset as u_dataset
from util import image as u_image
from util import keypoint as u_keypoint

from .layers import IresBlock, Normalization, PatchExtractor, PatchSampler


class FullModel(tf.keras.Model):
    def __init__(
        self,
        encoder_architecture: str,
        classifier_architecture: str,
        height: int,
        width: int,
        n_context: int = 0,
        only_train_encoder: bool = False,
        classifier_offsets: bool = True,
        n_meta: int = 0,
        encoder_use_batch_norm: bool = False,
        classifier_use_batch_norm: bool = False,
        categories_config: dict = None,
    ):
        """Constructs the FullModel

        Args:
            encoder_architecture: the name of the encoder architecture
            height: encoder input height
            width: encoder input width
            only_train_encoder: True if ONLY the encoder is the be trained. Then the classifiers have no impact on the loss function. Defaults to False.
            n_context: The size of the context vector. Defaults to 0.
            classifier_architecture: the name of classifier architecture. Defaults to None.
        """
        super().__init__()
        self.image_height = height
        self.image_width = width

        # Encoder config
        self.encoder_architecture = encoder_architecture
        self.n_context = n_context
        self.only_train_encoder = only_train_encoder
        self.encoder_use_batch_norm = encoder_use_batch_norm

        # Classifier config
        self.classifier_architecture = classifier_architecture
        self.patch_size = [32, 32]
        self.patch_channels = 3
        self.n_meta = n_meta
        self.classifier_offsets = classifier_offsets
        self.classifier_use_batch_norm = classifier_use_batch_norm

        self.full_image_size = tf.constant(
            [self.image_height, self.image_width * 2], dtype=tf.float32
        )  # constructor input image_width is halved due to YUYV
        self.categories = (
            categories_config
            if categories_config is not None
            else {
                "ball": {
                    "object_size": 0.175,
                    "object_height": 0.05,
                    "n_classes": 1,
                    "n_candidates": 5,
                },
            }
        )
        for _, value in self.categories.items():
            value["sampler"] = PatchSampler(
                value["n_candidates"]
            )  # The patch sampler for the category with a fixed number of candidates
            value["extractor"] = PatchExtractor(
                self.patch_size,
                value["object_size"],
                value.get("object_height", 0),
            )  # The patch extractor for the category with the fixed object parameters
            value["classifier"] = u_classifiers.get_classifier(
                self.classifier_architecture,
                self.patch_size,
                self.patch_channels,
                self.n_meta,
                self.n_context,
                value["n_classes"],
                self.classifier_offsets,
                self.classifier_use_batch_norm,
            )  # The patch classifier for the category with the fixed number of classes
        self.encoder = u_encoders.get_encoder(
            self.encoder_architecture,
            self.image_height,
            self.image_width,
            self.categories.keys(),
            self.n_context,
            self.encoder_use_batch_norm,
        )

    def encoder_loss(self, batch_data, maps):
        # Compute Binary Cross Entropy

        # Numeric stabilizer
        epsilon = tf.constant(1e-7, dtype=tf.float32)
        element_wise_bce = -(
            batch_data["object_mask"] * tf.math.log(maps[..., 2] + epsilon)
            + (1.0 - batch_data["object_mask"]) * tf.math.log(1.0 - maps[..., 2] + epsilon)
        )  # (B, 15, 20)
        tf.debugging.assert_non_negative(maps[..., 2], "maps[..., 2] is negative")
        tf.debugging.assert_all_finite(tf.math.log(epsilon), "tf.math.log(epsilon)")
        tf.debugging.assert_all_finite(
            tf.math.log(maps[..., 2] + epsilon), "tf.math.log(maps[..., 2] + epsilon)"
        )
        tf.debugging.assert_none_equal(1.0 - maps[..., 2] + epsilon, 0.0, message="equals 0.0")
        tf.debugging.assert_all_finite(
            tf.math.log(1.0 - maps[..., 2] + epsilon), "tf.math.log(1.0 - maps[..., 2] + epsilon"
        )
        tf.debugging.assert_all_finite(batch_data["object_mask"], "batch_data[object_mask]")
        tf.debugging.assert_all_finite(
            (1.0 - batch_data["object_mask"]), "(1.0 - batch_data[object_mask])"
        )

        tf.debugging.assert_all_finite(element_wise_bce, "element_wise_bce")
        element_wise_bce_multiplied = tf.multiply(
            element_wise_bce, batch_data["loss_mask"]
        )  # (B, 15, 20)

        tf.debugging.assert_all_finite(batch_data["loss_mask"], "batch_data[loss_mask]")
        tf.debugging.assert_all_finite(element_wise_bce_multiplied, "element_wise_bce_multiplied")

        bce_batched = tf.reduce_sum(element_wise_bce_multiplied, axis=[1, 2])  # (B, )

        # Compute MSE
        squared_error = tf.keras.losses.MeanSquaredError(reduction="none")(
            y_true=batch_data["offset_mask"], y_pred=maps[..., :2]
        )  # (B, 15, 20)
        squared_error_multiplied = tf.multiply(
            squared_error, batch_data["object_mask"]
        )  # (B, 15, 20)

        mse_batched = tf.reduce_mean(squared_error_multiplied, axis=[1, 2]) * 10000  # (B)
        tf.debugging.assert_all_finite(bce_batched, "encoder BCE")
        tf.debugging.assert_all_finite(mse_batched, "encoder mse")

        # Total loss
        loss_batched = bce_batched + mse_batched  # (B, )

        bce = tf.reduce_mean(bce_batched)  # Shape: ()
        mse = tf.reduce_mean(mse_batched)  # Shape: ()
        loss = tf.reduce_mean(loss_batched)  # Shape: ()

        return {"loss": loss, "mse": mse, "bce": bce}

    def classifier_loss(self, batch_data, results):
        # Compute MSE
        boxes = results["boxes"]  # [B, N, 4]
        coords_pred = results["positions"]  # [B, N, 2]
        coords_true = tf.expand_dims(
            u_dataset.get_coords_from_offsets(batch_data["offset_mask"]), axis=1
        )  # [B, 1, 2] (x, y)

        # Theoretical maximum error, distance between (0,0) and (max, max) of patch
        max_error = tf.norm(tf.cast(self.patch_size, dtype=tf.float32))  # Shape: ()

        # Normalize coords to the image dimensions. Because the boxes coords are also normalized.
        # Switch axes of full_image_size because: coords_true (x, y), full_image_size (y, x)
        coords_true_normalized = (
            coords_true / self.full_image_size[::-1][tf.newaxis, :]
        )  # [B, 1, 2]
        are_coords_true_inside_patch = u_keypoint.are_coords_in_patch(
            coords_true_normalized, boxes
        )  # [B, N]

        squared_error = tf.where(
            are_coords_true_inside_patch,
            tf.reduce_mean(tf.square(coords_pred - coords_true), axis=-1),
            tf.square(
                max_error
            ),  # If coords_true are inside the patch always calculate the MSE. Else the classifier's offset predictions are useless and should be ignored. Assign a constant max error that has gradient of zero.
        )  # [B, N]

        # TODO: implement solution for multi-class problems with categorical crossentropy (like line crossings)
        # Compute BinaryCrossEntropy / CategoricalCrossEntropy
        y_pred = results["classification"]  # [B, N]
        y_true = are_coords_true_inside_patch  # [B, N]

        bce = tf.keras.losses.BinaryCrossentropy(from_logits=False, name="classifier_bce")(
            y_true, y_pred
        )  # Shape: ()

        # If the classifier thinks that there is no object in the image, this error has a smaller contribution to the loss
        squared_error_multiplied = squared_error * y_pred  # [B, N]
        mse = tf.reduce_mean(squared_error_multiplied)  # Shape: ()

        tf.debugging.assert_all_finite(mse, "Classifier MSE")
        tf.debugging.assert_all_finite(bce, "Classifier BCE")

        loss = bce + mse  # Shape: ()

        return {"loss": loss, "mse": mse, "bce": bce}

    def _calculate_losses(self, batch_data, results, maps):
        encoder_losses = {
            key: self.encoder_loss(batch_data[key], maps[key]) for key in self.categories
        }
        classifier_losses = {
            key: self.classifier_loss(batch_data[key], results=value)
            for key, value in results.items()
        }  # (loss, mse, bce) for each category

        return {
            "encoder_loss": tf.reduce_sum([value["loss"] for value in encoder_losses.values()]),
            "encoder_bce": tf.reduce_sum([value["bce"] for value in encoder_losses.values()]),
            "encoder_mse": tf.reduce_sum([value["mse"] for value in encoder_losses.values()]),
            "classifier_loss": tf.reduce_sum(
                [value["loss"] for value in classifier_losses.values()]
            ),
            "classifier_bce": tf.reduce_sum([value["bce"] for value in classifier_losses.values()]),
            "classifier_mse": tf.reduce_sum([value["mse"] for value in classifier_losses.values()]),
        }

    def train_step(self, batch_data):
        with tf.GradientTape() as tape:
            outputs = self(
                (batch_data["image"], batch_data["camera"], batch_data["intrinsics"]), training=True
            )  # calls call()

            losses = self._calculate_losses(batch_data, outputs["results"], outputs["maps"])

            total_loss = (
                losses["encoder_loss"]
                if self.only_train_encoder
                else losses["encoder_loss"] + losses["classifier_loss"]
            )

        # Compute gradients
        gradients = tape.gradient(total_loss, self.trainable_variables)

        # Update weights
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables, strict=True))

        return {
            "total_loss": total_loss,
            "encoder_bce": losses["encoder_bce"],
            "encoder_mse": losses["encoder_mse"],
            "encoder_loss": losses["encoder_loss"],
            "classifier_bce": losses["classifier_bce"],
            "classifier_mse": losses["classifier_mse"],
            "classifier_loss": losses["classifier_loss"],
        }

    def test_step(self, batch_data):
        outputs = self(
            (batch_data["image"], batch_data["camera"], batch_data["intrinsics"]), training=False
        )  # calls call()

        losses = self._calculate_losses(batch_data, outputs["results"], outputs["maps"])

        total_loss = (
            losses["encoder_loss"]
            if self.only_train_encoder
            else losses["encoder_loss"] + losses["classifier_loss"]
        )

        return {
            "total_loss": total_loss,
            "encoder_bce": losses["encoder_bce"],
            "encoder_mse": losses["encoder_mse"],
            "encoder_loss": losses["encoder_loss"],
            "classifier_bce": losses["classifier_bce"],
            "classifier_mse": losses["classifier_mse"],
            "classifier_loss": losses["classifier_loss"],
        }

    def save(
        self, filepath, filename, only_save_encoder=False, overwrite=True, verbose=False, **kwargs
    ):
        """Save all the models at a given path

        Args:
            filepath: path to the saving directory
            filename: the filename of the .keras model files. Usually a timestamp
            only_save_encoder: True if ONLY the encoder should be saved. Defaults to False.
            overwrite: True if files with the same filename should be overwritten. Defaults to True.
            verbose: Print status messages that describe the status of the saving process. Defaults to False.
        """
        # Create a directory for the encoder
        os.makedirs(os.path.join(filepath, "encoder"), exist_ok=True)

        # Save the encoder
        encoder_path = os.path.join(filepath, "encoder", f"{filename}.keras")

        self.encoder.save(encoder_path, overwrite)

        if verbose:
            print("Encoder saved!")

        if not only_save_encoder:
            # Save the classifier of each category
            for name, value in self.categories.items():
                # Create directory if it does not exist.
                os.makedirs(os.path.join(filepath, "classifier", name), exist_ok=True)

                classifier_path = os.path.join(filepath, "classifier", name, f"{filename}.keras")
                value["classifier"].save(classifier_path, overwrite)

                if verbose:
                    print(f"{name.capitalize()}-Classifier saved!")

        if verbose:
            print("only_save_encoder = ", only_save_encoder)
            print("Saving complete!")

    @classmethod
    def load(
        cls,
        encoder_architecture: str,
        classifier_architecture: str,
        input_dims: list[int] | tuple[int],
        filepath: str,
        filename: str,
        n_context: int = 0,
        only_train_encoder: bool = False,
        classifier_offsets: bool = True,
        encoder_only: bool = False,
        verbose: bool = False,
        n_meta: int = 0,
        encoder_use_batch_norm: bool = False,
        classifier_use_batch_norm: bool = False,
        categories_config: dict = None,
        **kwargs,
    ):
        """load existing encoder and/or classifiers into the model.

        Args:
            encoder_architecture: The name of the used model architecture
            input_dims: the input dimensions of the encoder. (height, width)
            filepath: the filepath to models directory
            filename: the name of the .keras file
            only_train_encoder: True if only the encoder has an impact on the loss function. The classifier will have no impact on the training. Defaults to False.
            encoder_only: True if ONLY an existing encoder is loaded and the classifier. Defaults to False.
            verbose: Print status messages that describe the status of the loading process. Defaults to False.

        Returns:
            A tf.keras.Model with the loaded models.
        """
        # Rebuild model
        model = cls(
            encoder_architecture,
            classifier_architecture,
            *input_dims,
            n_context,
            only_train_encoder,
            classifier_offsets,
            n_meta,
            encoder_use_batch_norm,
            classifier_use_batch_norm,
            categories_config,
        )

        # Load the encoder
        encoder = tf.keras.models.load_model(
            os.path.join(filepath, "encoder", f"{filename}"),
            custom_objects={"IresBlock": IresBlock, "Normalization": Normalization},
        )
        model.encoder = encoder

        if verbose:
            print("Encoder loaded!")

        # Load each classifier
        if not encoder_only:
            for name, value in model.categories.items():
                try:
                    classifier_path = os.path.join(filepath, "classifier", name, f"{filename}")
                    classifier = tf.keras.models.load_model(
                        classifier_path,
                        custom_objects={"IresBlock": IresBlock, "Normalization": Normalization},
                    )
                    value["classifier"] = classifier

                    if verbose:
                        print(f"{name.capitalize()}-Classifier loaded!")
                except Exception as e:
                    if verbose:
                        print(f"Failed to load {name.capitalize()}-Classifier: {e}")

        # If only the encoder is loaded, the model needs to be compiled again for training.
        if encoder_only:
            model.compile(optimizer=tf.keras.optimizers.Adam(), jit_compile=False)

        if verbose:
            print("Only Train Encoder = ", only_train_encoder)
            print("Loading complete!")
        return model

    @tf.function
    def call(self, batch_data, training=None):
        """
        Args:
            image: The image (yuyv) from which the patches are extracted. [B, H_in, W_in/2, C]
            camera: The pose of the camera, represented as (roll angle, pitch angle, height above ground) [B, 3]
            intrinsics: The intrisics of the camera, represented as (cx, cy, fx, fy) [B, 4]

        Returns:
            Per category x,y,p tuples {key: [B, N_out, 2 + n_classes + 1?]}
        """
        image, camera, intrinsics = batch_data
        maps = self.encoder(image, training=training)  # Run the encoder on the image

        # assert isinstance(maps, list) == len(self.categories) > 1
        if isinstance(maps, list):  # If there is a context vector
            maps = dict(
                zip(self.encoder.output_names, maps, strict=True)
            )  # [B, H_out, W_out, 3], ([B, H_out, W_out, n_context])
        else:
            maps = {
                self.encoder.output_names[0]: maps
            }  # [B, H_out, W_out, 3] Encoder results for the first category

        image_yuv = u_image.convert_yuyv_to_yuv(image)

        results = {
            key: self._handle_category(
                image_yuv,
                camera,
                intrinsics,
                maps[key][..., 2],
                maps[key][..., :2],
                value["sampler"],
                value["extractor"],
                value["classifier"],
                training=training,
            )
            for key, value in self.categories.items()
        }  # Call _handle_category for each category and store the results in a dictionary

        # results: patches, masks
        # maps: [B, H_out, W_out, 3] (offsets, logits) or [B, H_out, W_out, n_context]
        return {"results": results, "maps": maps}

    def _handle_category(
        self,
        image,
        camera,
        intrinsics,
        logits,
        offsets,
        sampler,
        extractor,
        classifier,
        training=None,
    ):
        """

        Args:
            image: The full resolution image from which the patches are extracted. [B, H_in, W_in, C]
            camera: The pose of the camera, represented as (roll angle, pitch angle, height above ground) [B, 3]
            intrinsics: The intrisics of the camera, represented as (cx, cy, fx, fy) [B, 4]
            logits: The logits for the category. [B, H_out, W_out]
            offsets: The offsets for the category. Relative to the upper left corner of the patch. [B, H_out, W_out, 2]
            sampler: The patch sampler for the category with a fixed number of candidates
            extractor: The patch extractor for the category with the fixed object parameters
            classifier: The patch classifier for the category with the fixed number of classes
            training: Whether the model is in training mode or not.

        Returns:
            dict:
                patches: [B, N_out, *patch_size, channel_dims]
                masks: [B, N_out]
                boxes: [B, N_out, 4]
                coords: [B, N_out, 2]
                classification: [B, N_out]
                positions: [B, N_out, n_classes]

        """
        res_in = tf.shape(image)[-3:-1]  # [H_in, W_in] (ignore batch and channel dimensions)
        res_out = tf.shape(offsets)[-3:-1]  # [H_out, W_out]
        scale = tf.cast((res_in / res_out)[::-1], offsets.dtype)
        # TODO: we need a correction factor here if image is YUYV->YUV converted
        pixels = tf.cast(
            tf.stack(tf.meshgrid(tf.range(res_out[1]), tf.range(res_out[0])), axis=-1),
            offsets.dtype,
        )

        # TODO: maybe do something about the shape here?
        coords = tf.reshape(
            (offsets + pixels) * scale, (-1, res_out[0] * res_out[1], 2)
        )  # Per cell one coordinate pair
        logits = tf.reshape(logits, (-1, tf.reduce_prod(res_out)))

        # Gather n_candidates coordinates from the coordinate list
        patch_indices = sampler(logits, training=training)  # [B, N_out]
        coords = tf.gather(coords, patch_indices, batch_dims=1)  # [B, N_out, 2]
        (
            patches,
            masks,
            boxes,
            coords,
            intrinsics,
        ) = extractor(
            image, coords, camera, intrinsics, training=training
        )  # [B, N_out, H_out, W_out, C], [B, N_out]

        classification, offsets = classifier(
            [
                tf.reshape(
                    patches,
                    (
                        tf.shape(intrinsics)[0] * sampler.n_sample,
                        *self.patch_size,
                        self.patch_channels,
                    ),
                )
            ]
        )  # + meta + context

        classification = tf.reshape(classification, (tf.shape(intrinsics)[0], sampler.n_sample))
        boxes = tf.reshape(boxes, (tf.shape(intrinsics)[0], sampler.n_sample, 4))

        positions = coords + tf.reshape(
            offsets, (tf.shape(intrinsics)[0], sampler.n_sample, 2)
        )  # TODO: stop gradient for coords?

        return {
            "patches": patches,
            "patch_indices": patch_indices,
            "masks": masks,
            "boxes": boxes,
            "coords": coords,
            "logits": logits,
            "classification": classification,
            "positions": positions,
        }
