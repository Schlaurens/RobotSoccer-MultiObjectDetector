import os

import tensorflow as tf

from training import classifier_architectures as u_classifiers
from training import encoder_architectures as u_encoders
from util import dataset as u_dataset
from util import image as u_image
from util import keypoint as u_keypoint
from util.layers import IresBlock, Normalization, PatchExtractor, PatchSampler


class FullModel(tf.keras.Model):
    def __init__(
        self,
        encoder_architecture: str,
        classifier_architecture: str,
        height: int,
        width: int,
        encoder_channels: int = 4,
        cell_dims: list[int] = None,
        n_context: int = 0,
        train_encoder: bool = True,
        train_classifier: bool = True,
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
        self.train_encoder = train_encoder
        self.encoder_use_batch_norm = encoder_use_batch_norm
        self.encoder_channels = encoder_channels

        # Classifier config
        self.classifier_architecture = classifier_architecture
        self.patch_size = [32, 32]
        self.patch_channels = 1
        self.train_classifier = train_classifier
        self.n_meta = n_meta
        self.classifier_offsets = classifier_offsets
        self.classifier_use_batch_norm = classifier_use_batch_norm

        self.full_image_size = tf.constant(
            [self.image_height, self.image_width], dtype=tf.float32
        )  # constructor input image_width is halved due to YUYV

        self.dataset_config = u_dataset.DatasetConfig(
            (
                self.image_height,
                self.image_width * 2 if encoder_channels == 4 else self.image_width,
            ),
            cell_dims=cell_dims,
        )
        self.dataset_utils = u_dataset.DatasetUtils(self.dataset_config)

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
        if self.train_classifier:
            for _, value in self.categories.items():
                value["sampler"] = PatchSampler(
                    value["n_candidates"],
                    value["max_distance"],
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
            self.encoder_channels,
            self.categories.keys(),
            self.n_context,
            self.encoder_use_batch_norm,
        )

        object.__setattr__(self, "_test_metrics", {})  # not tracked by Keras
        object.__setattr__(self, "_train_metrics", {})  # not tracked by Keras

    def reset_metrics(self):
        super().reset_metrics()
        for m in self._train_metrics.values():
            m.reset_state()
        for m in self._test_metrics.values():
            m.reset_state()

    @property
    def metrics(self):
        metrics = []
        if self._train_metrics:
            metrics += list(self._train_metrics.values())
        if self._test_metrics:
            metrics += list(self._test_metrics.values())
        return metrics

    def encoder_loss(self, batch_data, interest, offsets, n_candidates):
        B = tf.shape(interest)[0]  # Batch Size

        # Compute Binary Cross Entropy

        # Numeric stabilizer
        epsilon = tf.constant(1e-7, dtype=tf.float32)
        element_wise_bce = -(
            batch_data["object_mask"] * tf.math.log(interest + epsilon)
            + (1.0 - batch_data["object_mask"]) * tf.math.log(1.0 - interest + epsilon)
        )  # (B, 15, 20)

        tf.debugging.assert_non_negative(interest, "interest is negative")
        tf.debugging.assert_all_finite(tf.math.log(epsilon), "tf.math.log(epsilon)")
        tf.debugging.assert_all_finite(
            tf.math.log(interest + epsilon), "tf.math.log(interest + epsilon)"
        )
        tf.debugging.assert_none_equal(1.0 - interest + epsilon, 0.0, message="equals 0.0")
        tf.debugging.assert_all_finite(
            tf.math.log(1.0 - interest + epsilon), "tf.math.log(1.0 - interest + epsilon"
        )
        tf.debugging.assert_all_finite(batch_data["object_mask"], "batch_data[object_mask]")
        tf.debugging.assert_all_finite(
            (1.0 - batch_data["object_mask"]), "(1.0 - batch_data[object_mask])"
        )
        tf.debugging.assert_all_finite(element_wise_bce, "element_wise_bce")

        element_wise_bce_multiplied = tf.multiply(
            element_wise_bce, batch_data["loss_mask"]
        )  # (B, 15, 20)
        bce_batched = tf.reduce_sum(element_wise_bce_multiplied, axis=[1, 2])  # (B, )

        tf.debugging.assert_all_finite(batch_data["loss_mask"], "batch_data[loss_mask]")
        tf.debugging.assert_all_finite(element_wise_bce_multiplied, "element_wise_bce_multiplied")

        # ==============
        # == Recall@k ==
        # ==============

        k = n_candidates

        # Flatten spatial dims for each sample
        flat_interest = tf.reshape(interest, [B, -1])  # (B, 300)
        flat_object_mask = tf.reshape(batch_data["object_mask"], [B, -1])  # (B, 300)

        top_k_indices = tf.math.top_k(flat_interest, k=k).indices  # (B, k)

        # Build top-k mask by scattering 1s at top-k indices
        batch_indices = tf.repeat(tf.range(B)[:, tf.newaxis], k, axis=1)  # (B, k)

        # Which samples, which cell index
        scatter_indices = tf.stack([batch_indices, top_k_indices], axis=-1)  # (B, k, 2)

        # Places a 1 at each index that is in the top k indices.
        top_k_mask = tf.tensor_scatter_nd_update(
            tf.zeros([B, tf.reduce_prod(self.dataset_config.output_dims)]),  # (B, 300)
            tf.reshape(scatter_indices, [-1, 2]),  # (B * k, 2)
            tf.ones([B * k]),
        )  # (B, 300)

        # Check overlap with object_mask
        tp = tf.reduce_sum(top_k_mask * flat_object_mask)  # ( )
        num_objects = tf.reduce_sum(flat_object_mask)  # ( )
        recall_at_k = tp / tf.maximum(num_objects, 1e-8)  # ( )

        class_distr = tf.cast(tp, tf.int32) / (B * n_candidates)  # ( )

        # Compute MSE
        squared_error = tf.square(
            tf.norm(batch_data["offset_mask"] - offsets, axis=-1)
        )  # (B, 15, 20)

        squared_error_multiplied = tf.multiply(
            squared_error, batch_data["object_mask"]
        )  # (B, 15, 20)

        mse_batched = tf.reduce_mean(squared_error_multiplied, axis=[1, 2]) * 10000  # (B, )

        gt_coord_mask = self.dataset_utils.get_coordinate_mask(batch_data["offset_mask"])
        pred_coord_mask = self.dataset_utils.get_coordinate_mask(offsets)

        error_in_pixels = tf.norm(
            (gt_coord_mask - pred_coord_mask) / self.dataset_config.image_res_scale[::-1], axis=-1
        )  # (B, 15, 20)

        error_in_pixels_multiplied = error_in_pixels * batch_data["object_mask"]  # (B, 15, 20)

        # The RMSE Metric (not used int the loss)
        sum_of_error_multiplied = tf.reduce_sum(error_in_pixels_multiplied)  # Shape: ( )
        num_ones = tf.maximum(tf.reduce_sum(batch_data["object_mask"]), 1e-8)  # Shape: ( )
        mae_metric = sum_of_error_multiplied / num_ones  # Shape: ( )

        tf.debugging.assert_all_finite(bce_batched, "Encoder BCE")
        tf.debugging.assert_all_finite(mae_metric, "Encoder MAE Metric")

        # Total loss
        loss_batched = bce_batched + mse_batched  # (B, )
        loss = tf.reduce_mean(loss_batched)  # Shape: ()
        bce = tf.reduce_mean(bce_batched)  # Shape: ()
        mse = tf.reduce_mean(mse_batched)  # Shape: ()

        return {
            "loss": loss,
            "mse": mse,
            "mae": mae_metric,
            "bce": bce,
            "recall_at_k": recall_at_k,
            "class_distribution": class_distr,
        }

    def classifier_loss(self, batch_data, results, object_name):
        # Compute MSE
        boxes = results["boxes"]  # (B, N, 4)
        coords_pred = results["positions"]  # (B, N, 2)

        # Reshape is needed for tf.gather to work.
        coord_mask = tf.reshape(
            self.dataset_utils.get_coordinate_mask(batch_data["offset_mask"]),
            (-1, self.dataset_config.output_dims[0] * self.dataset_config.output_dims[1], 2),
        )  # (B, 15 * 20, 2)

        # Get the coords of the cells of which patches were extracted.
        coords_true_of_patches = tf.gather(
            coord_mask, results["patch_indices"], batch_dims=1
        )  # (B, N, 2)

        # Normalize coords to the image dimensions. Because the boxes coords are also normalized.
        # Switch axes of full_image_size because: coords_true (x, y), full_image_size (y, x)
        coords_true_normalized = (
            coords_true_of_patches / self.full_image_size[::-1][tf.newaxis, :]
        )  # (B, N, 2)

        # Check whether the object coordinates are inside their respective patches.
        are_coords_true_inside_patch = u_keypoint.are_coords_in_patch(
            coords_true_normalized, boxes
        )  # (B, N)

        # =========================
        # == Classification Loss ==
        # =========================

        # The Encoder predictions for each of the patches
        encoder_predictions = tf.gather(
            results["logits"], results["patch_indices"], batch_dims=1
        )  # (B, N)

        # Compute BinaryCrossEntropy / CategoricalCrossEntropy
        y_pred = results["classification"]  # (B, N) | (B, N, N_O) depending on category type

        # Categories with more than two classes use CCE
        if object_name == u_dataset.CategoryNames.INTERSECTIONS.value:
            y_true = tf.one_hot(
                tf.cast(
                    self.dataset_utils.get_groundtruth_class_of_patches(
                        results,
                        batch_data,
                        padding=self.categories[object_name]["padding"],
                        batch_dims=1,
                    ),
                    tf.int32,
                ),
                len(u_dataset.IntersectionType),
                axis=-1,
            )  # (B, N, num_classes)

            cross_entropy_batched = tf.keras.losses.CategoricalCrossentropy(
                from_logits=False, reduction=None, name="classifier_cce"
            )(y_true, y_pred)  # (B, N)

            # Binary mask that is True at every sample index that should NOT be ignored. A sample should be ignored if every cell in their loss_mask is set to 0.
            use_sample = tf.expand_dims(
                tf.cast(
                    tf.reduce_any(tf.cast(batch_data["loss_mask"], tf.bool), axis=[1, 2]),
                    tf.float32,
                ),
                axis=-1,
            )  # (B, 1)

            # If a sample should be ignored the cross_entropy of that sample is set to a constant 0 which is not differentiable.
            # Also multiply the cross_entropy with the output of the encoder to weed out patches the encoder is not confident in.
            cross_entropy_multiplied = (
                cross_entropy_batched * use_sample * tf.stop_gradient(encoder_predictions)
            )  # (B, N)
            cross_entropy = tf.reduce_sum(
                tf.reduce_mean(cross_entropy_multiplied, axis=-1)
            )  # Shape: ()

            # Get probability of the predicted class for each candidate.
            # error_factor = tf.math.reduce_max(y_pred, axis=-1)  # (B, N)

            # Get combined probability of the positive classes.
            error_factor = 1 - y_pred[..., 0]  # (B, N)

        elif object_name in [
            u_dataset.CategoryNames.BALL.value,
            u_dataset.CategoryNames.PENALTYMARK.value,
        ]:  # For binary categories use BCE
            y_true = are_coords_true_inside_patch  # (B, N)

            cross_entropy_batched = tf.keras.losses.BinaryCrossentropy(
                from_logits=False, reduction=None, name="classifier_bce"
            )(y_true, y_pred)  # Shape: (B, N)

            cross_entropy_multiplied = cross_entropy_batched * tf.stop_gradient(
                tf.expand_dims(encoder_predictions, axis=-1)
            )

            cross_entropy = tf.reduce_sum(tf.reduce_mean(cross_entropy_multiplied, axis=-1))

            error_factor = tf.squeeze(y_pred, axis=-1)  # (B, N)
        else:
            raise ValueError("Invalid object_name.")

        error_factor = tf.where(
            tf.math.is_finite(error_factor),
            error_factor,
            tf.zeros_like(error_factor),
        )

        # ==============================
        # == Euclidean Error (Metric) ==
        # ==============================
        # Scale the norm with the full image resolution to make comparison with other resolutions possible.
        euclidean_error = tf.stop_gradient(
            tf.where(
                are_coords_true_inside_patch,
                tf.norm(
                    (coords_pred - coords_true_of_patches)
                    / self.dataset_config.image_res_scale[::-1],
                    axis=-1,
                ),
                0.0,
            )
        )  # (B, N)
        num_true_patches = tf.maximum(
            tf.reduce_sum(tf.cast(are_coords_true_inside_patch, tf.float32)), 1e-8
        )  # Shape: ( )
        # Ignore the patches in the mean that do not contain gt coords
        sum_euc_error = tf.reduce_sum(euclidean_error)  # Shape: ( )
        mean_euclidean_error = sum_euc_error / num_true_patches  # Shape: ( )

        # ========================
        # == Mean Squared Error ==
        # ========================

        # If coords_true are inside the patch always calculate the MSE. Else the classifier's offset predictions are useless and should be ignored. Assign a constant max error that has gradient of zero. Also if there are no coords_true because the sample was ignored, the results have no impact on the loss.
        squared_error = tf.where(
            are_coords_true_inside_patch,
            tf.reduce_sum(tf.square(coords_pred - coords_true_of_patches), axis=-1)
            * tf.stop_gradient(results["distances"]),
            0.0,
        )  # (B, N)
        # If the classifier thinks that there is no object in the image, this error has a smaller contribution to the loss
        squared_error_multiplied = squared_error * tf.stop_gradient(error_factor)  # (B, N)

        # Ignore the patches in the mean that do not contain gt coords
        sum_of_squared_errors = tf.reduce_sum(squared_error_multiplied)  # Shape: ( )
        mse = sum_of_squared_errors / num_true_patches  # Shape: ()

        tf.debugging.assert_all_finite(mse, "Classifier MSE")
        tf.debugging.assert_all_finite(cross_entropy, "Classifier CE")
        tf.debugging.assert_all_finite(mean_euclidean_error, "Classifier euclidean error")

        loss = cross_entropy + mse  # Shape: ()

        return {
            "loss": loss,
            "mse": mse,
            "euc_error": mean_euclidean_error,
            "ce": cross_entropy,
        }

    def _calculate_losses(self, batch_data, results, maps):
        result = {}

        # =========================
        # == Handle Encoder Loss ==
        # =========================
        if self.train_encoder:
            encoder_losses = {
                key: self.encoder_loss(
                    batch_data[key],
                    interest=tf.squeeze(maps[f"{key}_interest"], -1),
                    offsets=maps[f"{key}_offsets"],
                    n_candidates=self.categories[key]["n_candidates"],
                )
                for key in self.categories
            }

            result["encoder_loss"] = tf.reduce_sum(
                [value["loss"] for value in encoder_losses.values()]
            )

            for key in self.categories:
                result[f"encoder_bce_{key}"] = encoder_losses[key]["bce"]
                result[f"encoder_recall_at_k_{key}"] = encoder_losses[key]["recall_at_k"]
                result[f"encoder_class_distribution_{key}"] = encoder_losses[key][
                    "class_distribution"
                ]
                result[f"encoder_mse_{key}"] = encoder_losses[key]["mse"]
                result[f"encoder_mae_{key}"] = encoder_losses[key]["mae"]

        # ============================
        # == Handle Classifier Loss ==
        # ============================
        if self.train_classifier:
            classifier_losses = {
                key: self.classifier_loss(
                    batch_data[key],
                    results=value,
                    object_name=key,
                )
                for key, value in results.items()
            }

            result["classifier_loss"] = tf.reduce_sum(
                [value["loss"] for value in classifier_losses.values()]
            )

            for key in self.categories:
                result[f"classifier_ce_{key}"] = classifier_losses[key]["ce"]
                result[f"classifier_mse_{key}"] = classifier_losses[key]["mse"]
                result[f"classifier_euc_error_{key}"] = classifier_losses[key]["euc_error"]

        return result

    def train_step(self, batch_data):
        with tf.GradientTape() as tape:
            outputs = self(batch_data, training=True)  # calls call()

            losses = self._calculate_losses(batch_data, outputs["results"], outputs["maps"])

            total_loss = (
                losses.get("encoder_loss", 0.0) * self.train_encoder
                + losses.get("classifier_loss", 0.0) * self.train_classifier
            )

        # Compute gradients
        gradients = tape.gradient(total_loss, self.trainable_variables)

        # Update weights
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables, strict=True))

        losses["total_loss"] = total_loss

        # Initialize metrics lazily on first call
        if not self._train_metrics:
            object.__setattr__(
                self, "_train_metrics", {name: tf.keras.metrics.Mean(name=name) for name in losses}
            )

        # Update metric objects
        for key, value in losses.items():
            self._train_metrics[key].update_state(value)

        return {k: m.result() for k, m in self._train_metrics.items()}

    def test_step(self, batch_data):
        outputs = self(batch_data, training=False)  # calls call()
        losses = self._calculate_losses(batch_data, outputs["results"], outputs["maps"])

        total_loss = (
            losses.get("encoder_loss", 0.0) * self.train_encoder
            + losses.get("classifier_loss", 0.0) * self.train_classifier
        )

        losses["total_loss"] = total_loss

        # Initialize metrics lazily on first call
        if not self._test_metrics:
            object.__setattr__(
                self, "_test_metrics", {name: tf.keras.metrics.Mean(name=name) for name in losses}
            )

        # Update metric objects
        for key, value in losses.items():
            self._test_metrics[key].update_state(value)

        return {k: m.result() for k, m in self._test_metrics.items()}

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
        os.makedirs(os.path.join(filepath, f"{filename}", "encoder"), exist_ok=True)

        # Save the encoder
        encoder_path = os.path.join(filepath, f"{filename}", "encoder", f"{filename}")

        self.encoder.save(encoder_path + ".keras", overwrite)
        self.encoder.save(encoder_path + ".h5", overwrite)
        self.encoder.export(encoder_path + ".onnx", format="onnx")

        if verbose:
            print("Encoder saved!")

        if self.train_classifier:
            # Save the classifier of each category
            for name, value in self.categories.items():
                # Create directory if it does not exist.
                os.makedirs(
                    os.path.join(filepath, f"{filename}", "classifier", name), exist_ok=True
                )
                classifier_path = os.path.join(
                    filepath, f"{filename}", "classifier", name, f"{filename}"
                )

                value["classifier"].save(classifier_path + ".keras", overwrite)
                value["classifier"].save(classifier_path + ".h5", overwrite)
                value["classifier"].export(classifier_path + ".onnx", format="onnx")

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
        filepath: str,
        filename: str,
        input_dims: list[int] | tuple[int, int],
        encoder_channels: int = 4,
        cell_dims: list[int] | tuple[int, int] = None,
        n_context: int = 0,
        train_encoder: bool = True,
        train_classifier: bool = True,
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
            encoder_only: True if ONLY an existing encoder is loaded. Defaults to False.
            verbose: Print status messages that describe the status of the loading process. Defaults to False.

        Returns:
            A tf.keras.Model with the loaded models.
        """
        # Rebuild model
        model = cls(
            encoder_architecture,
            classifier_architecture,
            input_dims[0],
            input_dims[1],
            encoder_channels,
            cell_dims,
            n_context,
            train_encoder,
            train_classifier,
            classifier_offsets,
            n_meta,
            encoder_use_batch_norm,
            classifier_use_batch_norm,
            categories_config,
        )

        # Load the encoder
        encoder = tf.keras.models.load_model(
            os.path.join(filepath, "encoder", f"{filename}.keras"),
            custom_objects={"IresBlock": IresBlock, "Normalization": Normalization},
        )
        model.encoder = encoder

        if verbose:
            print("Encoder loaded!")

        # Load each classifier
        if train_classifier and not encoder_only:
            for name, value in model.categories.items():
                try:
                    classifier_path = os.path.join(
                        filepath, "classifier", name, f"{filename}.keras"
                    )
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

        # model.encoder.trainable = False
        # print(model.encoder.trainable_variables)

        model.compile(optimizer=tf.keras.optimizers.Adam(), jit_compile=False)

        if verbose:
            print("Train Encoder = ", train_encoder)
            print("Train Classifier = ", train_classifier)
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

        full_image = batch_data["image"]
        camera = batch_data["camera"]
        intrinsics = batch_data["intrinsics"]

        image_grayscale = u_image.convert_yuyv_to_yuv(full_image)[..., 0:1]  # (B, W_in, H_in, 1)

        if self.train_encoder:
            full_image = image_grayscale if self.encoder_channels == 1 else full_image
            maps = self.encoder(full_image, training=training)  # Run the encoder on the image

            if isinstance(maps, list):  # If there is a context vector
                maps = dict(zip(self.encoder.output_names, maps, strict=True))
            else:
                maps = {
                    self.encoder.output_names[0]: maps
                }  # [B, H_out, W_out, 3] Encoder results for the first category
        else:
            # Use cached encoder outputs to avoid inference
            maps = {
                k.removeprefix("encoder_"): v
                for k, v in batch_data.items()  # batch_data ist hier das ganze Dict
                if k.startswith("encoder_")
            }

        if not self.train_classifier:
            return {"results": None, "maps": maps}

        # Convert image to grayscale if only one channel is requested.
        if self.patch_channels == 1:
            full_image = image_grayscale

        context = None
        if "context" in maps:
            context = maps["context"]  # (B, H_out, W_out, n_context)

        results = {
            key: self._handle_category(
                full_image,
                camera,
                intrinsics,
                value["object_height"],
                maps[f"{key}_interest"],
                maps[f"{key}_offsets"],
                context,
                value["n_classes"],
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
        object_height,
        logits,
        offsets,
        context,
        n_classes,
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
            offsets: The offsets for the category. Relative to the middle of the patch. [B, H_out, W_out, 2]
            context: The context vector which is part of the encoder output. It encodes information about the whole image that might help the classifier.
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

        pixels = tf.cast(
            tf.stack(tf.meshgrid(tf.range(res_out[1]), tf.range(res_out[0])), axis=-1),
            offsets.dtype,
        )

        coords = tf.reshape(
            (offsets + pixels + 0.5) * scale, (-1, tf.reduce_prod(res_out), 2)
        )  # Per cell one coordinate pair. Coordinates from middle of cell, so add 0.5
        logits = tf.reshape(logits, (-1, tf.reduce_prod(res_out)))

        distance_mask = tf.reshape(
            self.dataset_utils.get_distance_mask_from_offsets(
                offsets, camera, intrinsics, object_height=object_height
            ),
            (-1, tf.reduce_prod(res_out)),
        )  # (B, H_out, W_out)

        # Gather n_candidates coordinates from the coordinate list
        patch_indices = sampler(logits, distance_mask, training=training)  # [B, N_out]
        coords = tf.gather(coords, patch_indices, batch_dims=1)  # [B, N_out, 2]

        (patches, masks, boxes, intrinsics, distances_in_camera) = extractor(
            image, coords, camera, intrinsics, training=training
        )  # [B, N_out, H_out, W_out, C], [B, N_out]

        patches_reshaped = tf.reshape(
            tf.stop_gradient(patches),
            (
                tf.shape(intrinsics)[0] * sampler.n_sample,
                *self.patch_size,
                self.patch_channels,
            ),
        )  # (B * N_out, patch_size, patch_size, n_channels)

        classifier_inputs = [patches_reshaped]

        # Add distances to classifier_input
        if self.n_meta == 1:
            distances_flat = tf.reshape(
                distance_mask, (-1, tf.reduce_prod(res_out))
            )  # (B, H_out * W_out)
            distances_of_chosen_cells = tf.gather(
                distances_flat, patch_indices, batch_dims=1
            )  # (B, N)
            distances_reshaped = tf.reshape(
                tf.stop_gradient(distances_of_chosen_cells), [-1]
            )  # (B * N)
            classifier_inputs += [distances_reshaped]

        # Add context vector to classifier_inputs (if n_context > 0)
        if context is not None:
            context_flat = tf.reshape(
                context, (-1, tf.reduce_prod(res_out), self.n_context)
            )  # (B, H_out * W_out, n_context)

            context_of_chosen_cells = tf.gather(
                context_flat, patch_indices, batch_dims=1
            )  # (B, N, n_context)

            context_reshaped = tf.reshape(
                context_of_chosen_cells,
                (tf.shape(intrinsics)[0] * sampler.n_sample, self.n_context),
            )  # (B * N, n_context)
            classifier_inputs += [context_reshaped]

        classification, offsets = classifier(classifier_inputs, training=training)

        classification = tf.reshape(
            classification, (tf.shape(intrinsics)[0], sampler.n_sample, n_classes)
        )  # (B, N, n_classes)
        boxes = tf.reshape(boxes, (tf.shape(intrinsics)[0], sampler.n_sample, 4))  # (B, N, 4)

        positions = tf.stop_gradient(coords) + tf.reshape(
            offsets, (tf.shape(intrinsics)[0], sampler.n_sample, 2)
        )  # (B, N, 2)

        return {
            "patches": patches,
            "patch_indices": patch_indices,
            "masks": masks,
            "boxes": boxes,
            "coords": coords,
            "logits": logits,
            "classification": classification,
            "positions": positions,
            "distances": distances_in_camera,
        }
