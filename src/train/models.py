import tensorflow as tf

from util import dataset as u_dataset

from .layers import PatchExtractor, PatchSampler


def get_encoder(height, width, category_names, n_context):
    image = tf.keras.layers.Input((height, width, 4))
    # TODO: input [B, H, W/2, 4] (treat each YUYV tuple as a pixel)
    x = image

    # 480x320x4

    x = tf.keras.layers.Conv2D(32, 3, strides=(2, 1), padding="same", use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization(scale=False)(x)
    x = tf.keras.layers.ReLU(6.0)(x)

    # 240x320x32

    # ires-block(16, expansion=1)

    # 240x320x16

    # ires-block(24, stride=2, expansion=6)
    # ires-block(24, stride=1, expansion=6)

    x = tf.keras.layers.Conv2D(32, 3, strides=(2, 2), padding="same", use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization(scale=False)(x)
    x = tf.keras.layers.ReLU(6.0)(x)

    # 120x160x24

    # ires-block(32, stride=2, expansion=6)
    # ires-block(32, stride=1, expansion=6)
    # ires-block(32, stride=1, expansion=6)

    x = tf.keras.layers.Conv2D(32, 3, strides=(2, 2), padding="same", use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization(scale=False)(x)
    x = tf.keras.layers.ReLU(6.0)(x)

    # 60x80x32

    # ires-block(64, stride=2, expansion=6)
    # ires-block(64, stride=1, expansion=6)
    # ires-block(64, stride=1, expansion=6)
    # ires-block(64, stride=1, expansion=6)

    x = tf.keras.layers.Conv2D(32, 3, strides=(2, 2), padding="same", use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization(scale=False)(x)
    x = tf.keras.layers.ReLU(6.0)(x)

    # 30x40x64

    # ires-block(96, stride=1, expansion=6)
    # ires-block(96, stride=1, expansion=6)
    # ires-block(96, stride=1, expansion=6)

    # 30x40x96

    # ires-block(160, stride=2, expansion=6)
    # ires-block(160, stride=1, expansion=6)
    # ires-block(160, stride=1, expansion=6)

    x = tf.keras.layers.Conv2D(32, 3, strides=(2, 2), padding="same", use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization(scale=False)(x)
    x = tf.keras.layers.ReLU(6.0)(x)

    # 15x20x160

    # ires-block(320, stride=1, expansion=6)

    output = []
    for name in category_names:
        # TODO: some activated stuff here?
        offset = tf.keras.layers.Conv2D(2, 1)(x)

        x = tf.keras.layers.Conv2D(1, 1)(x)
        interest = tf.keras.layers.Activation("sigmoid")(x)

        output += [tf.keras.layers.Concatenate(name=name)([offset, interest])]

    if n_context > 0:
        context = tf.keras.layers.Conv2D(n_context, 1, name="context")(x)
        output += [context]

    return tf.keras.Model(
        image, output, name="encoder"
    )  # input: image, output: [offset, interest] for each category + context


def get_patch_classifier(patch_size, channels_in, n_meta, n_context, n_classes, with_offset=True):
    image = tf.keras.layers.Input((*patch_size, channels_in))
    inputs = [image]

    if n_meta > 0:
        meta = tf.keras.layers.Input((n_meta,))
        inputs += [meta]

    if n_context > 0:
        context = tf.keras.layers.Input((n_context,))
        inputs += [n_context]

    x = tf.keras.layers.Flatten()(image)
    if n_meta > 0:
        x = tf.keras.layers.Concatenate()([image, meta])
    x = tf.keras.layers.Dense(32)(x)
    x = tf.keras.layers.ReLU(6.0)(x)
    if n_context > 0:
        x = tf.keras.layers.Concatenate()([image, context])
    x = tf.keras.layers.Dense(32)(x)
    x = tf.keras.layers.ReLU(6.0)(x)

    if n_classes < 2:
        x = tf.keras.layers.Dense(1)(x)
        out = tf.keras.layers.Activation("sigmoid")(x)
    else:
        tf.keras.layers.Dense(n_classes + 1)(x)  # + 1 for the background class
        out = tf.keras.layers.Activation("softmax")(x)

    if with_offset:
        offset = tf.keras.layers.Dense(2)(x)
        out = [out, offset]

    return tf.keras.Model(inputs, out, name="classifier")


class FullModel(tf.keras.Model):
    def __init__(self, height, width):
        super().__init__()  # Subclass of the Model class
        # Size of context vector
        self.patch_size = (32, 32)
        self.n_context = 0
        self.n_meta = 0
        self.categories = {
            "ball": {
                "object_size": 0.175,
                "object_height": 0.05,
                "n_classes": 1,
                "n_candidates": 5,
            },
            "penaltyMark": {
                "object_size": 0.175,
                "object_height": 0.0,
                "n_classes": 1,
                "n_candidates": 5,
            },
            # "field": {
            #     "object_size": 0.2,
            #     "n_classes": 6,  # penalty mark, center mark, X intersection, L intersection, T intersection, goal post
            #     "n_candidates": 7,
            # },
            # "player":
            #     "object_size": 0.5,
            #     "n_classes": 1,
            # },
        }
        for _, value in self.categories.items():
            value["sampler"] = PatchSampler(
                value["n_candidates"]
            )  # The patch sampler for the category with a fixed number of candidates
            value["extractor"] = PatchExtractor(
                object_size=value["object_size"], object_height=value.get("object_height", 0)
            )  # The patch extractor for the category with the fixed object parameters
            value["classifier"] = get_patch_classifier(
                self.patch_size, 3, self.n_meta, self.n_context, value["n_classes"]
            )  # The patch classifier for the category with the fixed number of classes
        # self.encoder is a Model
        self.encoder = get_encoder(height, width, self.categories.keys(), self.n_context)

    def encoder_loss(self, batch_data, maps):
        # Compute Binary Cross Entropy
        element_wise_bce = -(
            batch_data["object_mask"] * tf.math.log(maps[..., 2])
            + (1.0 - batch_data["object_mask"]) * tf.math.log(1.0 - maps[..., 2])
        )
        element_wise_bce_multiplied = tf.multiply(element_wise_bce, batch_data["loss_mask"])
        bce = tf.reduce_sum(element_wise_bce_multiplied)

        # Compute MSE
        squared_error = tf.keras.losses.MeanSquaredError(reduction="none")(
            y_true=batch_data["offset_mask"], y_pred=maps[..., :2]
        )
        squared_error_multiplied = tf.multiply(squared_error, batch_data["object_mask"])

        mse = tf.reduce_mean(squared_error_multiplied) * 10000

        # tf.print("Squared Error: ", tf.shape(squared_error))
        # tf.print("Squared Error multiplied: ", tf.shape(squared_error_multiplied))
        # tf.print("MSE: ", tf.shape(mse))
        # Total loss
        loss = tf.add(bce, mse)

        return {"loss": loss, "mse": mse, "bce": bce}

    def classifier_loss(self, batch_data, results):
        # TODO: implement solution for multi-class problems with categorical crossentropy (like line crossings)
        # Compute BinaryCrossEntropy / CategoricalCrossEntropy
        y_pred = results["classification"]  # [B, n_candidates, 1]
        coords_pred = results["coords"]  # [B, n_candidates, 2]
        # tf.print("Coords Pred: ", tf.shape(coords_pred))
        coords_true = tf.expand_dims(
            u_dataset.get_coords_from_offsets(batch_data["offset_mask"]), axis=1
        )  # [B, 1, 2]

        # Match coords_pred with coords_true with a threshold and get one pair.
        y_true = tf.zeros_like(y_pred, dtype=tf.float32)

        # tf.print("Coords True: ", tf.shape(coords_true))
        bce = tf.keras.losses.BinaryCrossentropy(from_logits=True, name="classifier_bce")(
            y_true, y_pred
        )
        # tf.print("y_pred: ", tf.shape(y_pred))
        # tf.print("y_true: ", tf.shape(y_true))
        # tf.print("Classifier BCE: ", tf.shape(bce))

        # Compute MeanSquaredError
        positions_pred = results[
            "positions"
        ]  # coords that were predicted by the encoder corrected with offset from classifier

        # tf.print("positions pred", tf.shape(positions_pred))
        mse = tf.keras.losses.MeanSquaredError(name="classifier_mse")(coords_true, positions_pred)

        loss = bce + mse

        return {"loss": loss, "mse": mse, "bce": bce}

    def train_step(self, batch_data):
        with tf.GradientTape() as tape:
            results, maps = self(
                (batch_data["image"], batch_data["camera"], batch_data["intrinsics"]), training=True
            )  # calls call()
            encoder_losses = {
                key: self.encoder_loss(batch_data[key], maps[key]) for key in self.categories
            }
            classifier_losses = {
                key: self.classifier_loss(batch_data[key], results=value)
                for key, value in results.items()
            }  # (loss, mse, bce) for each category
            encoder_loss = tf.reduce_sum([value["loss"] for value in encoder_losses.values()])
            encoder_bce = tf.reduce_sum([value["bce"] for value in encoder_losses.values()])
            encoder_mse = tf.reduce_sum([value["mse"] for value in encoder_losses.values()])
            classifier_loss = tf.reduce_sum([value["loss"] for value in classifier_losses.values()])
            classifier_bce = tf.reduce_sum([value["bce"] for value in classifier_losses.values()])
            classifier_mse = tf.reduce_sum([value["bce"] for value in classifier_losses.values()])

            print("Encoder Loss: ", encoder_loss)
            print("Classifier Loss: ", classifier_loss)

            loss = encoder_loss + classifier_loss

        # Compute gradients
        gradients = tape.gradient(loss, self.trainable_variables)

        # Update weights
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables, strict=True))

        return {"loss": loss, "encoder_bce": encoder_bce, "encoder_mse": encoder_mse}

    def test_step(self, batch_data):
        # TODO: remove duplicate code
        results, maps = self(
            (batch_data["image"], batch_data["camera"], batch_data["intrinsics"]), training=True
        )  # calls call()

        encoder_losses = {
            key: self.encoder_loss(batch_data[key], maps[key]) for key in self.categories
        }

        encoder_loss = tf.reduce_sum([value["loss"] for value in encoder_losses.values()])
        encoder_bce = tf.reduce_sum([value["bce"] for value in encoder_losses.values()])
        encoder_mse = tf.reduce_sum([value["mse"] for value in encoder_losses.values()])

        loss = encoder_loss

        # loss, mse, bce = self.encoder_loss(batch_data, maps)

        return {"loss": loss, "encoder_bce": encoder_bce, "encoder_mse": encoder_mse}

    def call(self, batch_data, training=None):
        """
        :param image: The full resolution image from which the patches are extracted.
            [B, H_in, W_in, C]
        :param camera: The pose of the camera, represented as (roll angle, pitch angle, height above ground)
            [B, 3]
        :param intrinsics: The intrisics of the camera, represented as (cx, cy, fx, fy)
            [B, 4]
        :return: Per category x,y,p tuples
            {key: [B, N_out, 2 + n_classes + 1?]}
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
        # Convert YUYV to YUV
        # Stack the the image along the channel dimensions in order to go from YUYV to Y1UVY2UV. Then reshape it to [B, H_in, W_in/2, 3]
        image_yuv_stack = tf.stack(
            [
                image[..., 0],
                image[..., 1],
                image[..., 3],
                image[..., 2],
                image[..., 1],
                image[..., 3],
            ],
            axis=-1,
        )
        image_yuv = tf.reshape(
            image_yuv_stack, (tf.shape(image)[0], tf.shape(image)[1], tf.shape(image)[2] * 2, 3)
        )  # [B, H_in, W_in*2, 3]

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

        if training:
            # results: patches, masks
            # maps: [B, H_out, W_out, 3] (offsets, logits) or [B, H_out, W_out, n_context]
            return results, maps

        return results

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
        :param image: The full resolution image from which the patches are extracted.
            [B, H_in, W_in, C]
        :param camera: The pose of the camera, represented as (roll angle, pitch angle, height above ground)
            [B, 3]
        :param intrinsics: The intrisics of the camera, represented as (cx, cy, fx, fy)
            [B, 4]
        :param logits: The logits for the category.
            [B, H_out, W_out]
        :param offsets: The offsets for the category. Relative to the upper left corner of the patch.
            [B, H_out, W_out, 2]
        :param sampler: The patch sampler for the category with a fixed number of candidates
        :param extractor: The patch extractor for the category with the fixed object parameters
        :param classifier: The patch classifier for the category with the fixed number of classes
        :param training: Whether the model is in training mode or not.
        :return:
            [B, N_out, n_classes]
            [B, N_out, 2]
            [B, N_out]
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
        patch_indices = sampler(logits)  # [B, N_out]
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
            tf.reshape(patches, (tf.shape(intrinsics)[0] * sampler.n_sample, *self.patch_size, 3))
        )  # + meta + context

        classification = tf.reshape(classification, (tf.shape(intrinsics)[0], sampler.n_sample))

        positions = coords + tf.reshape(
            offsets, (tf.shape(intrinsics)[0], sampler.n_sample, 2)
        )  # TODO: stop gradient for coords?
        return {
            "patches": patches,
            "masks": masks,
            "boxes": boxes,
            "coords": coords,
            "classification": classification,
            "positions": positions,
        }
