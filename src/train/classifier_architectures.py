import tensorflow as tf

from .layers import IresBlock, Normalization


def get_classifier(
    classifier_architecture: str,
    patch_size: tuple[int] | list[int],
    channels_in: int,
    n_meta: int,
    n_context: int,
    n_classes: int,
    with_offset: bool = True,
    use_batch_norm: bool = False,
    **kwargs,
):
    """Return the specified classifier model

    Args:
        classifier_architecture: The name of the classifier architecture
        patch_size: The size of the patch in pixel
        channels_in: The amount of the channels of the patch
        n_meta: ...
        n_context: The size of the context vector.
        n_classes: The amount of the classes to predict
        with_offset: Whether the classifier should predict an offset.
        batch_norm: Whether to use batch normalization or instance normalization.

    Raises:
        ValueError: When the provided encoder architecture is unknown

    Returns:
        A tf.keras.Model with the provided architecture
    """
    if classifier_architecture == "classifier_inverted_residual_single_category":
        return _get_classifier_inverted_residual_single_category(
            patch_size,
            channels_in,
            n_meta,
            n_context,
            n_classes,
            with_offset,
            use_batch_norm,
        )
    if classifier_architecture == "classifier_inverted_residual_single_category_v2":
        return _get_classifier_inverted_residual_single_category_v2(
            patch_size,
            channels_in,
            n_meta,
            n_context,
            n_classes,
            with_offset,
            use_batch_norm,
        )
    if classifier_architecture == "classifier_single_category":
        return _get_classifier_single_category(
            patch_size,
            channels_in,
            n_meta,
            n_context,
            n_classes,
            with_offset,
            use_batch_norm,
        )
    if classifier_architecture == "classifier_single_category_v2":
        return _get_classifier_single_category_v2(
            patch_size,
            channels_in,
            n_meta,
            n_context,
            n_classes,
            with_offset,
            use_batch_norm,
        )
    if classifier_architecture == "classifier_ires_single_category":
        return _get_classifier_ires_single_category(
            patch_size,
            channels_in,
            n_meta,
            n_context,
            n_classes,
            with_offset,
            use_batch_norm,
        )
    if classifier_architecture == "classifier_ires_single_category_v2":
        return _get_classifier_ires_single_category_v2(
            patch_size,
            channels_in,
            n_meta,
            n_context,
            n_classes,
            with_offset,
            use_batch_norm,
        )
    if classifier_architecture == "classifier_ires_single_category_v3":
        return _get_classifier_ires_single_category_v3(
            patch_size,
            channels_in,
            n_meta,
            n_context,
            n_classes,
            with_offset,
            use_batch_norm,
        )
    if classifier_architecture == "classifier_ires_single_category_v4":
        return _get_classifier_ires_single_category_v4(
            patch_size,
            channels_in,
            n_meta,
            n_context,
            n_classes,
            with_offset,
            use_batch_norm,
        )
    if classifier_architecture == "classifier_ires_single_category_v5":
        return _get_classifier_ires_single_category_v5(
            patch_size,
            channels_in,
            n_meta,
            n_context,
            n_classes,
            with_offset,
            use_batch_norm,
        )
    else:
        raise ValueError(f"Unknown classifier name: {classifier_architecture}")


def _get_common_classifier_output(x, n_classes, with_offset, inputs):
    """Return the common output logic for every classifier architecture.

    Args:
        x: The tensor output of the hidden encoder layers
        n_classes: The amount of the classes to predict
        with_offset: Whether the classifier should predict an offset.
        inputs: The inputs of the classifier.

    Returns:
        A tf.keras.Model
    """
    if n_classes < 2:
        x = tf.keras.layers.Dense(1)(x)
        out = tf.keras.layers.Activation("sigmoid")(x)
    else:
        x = tf.keras.layers.Dense(n_classes)(x)
        out = tf.keras.layers.Activation("softmax")(x)

    if with_offset:
        offset = tf.keras.layers.Dense(2)(x)
        out = [out, offset]

    return tf.keras.Model(inputs, out, name="classifier")


# ========= Classifier Architectures =========


def _get_classifier_inverted_residual_single_category(
    patch_size: list[int],
    channels_in: int,
    n_meta: int,
    n_context: int,
    n_classes: int,
    with_offset: bool,
    use_batch_norm: bool,
):
    image = tf.keras.layers.Input((*patch_size, channels_in))
    inputs = [image]

    if n_meta > 0:
        meta = tf.keras.layers.Input((n_meta,))
        inputs += [meta]

    if n_context > 0:
        context = tf.keras.layers.Input((n_context,))
        inputs += [n_context]

    # x = tf.keras.layers.Flatten()(image)
    x = image
    if n_meta > 0:
        x = tf.keras.layers.Concatenate()([image, meta])

    x = tf.keras.layers.Conv2D(16, 3, padding="same", use_bias=False)(x)
    x = Normalization(use_batch_norm, scale=False, groups=-1)(x)
    x = tf.keras.layers.ReLU(6.0)(x)

    if n_context > 0:
        x = tf.keras.layers.Concatenate()([image, context])
    x = tf.keras.layers.Conv2D(32, 3, padding="same", use_bias=False)(x)
    x = Normalization(use_batch_norm, scale=False, groups=-1)(x)
    x = tf.keras.layers.ReLU(6.0)(x)

    x = tf.keras.layers.Conv2D(32, 3, padding="same", use_bias=False)(x)
    x = Normalization(use_batch_norm, scale=False, groups=-1)(x)
    x = tf.keras.layers.ReLU(6.0)(x)

    x = tf.keras.layers.Conv2D(16, 3, padding="same", use_bias=False)(x)
    x = Normalization(use_batch_norm, scale=False, groups=-1)(x)
    x = tf.keras.layers.ReLU(6.0)(x)

    x = tf.keras.layers.Flatten()(x)

    return _get_common_classifier_output(x, n_classes, with_offset, inputs)


def _get_classifier_inverted_residual_single_category_v2(
    patch_size: list[int],
    channels_in: int,
    n_meta: int,
    n_context: int,
    n_classes: int,
    with_offset: bool,
    use_batch_norm: bool,
):
    image = tf.keras.layers.Input((*patch_size, channels_in))
    inputs = [image]

    if n_meta > 0:
        meta = tf.keras.layers.Input((n_meta,))
        inputs += [meta]

    if n_context > 0:
        context = tf.keras.layers.Input((n_context,))
        inputs += [n_context]

    x = image
    if n_meta > 0:
        x = tf.keras.layers.Concatenate()([image, meta])

    x = IresBlock(8, use_batch_norm, stride=1, expansion=4)(x)
    x = IresBlock(16, use_batch_norm, stride=1, expansion=4)(x)
    x = IresBlock(32, use_batch_norm, stride=1, expansion=4)(x)
    x = IresBlock(64, use_batch_norm, stride=1, expansion=4)(x)

    x = tf.keras.layers.Flatten()(x)

    return _get_common_classifier_output(x, n_classes, with_offset, inputs)


def _get_classifier_single_category(
    patch_size: list[int],
    channels_in: int,
    n_meta: int,
    n_context: int,
    n_classes: int,
    with_offset: bool,
    use_batch_norm: bool,
):
    image = tf.keras.layers.Input((*patch_size, channels_in))
    inputs = [image]

    if n_meta > 0:
        meta = tf.keras.layers.Input((n_meta,))
        inputs += [meta]

    if n_context > 0:
        context = tf.keras.layers.Input((n_context,))
        inputs += [n_context]

    x = image
    if n_meta > 0:
        x = tf.keras.layers.Concatenate()([image, meta])

    x = tf.keras.layers.Conv2D(16, 3, padding="same", use_bias=False)(x)
    x = Normalization(batch_norm=True, scale=False)(x)
    x = tf.keras.layers.ReLU(6.0)(x)

    x = tf.keras.layers.Conv2D(16, 3, padding="same", use_bias=False)(x)
    x = Normalization(batch_norm=True, scale=False)(x)
    x = tf.keras.layers.ReLU(6.0)(x)

    x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), padding="same")(x)

    x = tf.keras.layers.Conv2D(16, 3, padding="same", use_bias=False)(x)
    x = Normalization(batch_norm=True, scale=False)(x)
    x = tf.keras.layers.ReLU(6.0)(x)

    x = tf.keras.layers.Conv2D(16, 3, padding="same", use_bias=False)(x)
    x = Normalization(batch_norm=True, scale=False)(x)
    x = tf.keras.layers.ReLU(6.0)(x)

    x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), padding="same")(x)

    x = tf.keras.layers.Conv2D(32, 3, padding="same", use_bias=False)(x)
    x = Normalization(batch_norm=True, scale=False)(x)
    x = tf.keras.layers.ReLU(6.0)(x)

    x = tf.keras.layers.Conv2D(32, 3, padding="same", use_bias=False)(x)
    x = Normalization(batch_norm=True, scale=False)(x)
    x = tf.keras.layers.ReLU(6.0)(x)

    x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), padding="same")(x)

    x = tf.keras.layers.Conv2D(32, 3, padding="same", use_bias=False)(x)
    x = Normalization(batch_norm=True, scale=False)(x)
    x = tf.keras.layers.ReLU(6.0)(x)

    x = tf.keras.layers.Conv2D(32, 3, padding="same", use_bias=False)(x)
    x = Normalization(batch_norm=True, scale=False)(x)
    x = tf.keras.layers.ReLU(6.0)(x)

    x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), padding="same")(x)

    x = tf.keras.layers.Flatten()(x)

    x = tf.keras.layers.Dense(64)(x)
    x = tf.keras.layers.ReLU(6.0)(x)

    x = tf.keras.layers.Dense(32)(x)
    x = tf.keras.layers.ReLU(6.0)(x)

    x = tf.keras.layers.Dense(32)(x)
    x = tf.keras.layers.ReLU(6.0)(x)

    return _get_common_classifier_output(x, n_classes, with_offset, inputs)


def _get_classifier_single_category_v2(
    patch_size: list[int],
    channels_in: int,
    n_meta: int,
    n_context: int,
    n_classes: int,
    with_offset: bool,
    use_batch_norm: bool,
):
    image = tf.keras.layers.Input((*patch_size, channels_in))
    inputs = [image]

    if n_meta > 0:
        meta = tf.keras.layers.Input((n_meta,))
        inputs += [meta]

    if n_context > 0:
        context = tf.keras.layers.Input((n_context,))
        inputs += [n_context]

    x = image
    if n_meta > 0:
        x = tf.keras.layers.Concatenate()([image, meta])

    x = tf.keras.layers.Conv2D(16, 3, padding="same", use_bias=False)(x)
    x = Normalization(batch_norm=True, scale=False)(x)
    x = tf.keras.layers.ReLU(6.0)(x)

    x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), padding="same")(x)

    x = tf.keras.layers.Conv2D(16, 3, padding="same", use_bias=False)(x)
    x = Normalization(batch_norm=True, scale=False)(x)
    x = tf.keras.layers.ReLU(6.0)(x)

    x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), padding="same")(x)

    x = tf.keras.layers.Conv2D(32, 3, padding="same", use_bias=False)(x)
    x = Normalization(batch_norm=True, scale=False)(x)
    x = tf.keras.layers.ReLU(6.0)(x)

    x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), padding="same")(x)

    x = tf.keras.layers.Conv2D(32, 3, padding="same", use_bias=False)(x)
    x = Normalization(batch_norm=True, scale=False)(x)
    x = tf.keras.layers.ReLU(6.0)(x)

    x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), padding="same")(x)

    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(32)(x)
    x = tf.keras.layers.ReLU(6.0)(x)

    x = tf.keras.layers.Dense(32)(x)
    x = tf.keras.layers.ReLU(6.0)(x)

    return _get_common_classifier_output(x, n_classes, with_offset, inputs)


def _get_classifier_ires_single_category(
    patch_size: list[int],
    channels_in: int,
    n_meta: int,
    n_context: int,
    n_classes: int,
    with_offset: bool,
    use_batch_norm: bool,
):
    image = tf.keras.layers.Input((*patch_size, channels_in))
    inputs = [image]

    if n_meta > 0:
        meta = tf.keras.layers.Input((n_meta,))
        inputs += [meta]

    if n_context > 0:
        context = tf.keras.layers.Input((n_context,))
        inputs += [n_context]

    x = image
    if n_meta > 0:
        x = tf.keras.layers.Concatenate()([image, meta])

    x = IresBlock(16, use_batch_norm, stride=1, expansion=6)(x)
    x = IresBlock(16, use_batch_norm, stride=2, expansion=6)(x)
    x = IresBlock(32, use_batch_norm, stride=1, expansion=6)(x)
    x = IresBlock(32, use_batch_norm, stride=2, expansion=6)(x)
    x = IresBlock(64, use_batch_norm, stride=1, expansion=6)(x)

    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(64)(x)
    x = tf.keras.layers.ReLU(6.0)(x)

    x = tf.keras.layers.Dense(32)(x)
    x = tf.keras.layers.ReLU(6.0)(x)

    return _get_common_classifier_output(x, n_classes, with_offset, inputs)


def _get_classifier_ires_single_category_v2(
    patch_size: list[int],
    channels_in: int,
    n_meta: int,
    n_context: int,
    n_classes: int,
    with_offset: bool,
    use_batch_norm: bool,
):
    image = tf.keras.layers.Input((*patch_size, channels_in))
    inputs = [image]

    if n_meta > 0:
        meta = tf.keras.layers.Input((n_meta,))
        inputs += [meta]

    if n_context > 0:
        context = tf.keras.layers.Input((n_context,))
        inputs += [n_context]

    x = image
    if n_meta > 0:
        x = tf.keras.layers.Concatenate()([image, meta])

    x = IresBlock(8, use_batch_norm, stride=2, expansion=6)(x)
    x = IresBlock(16, use_batch_norm, stride=2, expansion=6)(x)
    x = IresBlock(32, use_batch_norm, stride=2, expansion=6)(x)

    x = tf.keras.layers.Flatten()(x)

    x = tf.keras.layers.Dense(32)(x)
    x = tf.keras.layers.ReLU(6.0)(x)

    return _get_common_classifier_output(x, n_classes, with_offset, inputs)


def _get_classifier_ires_single_category_v3(
    patch_size: list[int],
    channels_in: int,
    n_meta: int,
    n_context: int,
    n_classes: int,
    with_offset: bool,
    use_batch_norm: bool,
):
    image = tf.keras.layers.Input((*patch_size, channels_in))
    inputs = [image]

    if n_meta > 0:
        meta = tf.keras.layers.Input((n_meta,))
        inputs += [meta]

    if n_context > 0:
        context = tf.keras.layers.Input((n_context,))
        inputs += [n_context]

    x = image
    if n_meta > 0:
        x = tf.keras.layers.Concatenate()([image, meta])

    x = IresBlock(8, use_batch_norm, stride=2, expansion=6)(x)
    x = IresBlock(16, use_batch_norm, stride=2, expansion=4)(x)
    x = IresBlock(24, use_batch_norm, stride=2, expansion=4)(x)

    x = tf.keras.layers.Flatten()(x)

    x = tf.keras.layers.Dense(24)(x)
    x = tf.keras.layers.ReLU(6.0)(x)

    return _get_common_classifier_output(x, n_classes, with_offset, inputs)


def _get_classifier_ires_single_category_v4(
    patch_size: list[int],
    channels_in: int,
    n_meta: int,
    n_context: int,
    n_classes: int,
    with_offset: bool,
    use_batch_norm: bool,
):
    image = tf.keras.layers.Input((*patch_size, channels_in))
    inputs = [image]

    if n_meta > 0:
        meta = tf.keras.layers.Input((n_meta,))
        inputs += [meta]

    if n_context > 0:
        context = tf.keras.layers.Input((n_context,))
        inputs += [n_context]

    x = image
    if n_meta > 0:
        x = tf.keras.layers.Concatenate()([image, meta])

    x = IresBlock(8, use_batch_norm, stride=2, expansion=6)(x)
    x = IresBlock(16, use_batch_norm, stride=2, expansion=6)(x)
    x = IresBlock(16, use_batch_norm, stride=2, expansion=6)(x)

    x = tf.keras.layers.Flatten()(x)

    x = tf.keras.layers.Dense(24)(x)
    x = tf.keras.layers.ReLU(6.0)(x)

    return _get_common_classifier_output(x, n_classes, with_offset, inputs)

def _get_classifier_ires_single_category_v5(
    patch_size: list[int],
    channels_in: int,
    n_meta: int,
    n_context: int,
    n_classes: int,
    with_offset: bool,
    use_batch_norm: bool,
):
    image = tf.keras.layers.Input((*patch_size, channels_in))
    inputs = [image]

    if n_meta > 0:
        meta = tf.keras.layers.Input((n_meta,))
        inputs += [meta]

    if n_context > 0:
        context = tf.keras.layers.Input((n_context,))
        inputs += [n_context]

    x = image
    if n_meta > 0:
        x = tf.keras.layers.Concatenate()([image, meta])

    x = IresBlock(8, use_batch_norm, stride=2, expansion=6)(x)
    x = IresBlock(8, use_batch_norm, stride=1, expansion=6)(x)
    x = IresBlock(8, use_batch_norm, stride=2, expansion=6)(x)

    x = tf.keras.layers.Flatten()(x)

    x = tf.keras.layers.Dense(8)(x)
    x = tf.keras.layers.ReLU(6.0)(x)

    return _get_common_classifier_output(x, n_classes, with_offset, inputs)
