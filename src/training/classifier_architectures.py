import tensorflow as tf

from util.layers import IresBlockCompiledNN


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

    if classifier_architecture == "ires_v1":
        return _get_classifier_ires_v1(
            patch_size,
            channels_in,
            n_meta,
            n_context,
            n_classes,
            with_offset,
            use_batch_norm,
        )
    if classifier_architecture == "ires_v2":
        return _get_classifier_ires_v2(
            patch_size,
            channels_in,
            n_meta,
            n_context,
            n_classes,
            with_offset,
            use_batch_norm,
        )
    if classifier_architecture == "conv_v1":
        return _get_classifier_conv_v1(
            patch_size,
            channels_in,
            n_meta,
            n_context,
            n_classes,
            with_offset,
            use_batch_norm,
        )
    if classifier_architecture == "conv_v2":
        return _get_classifier_conv_v2(
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
        y = x
        y = tf.keras.layers.Dense(1)(y)
        out = tf.keras.layers.Activation("sigmoid")(y)
    else:
        y = x
        y = tf.keras.layers.Dense(n_classes)(y)
        out = tf.keras.layers.Activation("softmax")(y)

    if with_offset:
        offset = tf.keras.layers.Dense(2)(x)
        out = [out, offset]

    return tf.keras.Model(inputs, out, name="classifier")


# ========= Classifier Architectures =========


def _get_classifier_ires_v1(
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
        inputs += [context]

    x = image
    x = IresBlockCompiledNN(8, use_batch_norm, stride=2, expansion=6)(x)
    x = IresBlockCompiledNN(16, use_batch_norm, stride=2, expansion=4)(x)
    x = IresBlockCompiledNN(16, use_batch_norm, stride=2, expansion=4)(x)

    x = tf.keras.layers.Flatten()(x)

    if n_meta > 0:
        x = tf.keras.layers.Concatenate()([x, meta])

    if n_context > 0:
        x = tf.keras.layers.Concatenate()([x, context])

    x = tf.keras.layers.Dense(18)(x)
    x = tf.keras.layers.ReLU(6.0)(x)

    return _get_common_classifier_output(x, n_classes, with_offset, inputs)


def _get_classifier_ires_v2(
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
        inputs += [context]

    x = image
    x = IresBlockCompiledNN(8, use_batch_norm, stride=2, expansion=4)(x)
    x = IresBlockCompiledNN(10, use_batch_norm, stride=2, expansion=4)(x)
    x = IresBlockCompiledNN(16, use_batch_norm, stride=2, expansion=4)(x)

    x = tf.keras.layers.Flatten()(x)

    if n_meta > 0:
        x = tf.keras.layers.Concatenate()([x, meta])

    if n_context > 0:
        x = tf.keras.layers.Concatenate()([x, context])

    x = tf.keras.layers.Dense(16)(x)
    x = tf.keras.layers.ReLU(6.0)(x)

    return _get_common_classifier_output(x, n_classes, with_offset, inputs)


def _get_classifier_conv_v1(
    patch_size, channels_in, n_meta, n_context, n_classes, with_offset, use_batch_norm
):
    image = tf.keras.layers.Input((*patch_size, channels_in))
    inputs = [image]
    if n_meta > 0:
        meta = tf.keras.layers.Input((n_meta,))
        inputs += [meta]
    if n_context > 0:
        context = tf.keras.layers.Input((n_context,))
        inputs += [context]

    x = image
    # 32x32x3
    x = tf.keras.layers.Conv2D(16, 3, strides=2, padding="same", use_bias=False)(x)
    x = tf.keras.layers.ReLU(6.0)(x)
    # 16x16x16
    x = tf.keras.layers.DepthwiseConv2D(3, strides=2, padding="same", use_bias=False)(x)
    x = tf.keras.layers.Conv2D(32, 1, padding="same", use_bias=False)(x)
    x = tf.keras.layers.ReLU(6.0)(x)
    # 8x8x32
    x = tf.keras.layers.DepthwiseConv2D(3, strides=1, padding="same", use_bias=False)(x)
    x = tf.keras.layers.Conv2D(32, 1, padding="same", use_bias=False)(x)
    x = tf.keras.layers.ReLU(6.0)(x)
    # 8x8x32
    x = tf.keras.layers.DepthwiseConv2D(3, strides=2, padding="same", use_bias=False)(x)
    x = tf.keras.layers.Conv2D(48, 1, padding="same", use_bias=False)(x)
    x = tf.keras.layers.ReLU(6.0)(x)
    # 4x4x48
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    # 48

    if n_meta > 0:
        x = tf.keras.layers.Concatenate()([x, meta])
    if n_context > 0:
        x = tf.keras.layers.Concatenate()([x, context])

    x = tf.keras.layers.Dense(32)(x)
    x = tf.keras.layers.ReLU(6.0)(x)
    return _get_common_classifier_output(x, n_classes, with_offset, inputs)


def _get_classifier_conv_v2(
    patch_size, channels_in, n_meta, n_context, n_classes, with_offset, use_batch_norm
):
    """Depthwise-separable conv variant"""
    image = tf.keras.layers.Input((*patch_size, channels_in))
    inputs = [image]
    if n_meta > 0:
        meta = tf.keras.layers.Input((n_meta,))
        inputs += [meta]
    if n_context > 0:
        context = tf.keras.layers.Input((n_context,))
        inputs += [context]

    x = image
    # 32x32x3
    x = tf.keras.layers.Conv2D(8, 3, strides=2, padding="same", use_bias=False)(x)
    x = tf.keras.layers.ReLU(6.0)(x)
    # 16x16x8
    x = tf.keras.layers.DepthwiseConv2D(3, strides=2, padding="same", use_bias=False)(x)
    x = tf.keras.layers.Conv2D(16, 1, padding="same", use_bias=False)(x)
    x = tf.keras.layers.ReLU(6.0)(x)
    # 8x8x16
    x = tf.keras.layers.DepthwiseConv2D(3, strides=2, padding="same", use_bias=False)(x)
    x = tf.keras.layers.Conv2D(24, 1, padding="same", use_bias=False)(x)
    x = tf.keras.layers.ReLU(6.0)(x)
    # 4x4x24
    x = tf.keras.layers.DepthwiseConv2D(3, strides=2, padding="same", use_bias=False)(x)
    x = tf.keras.layers.Conv2D(32, 1, padding="same", use_bias=False)(x)
    x = tf.keras.layers.ReLU(6.0)(x)
    # 2x2x32
    x = tf.keras.layers.Flatten()(x)

    if n_meta > 0:
        x = tf.keras.layers.Concatenate()([x, meta])
    if n_context > 0:
        x = tf.keras.layers.Concatenate()([x, context])

    x = tf.keras.layers.Dense(24)(x)
    x = tf.keras.layers.ReLU(6.0)(x)
    return _get_common_classifier_output(x, n_classes, with_offset, inputs)
