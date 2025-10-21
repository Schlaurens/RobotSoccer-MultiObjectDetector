import tensorflow as tf


def get_encoder(
    encoder_architecture: str,
    height: int,
    width: int,
    category_names: list[str],
    n_context: int,
    use_batch_norm: bool = False,
    **kwargs,
):
    """Return the specified encoder model

    Args:
        encoder_architecture: The name of the encoder architecture
        height: The height of the input image
        width: The width of the input image
        category_names: The names of the different object categories the encoder needs to be able to detect
        n_context: The size of the context vector

    Raises:
        ValueError: When the provided encoder architecture is unknown

    Returns:
        A tf.keras.Model with the provided architecture
    """
    if encoder_architecture == "inverted_residual_light":
        return _get_encoder_inverted_residual_light(
            height, width, category_names, n_context, use_batch_norm, **kwargs
        )
    if encoder_architecture == "inverted_residual_single_category":
        return _get_encoder_inverted_residual_single_category(
            height, width, category_names, n_context, use_batch_norm, **kwargs
        )
    if encoder_architecture == "inverted_residual_single_category_v2":
        return _get_encoder_inverted_residual_single_category_v2(
            height, width, category_names, n_context, use_batch_norm, **kwargs
        )
    else:
        raise ValueError(f"Unknown encoder name: {encoder_architecture}")


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
    else:
        raise ValueError(f"Unknown encoder name: {classifier_architecture}")


def _get_common_encoder_output(x, category_names, n_context, image):
    """Return the common output logic for every encoder architecture. The different outputs for each categories are concatenated here.

    Args:
        x: The tensor output of the hidden encoder layers
        category_names: The names of the object categories. Used to build the output
        n_context: The size of the context vector
        image: The input image. Used to build the tf.keras.Model

    Returns:
        A tf.keras.Model
    """
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
        tf.keras.layers.Dense(n_classes + 1)(x)  # + 1 for the background class
        out = tf.keras.layers.Activation("softmax")(x)

    if with_offset:
        offset = tf.keras.layers.Dense(2)(x)
        out = [out, offset]

    return tf.keras.Model(inputs, out, name="classifier")


def _ires_block(x, filters, use_batch_norm, stride=1, expansion=6):
    """Inverted residual block as specified in MobileNetV2

    Args:
        x: output from previous layer
        filters: Number of filters
        batch_norm: If True use Batch Normalization. Else use Group Normalization.
        stride: The stride. Defaults to 1.
        expansion: Expand the number of filters by multiplying them with this number. Defaults to 6.

    Returns:
        The sum of residual and x.
    """
    groups = -1  # Number of groups in the GroupNorm. If -1 then the number of groups are the number of input channel (InstanceNorm)
    residual = x

    # Expansion phase: 1x1 convolution to expand channels
    x = tf.keras.layers.Conv2D(filters * expansion, 1, padding="same", use_bias=False)(x)
    if use_batch_norm:
        x = tf.keras.layers.BatchNormalization(scale=False)(x)
    else:
        x = tf.keras.layers.GroupNormalization(scale=False, groups=groups)(x)
    x = tf.keras.layers.ReLU(6.0)(x)

    # Use Depthwise convolution
    x = tf.keras.layers.DepthwiseConv2D(3, strides=stride, padding="same", use_bias=False)(x)
    if use_batch_norm:
        x = tf.keras.layers.BatchNormalization(scale=False)(x)
    else:
        x = tf.keras.layers.GroupNormalization(scale=False, groups=groups)(x)
    x = tf.keras.layers.ReLU(6.0)(x)

    # Projection phase: 1x1 convolution to project back to original channels
    x = tf.keras.layers.Conv2D(filters, 1, padding="same", use_bias=False)(x)
    if use_batch_norm:
        x = tf.keras.layers.BatchNormalization(scale=False)(x)
    else:
        x = tf.keras.layers.GroupNormalization(scale=False, groups=groups)(x)

    # If dimensions changed, project the residual
    if stride != 1 or residual.shape[-1] != filters:
        residual = tf.keras.layers.Conv2D(
            filters, 1, strides=stride, padding="same", use_bias=False
        )(residual)
        if use_batch_norm:
            residual = tf.keras.layers.BatchNormalization(scale=False)(residual)
        else:
            residual = tf.keras.layers.GroupNormalization(scale=False, groups=groups)(residual)

    # Add residual
    x = tf.keras.layers.Add()([x, residual])
    return x


# ========= Encoder Architectures =========


def _get_encoder_inverted_residual_light(
    height: int, width: int, category_names: list[str], n_context: int, use_batch_norm: bool
):
    image = tf.keras.layers.Input((height, width, 4))
    # Be careful not to make the tensors too much for the GPU memory. (keep the expansion low for the bigger tensors)
    x = image

    # 480x320x4
    # cannot be ires block due to uneven stride
    x = tf.keras.layers.Conv2D(24, 3, strides=(2, 1), padding="same", use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization(scale=False)(x)
    x = tf.keras.layers.ReLU(6.0)(x)

    # 240x320x24
    x = _ires_block(x, 24, use_batch_norm, stride=1, expansion=1)

    # 240x320x24
    x = _ires_block(x, 24, use_batch_norm, stride=2, expansion=1)

    # 120x160x24
    x = _ires_block(x, 24, use_batch_norm, stride=1, expansion=1)

    # 120x160x24
    x = _ires_block(x, 32, use_batch_norm, stride=2, expansion=1)

    # 60x80x32
    x = _ires_block(x, 32, use_batch_norm, stride=1, expansion=6)

    # 60x80x32
    x = _ires_block(x, 32, use_batch_norm, stride=2, expansion=6)

    # 30x40x64
    x = _ires_block(x, 32, use_batch_norm, stride=1, expansion=6)

    # 30x40x64
    x = _ires_block(x, 32, use_batch_norm, stride=2, expansion=6)

    # 15x20x64
    x = _ires_block(x, 32, use_batch_norm, stride=1, expansion=6)

    # 15x20x64
    return _get_common_encoder_output(x, category_names, n_context, image)


def _get_encoder_inverted_residual_single_category(
    height: int, width: int, category_names: list[str], n_context: int, use_batch_norm: bool
):
    image = tf.keras.layers.Input((height, width, 4))
    # Be careful not to make the tensors too much for the GPU memory. (keep the expansion low for the bigger tensors)
    x = image

    # 480x320x4
    # cannot be ires block due to uneven stride
    x = tf.keras.layers.Conv2D(16, 3, strides=(2, 1), padding="same", use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization(scale=False)(x)
    x = tf.keras.layers.ReLU(6.0)(x)

    # 240x320x24
    x = _ires_block(x, 16, use_batch_norm, stride=1, expansion=1)

    # 240x320x24
    x = _ires_block(x, 16, use_batch_norm, stride=2, expansion=1)

    # 120x160x24
    # x = _ires_block(x, 16, use_batch_norm, stride=1, expansion=1)

    # 120x160x24
    x = _ires_block(x, 24, use_batch_norm, stride=2, expansion=1)

    # 60x80x32
    x = _ires_block(x, 24, use_batch_norm, stride=1, expansion=1)

    # 60x80x32
    x = _ires_block(x, 32, use_batch_norm, stride=2, expansion=1)

    # 30x40x64
    x = _ires_block(x, 32, use_batch_norm, stride=1, expansion=1)

    # 30x40x64
    x = _ires_block(x, 32, use_batch_norm, stride=2, expansion=1)

    # 15x20x64
    x = _ires_block(x, 32, use_batch_norm, stride=1, expansion=1)

    # 15x20x64
    return _get_common_encoder_output(x, category_names, n_context, image)


def _get_encoder_inverted_residual_single_category_v2(
    height: int, width: int, category_names: list[str], n_context: int, use_batch_norm: bool
):
    image = tf.keras.layers.Input((height, width, 4))
    # Be careful not to make the tensors too much for the GPU memory. (keep the expansion low for the bigger tensors)
    x = image

    # 480x320x4
    # cannot be ires block due to uneven stride
    x = tf.keras.layers.Conv2D(16, 3, strides=(2, 1), padding="same", use_bias=False)(x)
    if use_batch_norm:
        x = tf.keras.layers.BatchNormalization(scale=False)(x)
    else:
        x = tf.keras.layers.GroupNormalization(scale=False, groups=-1)(x)
    x = tf.keras.layers.ReLU(6.0)(x)

    # 240x320x24
    x = _ires_block(x, 16, use_batch_norm, stride=1, expansion=1)

    # 240x320x24
    x = _ires_block(x, 16, use_batch_norm, stride=2, expansion=1)

    # 120x160x24
    # x = _ires_block(x, 16, use_batch_norm, stride=1, expansion=1)

    # 120x160x24
    x = _ires_block(x, 24, use_batch_norm, stride=2, expansion=4)

    # 60x80x32
    x = _ires_block(x, 24, use_batch_norm, stride=1, expansion=4)

    # 60x80x32
    x = _ires_block(x, 32, use_batch_norm, stride=2, expansion=4)

    # 30x40x64
    x = _ires_block(x, 32, use_batch_norm, stride=1, expansion=4)

    # 30x40x64
    x = _ires_block(x, 32, use_batch_norm, stride=2, expansion=4)

    # 15x20x64
    x = _ires_block(x, 32, use_batch_norm, stride=1, expansion=4)

    # 15x20x64
    return _get_common_encoder_output(x, category_names, n_context, image)


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
    groups = -1
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
    x = tf.keras.layers.Conv2D(32, 3, padding="same", use_bias=False)(x)
    if use_batch_norm:
        x = tf.keras.layers.BatchNormalization(scale=False)(x)
    else:
        x = tf.keras.layers.GroupNormalization(scale=False, groups=groups)(x)
    x = tf.keras.layers.ReLU(6.0)(x)

    if n_context > 0:
        x = tf.keras.layers.Concatenate()([image, context])
    x = tf.keras.layers.Conv2D(32, 3, padding="same", use_bias=False)(x)
    if use_batch_norm:
        x = tf.keras.layers.BatchNormalization(scale=False)(x)
    else:
        x = tf.keras.layers.GroupNormalization(scale=False, groups=groups)(x)
    x = tf.keras.layers.ReLU(6.0)(x)
    x = tf.keras.layers.Flatten()(x)

    return _get_common_classifier_output(x, n_classes, with_offset, inputs)
