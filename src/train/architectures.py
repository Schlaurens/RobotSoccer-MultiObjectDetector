import tensorflow as tf


def get_encoder(
    encoder_architecture: str,
    height: int,
    width: int,
    category_names: list[str],
    n_context: int,
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
    if encoder_architecture == "default_light":
        return _get_encoder_default_light(height, width, category_names, n_context, **kwargs)
    if encoder_architecture == "default_heavy":
        return _get_encoder_default_heavy(height, width, category_names, n_context, **kwargs)
    if encoder_architecture == "inverted_residual_light":
        return _get_encoder_inverted_residual_light(
            height, width, category_names, n_context, **kwargs
        )
    if encoder_architecture == "inverted_residual_single_category":
        return _get_encoder_inverted_residual_single_category(
            height, width, category_names, n_context, **kwargs
        )
    if encoder_architecture == "inverted_residual_single_category_v2":
        return _get_encoder_inverted_residual_single_category_v2(
            height, width, category_names, n_context, **kwargs
        )
    else:
        raise ValueError(f"Unknown encoder name: {encoder_architecture}")


def _get_common_output(x, category_names, n_context, image):
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


def _ires_block(x, filters, stride=1, expansion=6):
    """Inverted residual block as specified in MobileNetV2

    Args:
        x: output from previous layer
        filters: Number of filters
        stride: The stride. Defaults to 1.
        expansion: Expand the number of filters by multiplying them with this number. Defaults to 6.

    Returns:
        The sum of residual and x.
    """
    residual = x

    # Expansion phase: 1x1 convolution to expand channels
    x = tf.keras.layers.Conv2D(filters * expansion, 1, padding="same", use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization(scale=False)(x)
    x = tf.keras.layers.ReLU(6.0)(x)

    # Use Depthwise convolution
    x = tf.keras.layers.DepthwiseConv2D(3, strides=stride, padding="same", use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization(scale=False)(x)
    x = tf.keras.layers.ReLU(6.0)(x)

    # Projection phase: 1x1 convolution to project back to original channels
    x = tf.keras.layers.Conv2D(filters, 1, padding="same", use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization(scale=False)(x)

    # If dimensions changed, project the residual
    if stride != 1 or residual.shape[-1] != filters:
        residual = tf.keras.layers.Conv2D(
            filters, 1, strides=stride, padding="same", use_bias=False
        )(residual)
        residual = tf.keras.layers.BatchNormalization(scale=False)(residual)

    # Add residual
    x = tf.keras.layers.Add()([x, residual])
    return x


def _get_encoder_default_heavy(height, width, category_names, n_context):
    image = tf.keras.layers.Input((height, width, 4))
    x = image

    # 480x320x4
    x = tf.keras.layers.Conv2D(16, 3, strides=(2, 1), padding="same", use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization(scale=False)(x)
    x = tf.keras.layers.ReLU(6.0)(x)

    # 240x320x16
    x = tf.keras.layers.Conv2D(32, 3, strides=(2, 2), padding="same", use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization(scale=False)(x)
    x = tf.keras.layers.ReLU(6.0)(x)

    # 120x160x32
    x = tf.keras.layers.Conv2D(32, 3, strides=(2, 2), padding="same", use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization(scale=False)(x)
    x = tf.keras.layers.ReLU(6.0)(x)

    # 60x80x32
    x = tf.keras.layers.Conv2D(64, 3, strides=(2, 2), padding="same", use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization(scale=False)(x)
    x = tf.keras.layers.ReLU(6.0)(x)

    # 30x40x64
    x = tf.keras.layers.Conv2D(64, 3, strides=(2, 2), padding="same", use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization(scale=False)(x)
    x = tf.keras.layers.ReLU(6.0)(x)

    # 15x20x64
    return _get_common_output(x, category_names, n_context, image)


def _get_encoder_default_light(height, width, category_names, n_context):
    image = tf.keras.layers.Input((height, width, 4))
    # TODO: input [B, H, W/2, 4] (treat each YUYV tuple as a pixel)
    x = image

    # 480x320x4
    x = tf.keras.layers.Conv2D(32, 3, strides=(2, 1), padding="same", use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization(scale=False)(x)
    x = tf.keras.layers.ReLU(6.0)(x)

    # 240x320x32
    x = tf.keras.layers.Conv2D(32, 3, strides=(2, 2), padding="same", use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization(scale=False)(x)
    x = tf.keras.layers.ReLU(6.0)(x)

    # 120x160x32
    x = tf.keras.layers.Conv2D(32, 3, strides=(2, 2), padding="same", use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization(scale=False)(x)
    x = tf.keras.layers.ReLU(6.0)(x)

    # 60x80x32
    x = tf.keras.layers.Conv2D(32, 3, strides=(2, 2), padding="same", use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization(scale=False)(x)
    x = tf.keras.layers.ReLU(6.0)(x)

    # 30x40x32
    x = tf.keras.layers.Conv2D(32, 3, strides=(2, 2), padding="same", use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization(scale=False)(x)
    x = tf.keras.layers.ReLU(6.0)(x)

    # 15x20x160
    return _get_common_output(x, category_names, n_context, image)


def _get_encoder_inverted_residual_light(height, width, category_names, n_context):
    image = tf.keras.layers.Input((height, width, 4))
    # Be careful not to make the tensors too much for the GPU memory. (keep the expansion low for the bigger tensors)
    x = image

    # 480x320x4
    # cannot be ires block due to uneven stride
    x = tf.keras.layers.Conv2D(24, 3, strides=(2, 1), padding="same", use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization(scale=False)(x)
    x = tf.keras.layers.ReLU(6.0)(x)

    # 240x320x24
    x = _ires_block(x, 24, stride=1, expansion=1)

    # 240x320x24
    x = _ires_block(x, 24, stride=2, expansion=1)

    # 120x160x24
    x = _ires_block(x, 24, stride=1, expansion=1)

    # 120x160x24
    x = _ires_block(x, 32, stride=2, expansion=1)

    # 60x80x32
    x = _ires_block(x, 32, stride=1, expansion=6)

    # 60x80x32
    x = _ires_block(x, 32, stride=2, expansion=6)

    # 30x40x64
    x = _ires_block(x, 32, stride=1, expansion=6)

    # 30x40x64
    x = _ires_block(x, 32, stride=2, expansion=6)

    # 15x20x64
    x = _ires_block(x, 32, stride=1, expansion=6)

    # 15x20x64
    return _get_common_output(x, category_names, n_context, image)

def _get_encoder_inverted_residual_single_category(height, width, category_names, n_context):
    image = tf.keras.layers.Input((height, width, 4))
    # Be careful not to make the tensors too much for the GPU memory. (keep the expansion low for the bigger tensors)
    x = image

    # 480x320x4
    # cannot be ires block due to uneven stride
    x = tf.keras.layers.Conv2D(16, 3, strides=(2, 1), padding="same", use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization(scale=False)(x)
    x = tf.keras.layers.ReLU(6.0)(x)

    # 240x320x24
    x = _ires_block(x, 16, stride=1, expansion=1)

    # 240x320x24
    x = _ires_block(x, 16, stride=2, expansion=1)

    # 120x160x24
    # x = _ires_block(x, 16, stride=1, expansion=1)

    # 120x160x24
    x = _ires_block(x, 24, stride=2, expansion=1)

    # 60x80x32
    x = _ires_block(x, 24, stride=1, expansion=1)

    # 60x80x32
    x = _ires_block(x, 32, stride=2, expansion=1)

    # 30x40x64
    x = _ires_block(x, 32, stride=1, expansion=1)

    # 30x40x64
    x = _ires_block(x, 32, stride=2, expansion=1)

    # 15x20x64
    x = _ires_block(x, 32, stride=1, expansion=1)

    # 15x20x64
    return _get_common_output(x, category_names, n_context, image)

def _get_encoder_inverted_residual_single_category_v2(height, width, category_names, n_context):
    image = tf.keras.layers.Input((height, width, 4))
    # Be careful not to make the tensors too much for the GPU memory. (keep the expansion low for the bigger tensors)
    x = image

    # 480x320x4
    # cannot be ires block due to uneven stride
    x = tf.keras.layers.Conv2D(16, 3, strides=(2, 1), padding="same", use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization(scale=False)(x)
    x = tf.keras.layers.ReLU(6.0)(x)

    # 240x320x24
    x = _ires_block(x, 16, stride=1, expansion=1)

    # 240x320x24
    x = _ires_block(x, 16, stride=2, expansion=1)

    # 120x160x24
    # x = _ires_block(x, 16, stride=1, expansion=1)

    # 120x160x24
    x = _ires_block(x, 24, stride=2, expansion=4)

    # 60x80x32
    x = _ires_block(x, 24, stride=1, expansion=4)

    # 60x80x32
    x = _ires_block(x, 32, stride=2, expansion=4)

    # 30x40x64
    x = _ires_block(x, 32, stride=1, expansion=4)

    # 30x40x64
    x = _ires_block(x, 32, stride=2, expansion=4)

    # 15x20x64
    x = _ires_block(x, 32, stride=1, expansion=4)

    # 15x20x64
    return _get_common_output(x, category_names, n_context, image)
