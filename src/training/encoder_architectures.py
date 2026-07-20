import tensorflow as tf

from util.layers import IresBlockCompiledNN


def get_encoder(
    encoder_architecture: str,
    height: int,
    width: int,
    channels_in: int,
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
    if encoder_architecture == "ires_16x16_v1":
        return _get_encoder_ires_16x16_v1(
            height, width, category_names, n_context, use_batch_norm, channels_in, **kwargs
        )
    if encoder_architecture == "conv_16x16_v1":
        return _get_encoder_conv_16x16_v1(
            height, width, category_names, n_context, use_batch_norm, channels_in, **kwargs
        )
    else:
        raise ValueError(f"Unknown encoder name: {encoder_architecture}")


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
        y = x
        offset = tf.keras.layers.Conv2D(2, 1, name=f"{name}_offsets")(y)

        y = tf.keras.layers.Conv2D(1, 1)(y)
        interest = tf.keras.layers.Activation("sigmoid", name=f"{name}_interest")(y)
        output += [offset, interest]

    if n_context > 0:
        context_input = tf.keras.layers.Lambda(tf.stop_gradient)(x)
        context = tf.keras.layers.Conv2D(n_context, 1, name="context")(context_input)
        output += [context]

    return tf.keras.Model(
        image, output, name="encoder"
    )  # input: image, output: [offset, interest] for each category + context


# ========= Encoder Architectures =========

# =========================
# ======== 16 x 16 ========
# =========================


def _get_encoder_ires_16x16_v1(
    height: int,
    width: int,
    category_names: list[str],
    n_context: int,
    use_batch_norm: bool,
    channels_in: int = 4,
):
    image = tf.keras.layers.Input((height, width, channels_in))
    # Be careful not to make the tensors too much for the GPU memory. (keep the expansion low for the bigger tensors)
    x = image

    # 480x320x4
    # cannot be ires block due to uneven stride
    x = tf.keras.layers.Conv2D(
        8, 3, strides=(2, 1) if channels_in == 4 else (2, 2), padding="same", use_bias=True
    )(x)
    x = tf.keras.layers.ReLU(6.0)(x)

    # 240x320x24
    x = IresBlockCompiledNN(8, use_batch_norm, stride=1, expansion=1)(x)

    # 240x320x24
    x = IresBlockCompiledNN(16, use_batch_norm, stride=2, expansion=1)(x)

    # 120x160x24
    x = IresBlockCompiledNN(16, use_batch_norm, stride=2, expansion=4)(x)

    # 60x80x32
    x = IresBlockCompiledNN(24, use_batch_norm, stride=2, expansion=3)(x)

    # 15x20x64
    x = IresBlockCompiledNN(32, use_batch_norm, stride=1, expansion=3)(x)

    # 15x20x64
    return _get_common_encoder_output(x, category_names, n_context, image)


def _get_encoder_conv_16x16_v1(
    height: int,
    width: int,
    category_names: list[str],
    n_context: int,
    use_batch_norm: bool,
    channels_in: int = 4,
):
    image = tf.keras.layers.Input((height, width, channels_in))
    x = image
    # 240x160x4
    x = tf.keras.layers.Conv2D(
        8, 3, strides=(2, 1) if channels_in == 4 else (2, 2), padding="same", use_bias=True
    )(x)
    x = tf.keras.layers.ReLU(6.0)(x)
    # 120x160x8
    x = tf.keras.layers.DepthwiseConv2D(3, strides=2, padding="same", use_bias=False)(x)
    x = tf.keras.layers.Conv2D(16, 1, padding="same", use_bias=True)(x)
    x = tf.keras.layers.ReLU(6.0)(x)
    # 60x80x16
    x = tf.keras.layers.DepthwiseConv2D(3, strides=2, padding="same", use_bias=False)(x)
    x = tf.keras.layers.Conv2D(32, 1, padding="same", use_bias=True)(x)
    x = tf.keras.layers.ReLU(6.0)(x)
    # 30x40x32
    x = tf.keras.layers.DepthwiseConv2D(3, strides=1, padding="same", use_bias=False)(x)
    x = tf.keras.layers.Conv2D(40, 1, padding="same", use_bias=True)(x)
    x = tf.keras.layers.ReLU(6.0)(x)
    # 30x40x40
    x = tf.keras.layers.DepthwiseConv2D(3, strides=2, padding="same", use_bias=False)(x)
    x = tf.keras.layers.Conv2D(56, 1, padding="same", use_bias=True)(x)
    x = tf.keras.layers.ReLU(6.0)(x)
    # 15x20x56
    x = tf.keras.layers.DepthwiseConv2D(3, strides=1, padding="same", use_bias=False)(x)
    x = tf.keras.layers.Conv2D(64, 1, padding="same", use_bias=True)(x)
    x = tf.keras.layers.ReLU(6.0)(x)
    # 15x20x64
    return _get_common_encoder_output(x, category_names, n_context, image)
