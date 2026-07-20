import tensorflow as tf

from util.layers import IresBlockCompiledNN


def get_cpn(
    cpn_architecture: str,
    height: int,
    width: int,
    category_names: list[str],
    n_context: int,
    channels_in: int,
    colorspace: str = "yuyv",
    **kwargs,
):
    """Return the specified cpn model

    Args:
        cpn_architecture: The name of the cpn architecture
        height: The height of the input image
        width: The width of the input image
        category_names: The names of the different object categories the cpn needs to be able to detect
        n_context: The size of the context vector

    Raises:
        ValueError: When the provided cpn architecture is unknown

    Returns:
        A tf.keras.Model with the provided architecture
    """
    if colorspace not in ["yuyv", "grayscale"]:
        raise ValueError(f"Unknown Colorspace: {colorspace}")

    if cpn_architecture == "ires_16x16_v1":
        return _get_cpn_ires_16x16_v1(
            height, width, category_names, n_context, channels_in, colorspace, **kwargs
        )
    if cpn_architecture == "conv_16x16_v1":
        return _get_cpn_conv_16x16_v1(
            height, width, category_names, n_context, channels_in, colorspace, **kwargs
        )
    else:
        raise ValueError(f"Unknown cpn name: {cpn_architecture}")


def _get_common_cpn_output(x, category_names, n_context, image):
    """Return the common output logic for every cpn architecture. The different outputs for each categories are concatenated here.

    Args:
        x: The tensor output of the hidden cpn layers
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
        image, output, name="cpn"
    )  # input: image, output: [offset, interest] for each category + context


# ========= cpn Architectures =========

# =========================
# ======== 16 x 16 ========
# =========================


def _get_cpn_ires_16x16_v1(
    height: int,
    width: int,
    category_names: list[str],
    n_context: int,
    channels_in: int,
    colorspace: str,
):
    image = tf.keras.layers.Input((height, width, channels_in))
    # Be careful not to make the tensors too much for the GPU memory. (keep the expansion low for the bigger tensors)
    x = image

    # cannot be ires block due to uneven stride
    x = tf.keras.layers.Conv2D(
        8, 3, strides=(2, 1) if colorspace == "yuyv" else (2, 2), padding="same", use_bias=True
    )(x)
    x = tf.keras.layers.ReLU(6.0)(x)

    x = IresBlockCompiledNN(8, stride=1, expansion=1)(x)

    x = IresBlockCompiledNN(16, stride=2, expansion=1)(x)

    x = IresBlockCompiledNN(16, stride=2, expansion=4)(x)

    x = IresBlockCompiledNN(24, stride=2, expansion=3)(x)

    x = IresBlockCompiledNN(32, stride=1, expansion=3)(x)

    return _get_common_cpn_output(x, category_names, n_context, image)


def _get_cpn_conv_16x16_v1(
    height: int,
    width: int,
    category_names: list[str],
    n_context: int,
    channels_in: int,
    colorspace: str,
):
    image = tf.keras.layers.Input((height, width, channels_in))
    x = image

    x = tf.keras.layers.DepthwiseConv2D(3, strides=1, padding="same", use_bias=False)(x)
    x = tf.keras.layers.Conv2D(16, 1, padding="same", use_bias=True)(x)
    x = tf.keras.layers.ReLU(6.0)(x)

    x = tf.keras.layers.DepthwiseConv2D(3, strides=1, padding="same", use_bias=False)(x)
    x = tf.keras.layers.Conv2D(16, 1, padding="same", use_bias=True)(x)
    x = tf.keras.layers.ReLU(6.0)(x)

    x = tf.keras.layers.Conv2D(
        32, 3, strides=(2, 1) if colorspace == "yuyv" else (2, 2), padding="same", use_bias=True
    )(x)
    x = tf.keras.layers.ReLU(6.0)(x)

    x = tf.keras.layers.DepthwiseConv2D(3, strides=1, padding="same", use_bias=False)(x)
    x = tf.keras.layers.Conv2D(32, 1, padding="same", use_bias=True)(x)
    x = tf.keras.layers.ReLU(6.0)(x)

    x = tf.keras.layers.DepthwiseConv2D(3, strides=2, padding="same", use_bias=False)(x)
    x = tf.keras.layers.Conv2D(32, 1, padding="same", use_bias=True)(x)
    x = tf.keras.layers.ReLU(6.0)(x)

    x = tf.keras.layers.DepthwiseConv2D(3, strides=1, padding="same", use_bias=False)(x)
    x = tf.keras.layers.Conv2D(32, 1, padding="same", use_bias=True)(x)
    x = tf.keras.layers.ReLU(6.0)(x)

    x = tf.keras.layers.DepthwiseConv2D(3, strides=2, padding="same", use_bias=False)(x)
    x = tf.keras.layers.Conv2D(48, 1, padding="same", use_bias=True)(x)
    x = tf.keras.layers.ReLU(6.0)(x)

    x = tf.keras.layers.DepthwiseConv2D(3, strides=1, padding="same", use_bias=False)(x)
    x = tf.keras.layers.Conv2D(48, 1, padding="same", use_bias=True)(x)
    x = tf.keras.layers.ReLU(6.0)(x)

    x = tf.keras.layers.DepthwiseConv2D(3, strides=2, padding="same", use_bias=False)(x)
    x = tf.keras.layers.Conv2D(64, 1, padding="same", use_bias=True)(x)
    x = tf.keras.layers.ReLU(6.0)(x)

    x = tf.keras.layers.DepthwiseConv2D(3, strides=1, padding="same", use_bias=False)(x)
    x = tf.keras.layers.Conv2D(64, 1, padding="same", use_bias=True)(x)
    x = tf.keras.layers.ReLU(6.0)(x)

    return _get_common_cpn_output(x, category_names, n_context, image)
