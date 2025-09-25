import tensorflow as tf


def get_encoder(encoder_name, height, width, category_names, n_context, **kwargs):
    if encoder_name == "default":
        return _get_encoder_default(height, width, category_names, n_context, **kwargs)
    if encoder_name == "default_heavy":
        return _get_encoder_heavy(height, width, category_names, n_context, **kwargs)
    else:
        raise ValueError(f"Unknown encoder name: {encoder_name}")


def _get_common_output(x, category_names, n_context, image):
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


def _get_encoder_heavy(height, width, category_names, n_context):
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

def _get_encoder_default(height, width, category_names, n_context):
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

    return _get_common_output(x, category_names, n_context, image)
