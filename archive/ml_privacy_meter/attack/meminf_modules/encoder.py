import tensorflow as tf

keraslayers = tf.compat.v1.keras.layers


def create_encoder(encoder_inputs):
    """
    Create encoder model for membership inference attack.
    Individual attack input components are concatenated and passed to encoder.
    """
    initializer = tf.compat.v1.keras.initializers.random_normal(0.0, 0.01)
    appended = keraslayers.concatenate(encoder_inputs, axis=1)

    encoder = keraslayers.Dense(
        256,
        input_shape=(int(appended.shape[1]),),
        activation=tf.nn.relu,
        kernel_initializer=initializer,
        bias_initializer='zeros')(appended)
    encoder = keraslayers.Dense(
        128,
        activation=tf.nn.relu,
        kernel_initializer=initializer,
        bias_initializer='zeros')(encoder)
    encoder = keraslayers.Dense(
        64,
        activation=tf.nn.relu,
        kernel_initializer=initializer,
        bias_initializer='zeros')(encoder)
    encoder = keraslayers.Dense(
        1,
        activation=tf.nn.sigmoid,
        kernel_initializer=initializer,
        bias_initializer='zeros')(encoder)
    return encoder
