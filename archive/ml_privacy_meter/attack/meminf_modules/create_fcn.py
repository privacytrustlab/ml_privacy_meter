import tensorflow as tf

keraslayers = tf.compat.v1.keras.layers


def fcn_module(inputsize, layer_size=128):
    """
    Creates a FCN submodule. Used in different attack components.
    Args:
    ------
    inputsize: size of the input layer
    """
    initializer = tf.compat.v1.keras.initializers.random_normal(0.0, 0.01)

    # FCN module
    fcn = tf.compat.v1.keras.Sequential(
        [
            keraslayers.Dense(
                layer_size,
                activation=tf.nn.relu,
                input_shape=(inputsize,),
                kernel_initializer=initializer,
                bias_initializer='zeros'
            ),
            keraslayers.Dense(
                64,
                activation=tf.nn.relu,
                kernel_initializer=initializer,
                bias_initializer='zeros'
            )
        ]
    )
    return fcn
