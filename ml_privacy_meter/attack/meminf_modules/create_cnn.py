import tensorflow as tf

keraslayers = tf.compat.v1.keras.layers


def cnn_for_fcn_gradients(input_shape):
    """
    Creates a CNN submodule for Dense layer gradients.
    """
    # Input container
    dim1 = int(input_shape[0])
    dim2 = int(input_shape[1])
    initializer = tf.compat.v1.keras.initializers.random_normal(0.0, 0.01)

    # CNN module
    cnn = tf.compat.v1.keras.Sequential(
        [
            keraslayers.Dropout(0.2, input_shape=(dim1, dim2, 1,),),
            keraslayers.Conv2D(
                100,
                kernel_size=(1, dim2),
                strides=(1, 1),
                padding='valid',
                activation=tf.nn.relu,
                data_format="channels_last",
                #input_shape=(dim1, dim2, 1,),
                kernel_initializer=initializer,
                bias_initializer='zeros',
            ),
            keraslayers.Flatten(),
            keraslayers.Dropout(0.2),
            keraslayers.Dense(
                2024,
                activation=tf.nn.relu,
                kernel_initializer=initializer,
                bias_initializer='zeros'
            ),
            keraslayers.Dropout(0.2, input_shape=(dim1, dim2, 1,),),
            keraslayers.Dense(
                512,
                activation=tf.nn.relu,
                kernel_initializer=initializer,
                bias_initializer='zeros'
            ),
            keraslayers.Dense(
                256,
                activation=tf.nn.relu,
                kernel_initializer=initializer,
                bias_initializer='zeros'
            )
        ]
    )
    return cnn


def cnn_for_cnn_layeroutputs(input_shape):
    """
    Creates a CNN submodule for Conv Layer outputs
    """
    # Input container
    dim2 = int(input_shape[1])
    dim3 = int(input_shape[2])
    dim4 = int(input_shape[3])
    initializer = tf.compat.v1.keras.initializers.random_normal(0.0, 0.01)

    # CNN module
    cnn = tf.compat.v1.keras.Sequential(
        [
            keraslayers.Conv2D(
                dim4,
                kernel_size=(dim2, dim3),
                strides=(1, 1),
                padding='valid',
                activation=tf.nn.relu,
                data_format="channels_last",
                input_shape=(dim2, dim3, dim4,),
                kernel_initializer=initializer,
                bias_initializer='zeros',
            ),
            keraslayers.Flatten(),
            keraslayers.Dropout(0.2),
            keraslayers.Dense(
                1024,
                activation=tf.nn.relu,
                kernel_initializer=initializer,
                bias_initializer='zeros'
            ),
            keraslayers.Dropout(0.2),
            keraslayers.Dense(
                512,
                activation=tf.nn.relu,
                kernel_initializer=initializer,
                bias_initializer='zeros'
            ),
            keraslayers.Dense(
                128,
                activation=tf.nn.relu,
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
    return cnn


def cnn_for_cnn_gradients(input_shape):
    """
    Creates a CNN submodule for Conv layer gradients
    """
    dim1 = int(input_shape[3])
    dim2 = int(input_shape[0])
    dim3 = int(input_shape[1])
    dim4 = int(input_shape[2])
    initializer = tf.compat.v1.keras.initializers.random_normal(0.0, 0.01)
    # CNN module
    cnn = tf.compat.v1.keras.Sequential(
        [
            keraslayers.Conv2D(
                dim1,
                kernel_size=(dim2, dim3),
                strides=(1, 1),
                padding='same',
                activation=tf.nn.relu,
                input_shape=(dim1, dim2, dim4),
                kernel_initializer=initializer,
                bias_initializer='zeros',
                name='cnn_grad_layer'
            ),
            keraslayers.Flatten(name='flatten_layer'),
            keraslayers.Dropout(0.2),
            keraslayers.Dense(
                64,
                activation=tf.nn.relu,
                kernel_initializer=initializer,
                bias_initializer='zeros'
            )

        ]
    )
    return cnn
