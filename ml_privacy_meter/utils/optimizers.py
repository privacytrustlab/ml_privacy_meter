import tensorflow as tf


def optimizer_op(optimizer, learning_rate, momentum=0.9, decay=0.0005):
    """
    Returns TensorFlow optimizer for supported optimizers
    """
    op_lower_case = optimizer.lower()
    if op_lower_case == "adadelta":
        return tf.optimizers.Adadelta(learning_rate)
    elif op_lower_case == "adagrad":
        return tf.optimizers.Adagrad(learning_rate)
    elif op_lower_case == "adam":
        return tf.optimizers.Adam(learning_rate)
    elif op_lower_case == "sgd":
        return tf.optimizers.SGD(learning_rate)
    elif op_lower_case == "momentum":
        return tf.optimizers.SGD(learning_rate, momentum, decay)
    elif op_lower_case == "rmsprop":
        return tf.optimizers.RMSprop(learning_rate)
