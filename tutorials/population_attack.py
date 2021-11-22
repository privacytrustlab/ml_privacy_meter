import os
import numpy as np
import tensorflow as tf
from tensorflow.python.keras import Sequential
from tensorflow.python.keras.layers import Dense, Conv2D, MaxPool2D, Flatten, Dropout

import ml_privacy_meter

# Set input format for image data
from tensorflow.keras import backend as K
K.set_image_data_format('channels_last')


def preprocess_cifar10_dataset():
    input_shape, num_classes = (32, 32, 3), 10

    # Split the data between train and test sets
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

    # Scale images to the [0, 1] range
    x_train = x_train.astype("float32") / 255
    x_test = x_test.astype("float32") / 255

    # Convert labels into one hot vectors
    y_train = tf.keras.utils.to_categorical(y_train, num_classes)
    y_test = tf.keras.utils.to_categorical(y_test, num_classes)

    return x_train, y_train, x_test, y_test, input_shape, num_classes


def get_cnn_classifier(input_shape, num_classes, regularizer):
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu',
                     input_shape=input_shape, kernel_regularizer=regularizer))
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(Conv2D(64, kernel_size=(3, 3), activation='relu',
                     kernel_regularizer=regularizer))
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))
    return model


if __name__ == '__main__':
    # Part 1: Train and attack a tensorflow model
    loss_fn = 'categorical_crossentropy'
    optim_fn = 'adam'
    epochs = 50
    batch_size = 64
    regularizer_penalty = 0.01
    regularizer = tf.keras.regularizers.l2(l=regularizer_penalty)

    # get dataset
    x_train, y_train, x_test, y_test, input_shape, num_classes = preprocess_cifar10_dataset()
    num_datapoints = 5000

    model_train_dirpath = 'models'
    tensorflow_model_filepath = f"{model_train_dirpath}/tutorial_tensorflow_cifar10/final_model"
    if os.path.isdir(tensorflow_model_filepath):
        print(f"Model already trained. Continuing...")
    else:
        print("Training model...")
        x = np.array(x_train[:num_datapoints])
        y = np.array(y_train[:num_datapoints])

        # initialize and compile model
        model = get_cnn_classifier(input_shape, num_classes, regularizer)
        model.summary()
        model.compile(optimizer=optim_fn, loss=loss_fn, metrics=['accuracy'])

        # train and save model
        model.fit(x, y, batch_size=batch_size, epochs=epochs, verbose=2)
        model.save(f"{model_train_dirpath}/tutorial_tensorflow_cifar10/final_model")

    # create population attack object
    exp_name = 'tutorial_tensorflow_cifar10'
    loss_fn = tf.keras.losses.CategoricalCrossentropy(from_logits=True, reduction=tf.keras.losses.Reduction.NONE)
    num_data_in_class = 200
    population_attack_obj = ml_privacy_meter.attack.population_meminf.PopulationAttack(
        exp_name=exp_name,
        x_population=x_train[num_datapoints:], y_population=y_train[num_datapoints:],
        x_target_train=x_train[:num_datapoints], y_target_train=y_train[:num_datapoints],
        x_target_test=x_test[:num_datapoints], y_target_test=y_train[:num_datapoints],
        target_model_filepath=tensorflow_model_filepath,
        target_model_type='tensorflow',
        loss_fn=loss_fn,
        num_data_in_class=num_data_in_class,
        seed=1234
    )

    population_attack_obj.prepare_attack()

    alphas = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
    population_attack_obj.run_attack(alphas=alphas)



