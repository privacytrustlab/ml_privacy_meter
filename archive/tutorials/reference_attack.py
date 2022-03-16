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


def get_tensorflow_cnn_classifier(input_shape, num_classes, regularizer):
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
    epochs = 10
    batch_size = 64
    regularizer_penalty = 0.01
    regularizer = tf.keras.regularizers.l2(l=regularizer_penalty)

    # Get dataset
    x_train, y_train, x_test, y_test, input_shape, num_classes = preprocess_cifar10_dataset()
    num_datapoints = 5000

    # Split dataset for target model
    x_target_train, y_target_train = x_train[:num_datapoints], y_train[:num_datapoints]
    x_target_test, y_target_test = x_test[:num_datapoints], y_test[:num_datapoints]

    # Train target model
    model_train_dirpath = 'models'
    tensorflow_model_filepath = f"{model_train_dirpath}/reference_tutorial_tensorflow_cifar10/final_model"
    if os.path.isdir(tensorflow_model_filepath):
        print(f"Model already trained. Continuing...")
    else:
        print("Training model...")

        # initialize and compile model
        model = get_tensorflow_cnn_classifier(input_shape, num_classes, regularizer)
        model.summary()
        model.compile(optimizer=optim_fn, loss=loss_fn, metrics=['accuracy'])

        # train and save model
        model.fit(x_target_train, y_target_train, batch_size=batch_size, epochs=epochs, verbose=2)
        model.save(f"{model_train_dirpath}/reference_tutorial_tensorflow_cifar10/final_model")

        del(model)

    # Split dataset for reference models
    num_reference_models = 5
    x_ref_train_indices_list = []
    for idx in range(num_reference_models):
        start_idx = (idx + 1) * num_datapoints
        end_idx = (idx + 2) * num_datapoints
        ref_indices = np.array(list(range(start_idx, end_idx)))
        x_ref_train_indices_list.append(ref_indices)

    # Train reference models
    ref_model_filepath_list = [
        f"{model_train_dirpath}/reference_tutorial_tensorflow_cifar10/reference_model_{idx}/final_model"
        for idx in range(num_reference_models)
    ]
    ref_model_type_list = [ml_privacy_meter.utils.attack_utils.MODEL_TYPE_TENSORFLOW] * num_reference_models
    for idx in range(num_reference_models):
        ref_model_filepath = ref_model_filepath_list[idx]
        ref_model_indices = x_ref_train_indices_list[idx]
        if os.path.isdir(ref_model_filepath):
            print(f"Model already trained. Continuing...")
        else:
            print(f"Training reference model {idx}...")
            x_ref_train = x_train[ref_model_indices]
            y_ref_train = y_train[ref_model_indices]

            # initialize and compile model
            ref_model = get_tensorflow_cnn_classifier(input_shape, num_classes, regularizer)
            ref_model.summary()
            ref_model.compile(optimizer=optim_fn, loss=loss_fn, metrics=['accuracy'])

            # train and save model
            ref_model.fit(x_ref_train, y_ref_train, batch_size=batch_size, epochs=epochs, verbose=2)
            ref_model.save(f"{model_train_dirpath}/reference_tutorial_tensorflow_cifar10/"
                           f"reference_model_{idx}/final_model")

            del(ref_model)

    # Create reference attack object
    exp_name = 'reference_tutorial_tensorflow_cifar10'
    loss_fn = tf.keras.losses.CategoricalCrossentropy(from_logits=True, reduction=tf.keras.losses.Reduction.NONE)
    reference_attack_obj = ml_privacy_meter.attack.reference_meminf.ReferenceAttack(
        exp_name=exp_name,
        x_population=x_train[num_datapoints:], y_population=y_train[num_datapoints:],
        x_target_train=x_target_train, y_target_train=y_target_train,
        x_target_test=x_target_test, y_target_test=y_target_test,
        target_model_filepath=tensorflow_model_filepath,
        target_model_type='tensorflow',
        x_ref_train_indices_list=x_ref_train_indices_list,
        ref_model_filepath_list=ref_model_filepath_list,
        ref_model_type_list=ref_model_type_list,
        loss_fn=loss_fn, seed=1234
    )

    reference_attack_obj.prepare_attack()

    alphas = [0.1, 0.3, 0.5]
    reference_attack_obj.run_attack(alphas=alphas)

    reference_attack_obj.visualize_attack(alphas=alphas)
