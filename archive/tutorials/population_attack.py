import os
from pathlib import Path

import numpy as np

import tensorflow as tf
from tensorflow.python.keras import Sequential
from tensorflow.python.keras.layers import Dense, Conv2D, MaxPool2D, Flatten, Dropout

from openvino.inference_engine import IECore

import torch
from torch.utils.data import TensorDataset, DataLoader
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F

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


class PyTorchCnnClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)  # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


if __name__ == '__main__':
    # Set attack hyperparameters
    num_data_in_class = 400
    
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
    tensorflow_model_filepath = f"{model_train_dirpath}/population_tutorial_tensorflow_cifar10/final_model"
    if os.path.isdir(tensorflow_model_filepath):
        print(f"Model already trained. Continuing...")
    else:
        print("Training model...")
        x = np.array(x_train[:num_datapoints])
        y = np.array(y_train[:num_datapoints])

        # initialize and compile model
        model = get_tensorflow_cnn_classifier(input_shape, num_classes, regularizer)
        model.summary()
        model.compile(optimizer=optim_fn, loss=loss_fn, metrics=['accuracy'])

        # train and save model
        model.fit(x, y, batch_size=batch_size, epochs=epochs, verbose=2)
        model.save(f"{model_train_dirpath}/population_tutorial_tensorflow_cifar10/final_model")

    # create population attack object
    exp_name = 'population_tutorial_tensorflow_cifar10'
    loss_fn = tf.keras.losses.CategoricalCrossentropy(from_logits=True, reduction=tf.keras.losses.Reduction.NONE)
    population_attack_obj = ml_privacy_meter.attack.population_meminf.PopulationAttack(
        exp_name=exp_name,
        x_population=x_train[num_datapoints:], y_population=y_train[num_datapoints:],
        x_target_train=x_train[:num_datapoints], y_target_train=y_train[:num_datapoints],
        x_target_test=x_test[:num_datapoints], y_target_test=y_test[:num_datapoints],
        target_model_filepath=tensorflow_model_filepath,
        target_model_type=ml_privacy_meter.utils.attack_utils.MODEL_TYPE_TENSORFLOW,
        loss_fn=loss_fn,
        num_data_in_class=num_data_in_class,
        seed=1234
    )

    population_attack_obj.prepare_attack()

    alphas = [0.1, 0.3, 0.5]
    population_attack_obj.run_attack(alphas=alphas)

    population_attack_obj.visualize_attack(alphas=alphas)

    # Part 2: Train and attack an openvino model
    # convert existing tensorflow model to an openvino model
    openvino_model_filepath = Path(f"{model_train_dirpath}/tutorial_tensorflow_cifar10/saved_model.xml")
    mo_command = f"""mo
                  --saved_model_dir "{tensorflow_model_filepath}"
                  --disable_nhwc_to_nchw
                  --input_model_is_text
                  --input_shape "[64, 32, 32, 3]"
                  --data_type FP16
                  --output_dir "{openvino_model_filepath.parent}"
                  """
    mo_command = " ".join(mo_command.split())
    if not openvino_model_filepath.exists():
        print("Converting tensorflow model to openvino IR model (this may take a few minutes)...")
        os.system(mo_command)
    else:
        print(f"openvino IR model already exists at {openvino_model_filepath}")

    exp_name = 'tutorial_openvino_cifar10'
    loss_fn = tf.keras.losses.CategoricalCrossentropy(from_logits=True, reduction=tf.keras.losses.Reduction.NONE)
    population_attack_obj = ml_privacy_meter.attack.population_meminf.PopulationAttack(
        exp_name=exp_name,
        x_population=x_train[num_datapoints:], y_population=y_train[num_datapoints:],
        x_target_train=x_train[:num_datapoints], y_target_train=y_train[:num_datapoints],
        x_target_test=x_test[:num_datapoints], y_target_test=y_test[:num_datapoints],
        target_model_filepath=openvino_model_filepath,
        target_model_type=ml_privacy_meter.utils.attack_utils.MODEL_TYPE_OPENVINO,
        loss_fn=loss_fn,
        num_data_in_class=num_data_in_class,
        seed=1234
    )

    population_attack_obj.prepare_attack()

    alphas = [0.1, 0.3, 0.5]
    population_attack_obj.run_attack(alphas=alphas)

    population_attack_obj.visualize_attack(alphas=alphas)

    # Part 3: Train and attack a pytorch model
    epochs = 50
    batch_size = 64

    model_train_dirpath = 'models'
    pytorch_model_filepath = f"{model_train_dirpath}/tutorial_pytorch_cifar10/final_model.pth"
    if os.path.isfile(pytorch_model_filepath):
        print(f"Model already trained. Continuing...")
    else:
        os.mkdir(Path(pytorch_model_filepath).parent)

        print("Training model...")
        x = np.array(x_train[:num_datapoints])
        y = np.array(y_train[:num_datapoints])

        # create train loader
        tensor_x, tensor_y = torch.Tensor(x), torch.Tensor(y)
        tensor_x = tensor_x.permute(0, 3, 1, 2)
        train_dataset = TensorDataset(tensor_x, tensor_y)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        # initialize and compile model
        model = PyTorchCnnClassifier()
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

        for epoch in range(epochs):
            print(f"Starting epoch {epoch}...")
            for i, data in enumerate(train_loader, 0):
                inputs, labels = data
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

        torch.save(model.state_dict(), pytorch_model_filepath)

    # create population attack object
    exp_name = 'tutorial_pytorch_cifar10'
    loss_fn = tf.keras.losses.CategoricalCrossentropy(from_logits=True, reduction=tf.keras.losses.Reduction.NONE)
    population_attack_obj = ml_privacy_meter.attack.population_meminf.PopulationAttack(
        exp_name=exp_name,
        x_population=x_train[num_datapoints:], y_population=y_train[num_datapoints:],
        x_target_train=x_train[:num_datapoints], y_target_train=y_train[:num_datapoints],
        x_target_test=x_test[:num_datapoints], y_target_test=y_test[:num_datapoints],
        target_model_filepath=pytorch_model_filepath,
        target_model_type=ml_privacy_meter.utils.attack_utils.MODEL_TYPE_PYTORCH,
        target_model_class=PyTorchCnnClassifier,  # pass in the model class for pytorch
        loss_fn=loss_fn,
        num_data_in_class=num_data_in_class,
        seed=1234
    )

    population_attack_obj.prepare_attack()

    alphas = [0.1, 0.3, 0.5]
    population_attack_obj.run_attack(alphas=alphas)

    population_attack_obj.visualize_attack(alphas=alphas)
