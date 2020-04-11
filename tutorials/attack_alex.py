import numpy as np
import tensorflow as tf
import tensorflow.compat.v1.keras.layers as keraslayers
from tensorflow.compat.v1.train import Saver

import ml_privacy_meter

# input_features should be changed according to the model
input_shape = (32, 32, 3)

# Load saved target model to attack
cprefix = 'logs/pretrained'
cmodelA = tf.keras.models.load_model(cprefix)


# `saved_path` is required for obtaining the training data that was used to
# train the target classification model. This is because
# the datapoints that form the memberset of the training data of the attack
# model has to be a subset of the training data of target classification model.
# User can store the training data wherever he/she wants but the only requirement
# is that the file has to be stored in '.npy' format. The contents should be of
# the same format as the .txt file of the dataset.
saved_path = "datasets/cifar100.txt.npy"

# Similar to `saved_path` being used to form the memberset for attack model,
# `dataset_path` is used for forming the nonmemberset of the training data of
# attack model.
dataset_path = 'datasets/cifar100.txt'

datahandlerA = ml_privacy_meter.utils.attack_data.attack_data(dataset_path=dataset_path,
                                                              member_dataset_path=saved_path,
                                                              batch_size=100,
                                                              attack_percentage=10, input_shape=input_shape,
                                                              normalization=True)


# Set means and standard deviations for normalization.
# If unset, they are calculated from the dataset.
datahandlerA.means, datahandlerA.stddevs = [
    0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010]

attackobj = ml_privacy_meter.attack.whitebox.initialize(
    target_train_model=cmodelA,
    target_attack_model=cmodelA,
    train_datahandler=datahandlerA,
    attack_datahandler=datahandlerA,
    layers_to_exploit=[26],
    gradients_to_exploit=[6],
    device=None)

attackobj.train_attack()
