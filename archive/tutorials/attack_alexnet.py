import numpy as np

import ml_privacy_meter
import tensorflow as tf

# Load saved target model to attack
cprefix = 'alexnet_pretrained'
cmodelA = tf.keras.models.load_model(cprefix)

cmodelA.summary()


def preprocess_cifar100_dataset():
    input_shape = (32, 32, 3)
    num_classes = 100

    # Split the data between train and test sets
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar100.load_data()

    return x_train, y_train, x_test, y_test, input_shape, num_classes


x_train, y_train, x_test, y_test, input_shape, num_classes = preprocess_cifar100_dataset()

# training data of the target model
num_datapoints = 5000
x_target_train, y_target_train = x_train[:num_datapoints], y_train[:num_datapoints]

# population data (training data is a subset of this)
x_population = np.concatenate((x_train, x_test))
y_population = np.concatenate((y_train, y_test))

datahandlerA = ml_privacy_meter.utils.attack_data.AttackData(x_population=x_population,
                                                             y_population=y_population,
                                                             x_target_train=x_target_train,
                                                             y_target_train=y_target_train,
                                                             batch_size=100,
                                                             attack_percentage=10, input_shape=input_shape,
                                                             normalization=True)


# Set means and standard deviations for normalization.
# If unset, they are calculated from the dataset.
datahandlerA.means, datahandlerA.stddevs = [0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010]

attackobj = ml_privacy_meter.attack.meminf.initialize(
    target_train_model=cmodelA,
    target_attack_model=cmodelA,
    train_datahandler=datahandlerA,
    attack_datahandler=datahandlerA,
    layers_to_exploit=[26],
    gradients_to_exploit=[6],
    device=None, epochs=10, model_name='blackbox1')

attackobj.train_attack()
attackobj.test_attack()
