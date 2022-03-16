import ml_privacy_meter
import tensorflow as tf
import tensorflow.compat.v1.keras.layers as keraslayers
import numpy as np
from sklearn.model_selection import train_test_split


# Model to train attack model on. Should be same as the one trained.
def classification_dnn(input_features):
    # Creating the initializer
    initializer = tf.compat.v1.keras.initializers.random_normal(0.0, 0.01)
    model = tf.compat.v1.keras.Sequential(
        [
            keraslayers.Dense(
                1024,
                activation=tf.nn.tanh,
                input_shape=(input_features,),
                kernel_initializer=initializer,
                bias_initializer='zeros'
            ),
            keraslayers.Dense(
                512,
                activation=tf.nn.tanh,
                kernel_initializer=initializer,
                bias_initializer='zeros'
            ),
            keraslayers.Dense(
                256,
                activation=tf.nn.tanh,
                kernel_initializer=initializer,
                bias_initializer='zeros'
            ),
            keraslayers.Dense(
                128,
                activation=tf.nn.tanh,
                kernel_initializer=initializer,
                bias_initializer='zeros'
            ),
            keraslayers.Dense(
                100,
                kernel_initializer=initializer,
                bias_initializer='zeros'
            )
        ]
    )
    return model


# input_features should be changed according to the model
input_features = 600

#cmodelA = classification_dnn(input_features)

# Restore weights of the target classification model
# on which the attack model will be trained.
# `cprefix` path can be decided by the user.
cprefix = 'logs/fcn'
#class_ckpt_dir = tf.train.latest_checkpoint(cprefix)
# cmodelA.load_weights(class_ckpt_dir)

cmodelA = tf.keras.models.load_model(cprefix)


def preprocess_purchase100_dataset():
    input_shape = (600, )
    num_classes = 100

    # Read raw dataset
    dataset_path = "datasets/dataset_purchase"
    with open(dataset_path, "r") as f:
        purchase_dataset = f.readlines()

    # Separate features and labels into different arrays
    x, y = [], []
    for datapoint in purchase_dataset:
        split = datapoint.rstrip().split(",")
        label = int(split[0]) - 1  # The first value is the label
        features = np.array(split[1:], dtype=np.float32)  # The next values are the features

        x.append(features)
        y.append(label)

    x = np.array(x)

    # Split data into train, test sets
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=1234)

    return x_train, y_train, x_test, y_test, input_shape, num_classes


x_train, y_train, x_test, y_test, input_shape, num_classes = preprocess_purchase100_dataset()

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
                                                             batch_size=64,
                                                             attack_percentage=50,
                                                             input_shape=input_shape)

attackobj = ml_privacy_meter.attack.meminf.initialize(
    target_train_model=cmodelA,
    target_attack_model=cmodelA,
    learning_rate=0.0001, optimizer='adam',
    train_datahandler=datahandlerA,
    attack_datahandler=datahandlerA,
    layers_to_exploit=[5],
    gradients_to_exploit=[3, 4, 5])
attackobj.train_attack()
