import ml_privacy_meter
import tensorflow as tf
import tensorflow.compat.v1.keras.layers as keraslayers
from tensorflow.compat.v1.train import Saver


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

# `saved_path` is required for obtaining the training data that was used to
# train the target classification model. This is because
# the datapoints that form the memberset of the training data of the attack
# model has to be a subset of the training data of target classification model.
# User can store the training data wherever he/she wants but the only requirement
# is that the file has to be stored in '.npy' format.
saved_path = "datasets/purchase100.txt.npy"

# Similar to `saved_path` being used to form the memberset for attack model,
# `dataset_path` is used for forming the nonmemberset of the training data of
# attack model.
dataset_path = 'datasets/purchase100.txt'

datahandlerA = ml_privacy_meter.utils.attack_data.attack_data(dataset_path=dataset_path,
                                                              member_dataset_path=saved_path,
                                                              batch_size=64,
                                                              attack_percentage=50, input_shape=(input_features, ))

attackobj = ml_privacy_meter.attack.meminf.meminf(
    target_train_model=cmodelA,
    target_attack_model=cmodelA,
    learning_rate=0.0001, optimizer='adam',
    train_datahandler=datahandlerA,
    attack_datahandler=datahandlerA,
    layers_to_exploit=[5],
    gradients_to_exploit=[3, 4, 5])
attackobj.train_attack()
