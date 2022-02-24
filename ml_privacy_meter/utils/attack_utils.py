import os
import time

import numpy as np

import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.compat.v1.train import Saver

from openvino.inference_engine import IECore
import torch

MODEL_TYPE_OPENVINO = 'openvino'
MODEL_TYPE_TENSORFLOW = 'tensorflow'
MODEL_TYPE_PYTORCH = 'pytorch'

def sanity_check(layers, layers_to_exploit):
    """
    Basic sanity check for layers and gradients to exploit based on model layers
    """
    if layers_to_exploit and len(layers_to_exploit):
        assert np.max(layers_to_exploit) <= len(layers),\
            "layer index greater than the last layer"


def time_taken(self, start_time, end_time):
    """
    Calculates difference between 2 times
    """
    delta = end_time - start_time
    hours = int(delta / 3600)
    delta -= hours * 3600
    minutes = int(delta / 60)
    delta -= minutes * 60
    seconds = delta
    return hours, minutes, np.int(seconds)


def get_predictions(model_filepath, model_type, data, model_class=None):
    predictions = []
    if model_type == MODEL_TYPE_OPENVINO:
        ie = IECore()
        net = ie.read_network(model=model_filepath)
        exec_net = ie.load_network(network=net, device_name='CPU')
        input_layer = next(iter(net.input_info))
        output_layer = next(iter(net.outputs))

        input_shape_net = net.input_info[input_layer].tensor_desc.dims

        # reshape network so that its batch_size = len(data)
        new_input_shape_net = input_shape_net
        new_input_shape_net[0] = len(data)
        net.reshape({input_layer: new_input_shape_net})
        exec_net = ie.load_network(network=net, device_name='CPU')

        predictions = exec_net.infer({input_layer: data})[output_layer]
    elif model_type == MODEL_TYPE_TENSORFLOW:
        model = tf.keras.models.load_model(model_filepath)
        predictions = model(data)
    elif model_type == MODEL_TYPE_PYTORCH:
        model = model_class()  # pytorch models need to be instantiated
        model.load_state_dict(torch.load(model_filepath))
        model.eval()

        # pytorch models need channels-first data
        data_nchw = torch.Tensor(data)
        data_nchw = data_nchw.permute(0, 3, 1, 2)

        predictions = model(data_nchw).detach().numpy()
    else:
        raise ValueError("Please specify one of the supported model types: `openvino`, `tensorflow`, or `pytorch`!")
    return predictions


def get_per_class_indices(x, y, num_data_in_class, seed):
    num_classes = y.shape[1]

    per_class_splitter = StratifiedShuffleSplit(n_splits=1,
                                                train_size=(num_data_in_class * num_classes),
                                                test_size=100,
                                                random_state=seed)

    split_indices = []
    for indices, _ in per_class_splitter.split(x, y):
        split_indices = indices

    per_class_indices = []
    for c in range(num_classes):
        indices = []
        for idx in split_indices:
            x_point, y_point = x[idx], y[idx]
            if c == np.argmax(y_point):
                indices.append(idx)
        # print(f"Number of samples from class {c} = {len(indices)}")
        per_class_indices.append(indices)

    return per_class_indices


def calculate_loss_threshold(alpha, loss_distribution):
    threshold = np.quantile(loss_distribution, q=alpha, interpolation='lower')
    return threshold


class attack_utils():
    """
    Utilities required for conducting membership inference attack
    """

    def __init__(self, directory_name='latest'):
        self.root_dir = os.path.abspath(os.path.join(
                                        os.path.dirname(__file__),
                                        "..", ".."))
        self.log_dir = os.path.join(self.root_dir, "logs")
        self.aprefix = os.path.join(self.log_dir,
                                    directory_name,
                                    "attack",
                                    "model_checkpoints")
        self.dataset_directory = os.path.join(self.root_dir, "datasets")

        if not os.path.exists(self.aprefix):
            os.makedirs(self.aprefix)
        if not os.path.exists(self.dataset_directory):
            os.makedirs(self.dataset_directory)

    def get_gradshape(self, variables, layerindex):
        """
        Returns the shape of gradient matrices
        Args:
        -----
        model: model to attack 
        """
        g = (layerindex-1)*2
        gradshape = variables[g].shape
        return gradshape

    def get_gradient_norm(self, gradients):
        """
        Returns the norm of the gradients of loss value
        with respect to the parameters

        Args:
        -----
        gradients: Array of gradients of a batch 
        """
        gradient_norms = []
        for gradient in gradients:
            summed_squares = [K.sum(K.square(g)) for g in gradient]
            norm = K.sqrt(sum(summed_squares))
            gradient_norms.append(norm)
        return gradient_norms

    def get_entropy(self, model, features, output_classes):
        """
        Calculates the prediction uncertainty
        """
        entropyarr = []
        for feature in features:
            feature = tf.reshape(feature, (1, len(feature.numpy())))
            predictions = model(feature)
            predictions = tf.nn.softmax(predictions)
            mterm = tf.reduce_sum(input_tensor=tf.multiply(predictions,
                                                           np.log(predictions)))
            entropy = (-1/np.log(output_classes)) * mterm
            entropyarr.append(entropy)
        return entropyarr

    def split(self, x):
        """
        Splits the array into number of elements equal to the
        size of the array. This is required for per example
        computation.
        """
        split_x = tf.split(x, len(x.numpy()))
        return split_x

    def get_savers(self, attackmodel):
        """
        Creates prefixes for storing classification and inference
        model
        """
        # Prefix for storing attack model checkpoints
        prefix = os.path.join(self.aprefix, "ckpt")
        # Saver for storing checkpoints
        attacksaver = Saver(attackmodel.variables)
        return prefix, attacksaver

    def createOHE(self, num_output_classes):
        """
        creates one hot encoding matrix of all the vectors
        in a given range of 0 to number of output classes.
        """
        return tf.one_hot(tf.range(0, num_output_classes),
                          num_output_classes,
                          dtype=tf.float32)

    def one_hot_encoding(self, labels, ohencoding):
        """
        Creates a one hot encoding of the labels used for 
        inference model's sub neural network

        Args: 
        ------
        zero_index: `True` implies labels start from 0
        """
        labels = tf.cast(labels, tf.int64).numpy()
        return tf.stack(list(map(lambda x: ohencoding[x], labels)))

    def intersection(self, to_remove, remove_from, batch_size):
        """
        Finds the intersection between `to_remove` and `remove_from`
        and removes this intersection from `remove_from` 
        """
        to_remove = to_remove.unbatch()
        remove_from = remove_from.unbatch()

        m1, m2 = dict(), dict()
        for example in to_remove:
            hashval = hash(bytes(np.array(example)))
            m1[hashval] = example
        for example in remove_from:
            hashval = hash(bytes(np.array(example)))
            m2[hashval] = example

        # Removing the intersection
        extracted = {key: value for key,
                     value in m2.items() if key not in m1.keys()}
        dataset = extracted.values()
        features, labels = [], []
        for d in dataset:
            features.append(d[0])
            labels.append(d[1])
        finaldataset = tf.compat.v1.data.Dataset.from_tensor_slices(
            (features, labels))
        return finaldataset.batch(batch_size=batch_size)
