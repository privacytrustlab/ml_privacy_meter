from typing import Callable
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score
from abc import ABC, abstractmethod

########################################################################################################################

"""
The code here has the internal workings of our tool. 
"""


class Dataset:
    def __init__(self, x, y):
        # TODO: how to make this work with non-classification datasets?
        self.x = x
        self.y = y

    def get_dataset(self):
        return self.x, self.y

    def get_dataset_from_indices(self, indices):
        return self.x[indices], self.y[indices]


class Model(ABC):
    def __init__(self, model_obj, train_indices):
        """
        Encapsulates a model that can be used as a target model, reference model, etc.

        Params:
            model_obj: The model object.
            train_indices: The indices of the dataset model_obj was trained on.
        """
        self.model_obj = model_obj
        # TODO: should the model have a reference to the dataset it was trained on or just indices?
        self.train_indices = train_indices

    @abstractmethod
    def get_outputs(self, data):
        pass


# TODO: Should this be an abstract class instead? So users can define their own prepare_audit and run_audit functions?
class Audit:
    def __init__(self, target_model: Model,
                 target_train_dataset: Dataset,
                 target_test_dataset: Dataset,
                 population_dataset: Dataset,
                 signal_func: Callable[[Model, Dataset], np.ndarray],  # TODO: confirm signal function definition
                 threshold_func: Callable[[np.ndarray, float], float]):  # TODO: confirm threshold function definition
        self.target_model = target_model
        self.target_train_dataset = target_train_dataset
        self.target_test_dataset = target_test_dataset

        # TODO: reference attack has a list of models over which the signal needs to be computed
        # TODO: how to specify model + dataset + signal_func tuple?
        self.population_dataset = population_dataset

        self.signal_func = signal_func
        self.threshold_func = threshold_func

        # TODO: is it better to store this information on the file system or in memory?
        self.member_signals = None
        self.non_member_signals = None
        self.population_signals = None
        self.prepare_audit()

    def prepare_audit(self):
        # TODO: how to incorporate transform functions into this workflow?
        # get attack results on train and test datasets
        self.member_signals = self.signal_func(self.target_model, self.target_train_dataset)
        self.non_member_signals = self.signal_func(self.target_model, self.target_test_dataset)

        # TODO: confirm that all existing attacks follow this workflow
        # get null hypothesis from population dataset
        self.population_signals = self.signal_func(self.target_model, self.population_dataset)

    def run_audit(self, alphas):
        for alpha in alphas:
            # TODO: how to specify that thresholds are computed per-class or per-point?
            threshold = self.threshold_func(self.population_signals, alpha)

            member_preds = []
            for signal in self.member_signals:
                if signal <= threshold:
                    member_preds.append(1)
                else:
                    member_preds.append(0)

            non_member_preds = []
            for signal in self.non_member_signals:
                if signal <= threshold:
                    non_member_preds.append(1)
                else:
                    non_member_preds.append(0)

            preds = np.concatenate([member_preds, non_member_preds])

            y_eval = [1] * len(self.member_signals)
            y_eval.extend([0] * len(self.non_member_signals))

            # TODO: save attack results
            acc = accuracy_score(y_eval, preds)
            roc_auc = roc_auc_score(y_eval, preds)
            tn, fp, fn, tp = confusion_matrix(y_eval, preds).ravel()

            print(
                f"Audit performance:\n"
                f"Alpha = {alpha}\n"
                f"Accuracy = {acc}\n"
                f"ROC AUC Score = {roc_auc}\n"
                f"FPR: {fp / (fp + tn)}\n"
                f"TN, FP, FN, TP = {tn, fp, fn, tp}"
            )


########################################################################################################################

"""
The code here will be defined by the user.
"""

import tensorflow as tf
from tensorflow.python.keras import Sequential
from tensorflow.python.keras.layers import Dense, Conv2D, MaxPool2D, Flatten, Dropout


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


def get_trained_model(x_train, y_train, num_datapoints=5000):
    loss_fn = 'categorical_crossentropy'
    optim_fn = 'adam'
    epochs = 5
    batch_size = 64
    regularizer_penalty = 0.01
    regularizer = tf.keras.regularizers.l2(l=regularizer_penalty)

    train_indices = list(range(num_datapoints))
    x = np.array(x_train[train_indices])
    y = np.array(y_train[train_indices])

    # initialize and compile model
    model = get_tensorflow_cnn_classifier(input_shape, num_classes, regularizer)
    model.summary()
    model.compile(optimizer=optim_fn, loss=loss_fn, metrics=['accuracy'])

    # train and save model
    model.fit(x, y, batch_size=batch_size, epochs=epochs, verbose=2)

    return model, train_indices


########################################################################################################################

"""
The code here will be used to perform an audit using our tool.
"""
if __name__ == '__main__':
    x_train, y_train, x_test, y_test, input_shape, num_classes = preprocess_cifar10_dataset()

    # user can train a model beforehand and load the model object from a checkpoint file.
    # this will ideally work for any ML library because our tool will use the query interface
    # defined by the user.
    num_datapoints = 5000
    model, train_indices = get_trained_model(x_train=x_train, y_train=y_train,
                                             num_datapoints=num_datapoints)

    # user extends model class here with their query interface function.
    class CustomModel(Model):
        def get_outputs(self, data):
            return self.model_obj(data)

    target_model = CustomModel(model_obj=model, train_indices=train_indices)

    train_dataset = Dataset(x=x_train[train_indices], y=y_train[train_indices])

    test_indices = list(range(num_datapoints))
    test_dataset = Dataset(x=x_test[test_indices], y=y_test[test_indices])

    population_indices = list(range(num_datapoints, num_datapoints*3))
    population_dataset = Dataset(x=x_train[population_indices], y=y_train[population_indices])

    # user defines the signal computation function.
    # we can include some signal functions as built-ins.
    def loss_signal_func(model: Model, dataset: Dataset):
        print("Getting loss signal using model on dataset...")

        loss_fn = tf.keras.losses.CategoricalCrossentropy(from_logits=True,
                                                          reduction=tf.keras.losses.Reduction.NONE)
        y_true = dataset.y
        y_pred = model.get_outputs(dataset.x)

        return loss_fn(y_true=y_true, y_pred=y_pred)

    # user defines the threshold computation function. we can include some threshold functions as built-ins.
    # e.g. get threshold from array of signals, or fit a continuous distribution over the array of signals.
    def threshold_func(distribution, alpha):
        threshold = np.quantile(distribution, q=alpha, interpolation='lower')
        return threshold

    # user constructs ML Privacy Meter audit object
    audit_obj = Audit(target_model=target_model,
                      target_train_dataset=train_dataset,
                      target_test_dataset=test_dataset,
                      population_dataset=population_dataset,
                      signal_func=loss_signal_func,
                      threshold_func=threshold_func)

    # user prepares the audit and runs it
    audit_obj.prepare_audit()
    alphas = [0.1, 0.2, 0.3]
    audit_obj.run_audit(alphas=alphas)
