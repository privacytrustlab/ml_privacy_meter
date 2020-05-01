import numpy as np
import tensorflow as tf
from sklearn.utils import shuffle


def compute_hashes(to_compute):
    """
    """
    hasharr = []
    for arr in to_compute:
        hashval = hash(bytes(arr))
        hasharr.append(hashval)
    return hasharr


def get_tfdataset(features, labels):
    """
    """
    return tf.data.Dataset.from_tensor_slices((features, labels))


class attack_data:
    """
    """

    def __init__(self, dataset_path, member_dataset_path, batch_size,
                 attack_percentage, normalization=False,
                 input_shape=None):

        self.batch_size = batch_size

        # Loading the training (member) dataset
        self.train_data = np.load(member_dataset_path)
        self.training_size = len(self.train_data)

        self.attack_size = int(attack_percentage / float(100) * self.training_size)

        # Specifically for image datasets
        self.input_shape = input_shape

        self.normalization = normalization

        # Loading and shuffle the dataset
        self.dataset = self._extract(dataset_path)
        np.random.shuffle(self.dataset)

        self.input_channels = self.input_shape[-1]

        # To avoid using any of training examples for testing
        self.train_hashes = compute_hashes(self.train_data)

    def _extract(self, filepath):
        """
        """
        with open(filepath, "r") as f:
            dataset = f.readlines()
        dataset = list(map(lambda i: i.strip('\n').split(';'), dataset))
        dataset = np.asarray(dataset)
        return dataset

    def generate(self, dataset):
        """
        Parses each record of the dataset and extracts 
        the class (first column of the record) and the 
        features. This assumes 'csv' form of data.
        """
        features, labels = dataset[:, :-1], dataset[:, -1]
        features = map(lambda y: np.array(
            list(map(lambda i: i.split(","), y))).flatten(), features)
        features = np.array(list(features))

        features = np.ndarray.astype(features, np.float32)

        if self.input_shape:
            if len(self.input_shape) == 3:
                reshape_input = (len(features),) + (self.input_shape[2], self.input_shape[0], self.input_shape[1])
                features = np.transpose(np.reshape(features, reshape_input), (0, 2, 3, 1))
            else:
                reshape_input = (len(features),) + self.input_shape
                features = np.reshape(features, reshape_input)
        labels = np.ndarray.astype(labels, np.float32)
        return features, labels

    def compute_moments(self, f):
        """
        """
        self.means = []
        self.stddevs = []
        for i in range(self.input_channels):
            # very specific to 3-dimensional input
            pixels = f[:, :, :, i].ravel()
            self.means.append(np.mean(pixels, dtype=np.float32))
            self.stddevs.append(np.std(pixels, dtype=np.float32))
        self.means = list(map(lambda i: np.float32(i/255), self.means))
        self.stddevs = list(map(lambda i: np.float32(i/255), self.stddevs))

    def normalize(self, f):
        """
        """
        normalized = (f/255 - self.means) / self.stddevs
        return normalized

    def load_train(self):
        """
        Returns a tf.data.Dataset object for training
        """
        tsize = self.training_size
        asize = self.attack_size
        member_train = self.train_data[:asize]
        self.nonmember_train = []

        index = 0

        while len(self.nonmember_train) != len(member_train) and index < len(self.dataset):
            datapoint = self.dataset[index]
            datapointhash = hash(bytes(datapoint))
            if datapointhash not in self.train_hashes:
                self.nonmember_train.append(datapoint)
            index += 1
        self.nonmember_train = np.vstack(self.nonmember_train)

        m_features, m_labels = self.generate(member_train)
        nm_features, nm_labels = self.generate(self.nonmember_train)
        if self.normalization:
            train_features, _ = self.generate(self.train_data)
            if not self.means and not self.stddevs:
                self.compute_moments(train_features)
            m_features = self.normalize(m_features)
            nm_features = self.normalize(nm_features)

        mtrain = get_tfdataset(m_features, m_labels)
        nmtrain = get_tfdataset(nm_features, nm_labels)

        mtrain = mtrain.batch(self.batch_size)
        nmtrain = nmtrain.batch(self.batch_size)

        return mtrain, nmtrain, nm_features, nm_labels

    def load_vis(self, batch_size=256):
        """
        Returns a tf.data.Dataset object for visualization testing
        """
        member_train = self.train_data
        self.nonmember_train = []

        index = 0

        while len(self.nonmember_train) != len(member_train) and index < len(self.dataset):
            datapoint = self.dataset[index]
            datapointhash = hash(bytes(datapoint))
            if datapointhash not in self.train_hashes:
                self.nonmember_train.append(datapoint)
            index += 1
        self.nonmember_train = np.vstack(self.nonmember_train)

        m_features, m_labels = self.generate(member_train)
        nm_features, nm_labels = self.generate(self.nonmember_train)
        if self.normalization:
            train_features, _ = self.generate(self.train_data)
            if not self.means and not self.stddevs:
                self.compute_moments(train_features)
            m_features = self.normalize(m_features)
            nm_features = self.normalize(nm_features)

        np.save('logs/m_features', m_features)
        np.save('logs/m_labels', m_labels)
        np.save('logs/nm_features', nm_features)
        np.save('logs/nm_labels', nm_labels)

        mtrain = get_tfdataset(m_features, m_labels)
        nmtrain = get_tfdataset(nm_features, nm_labels)

        mtrain = mtrain.batch(batch_size)
        nmtrain = nmtrain.batch(batch_size)

        return mtrain, nmtrain, nm_features, nm_labels


    def load_test(self):
        """
        Returns a tf.data.Dataset object for testing
        """
        tsize = self.training_size
        asize = self.attack_size

        member_test = self.train_data[asize:] 
        nonmember_test = []

        nmtrainhashes = compute_hashes(self.nonmember_train)
        index = 0
        while len(nonmember_test) != len(member_test) and index < len(self.dataset):
            datapoint = self.dataset[index]
            datapointhash = hash(bytes(datapoint))
            if (datapointhash not in self.train_hashes and
                    datapointhash not in nmtrainhashes):
                nonmember_test.append(datapoint)
            index += 1
        nonmember_test = np.vstack(nonmember_test)

        m_features, m_labels = self.generate(member_test)
        nm_features, nm_labels = self.generate(nonmember_test)

        if self.normalization:
            m_features = self.normalize(m_features)
            nm_features = self.normalize(nm_features)

        mtest = get_tfdataset(m_features, m_labels)
        nmtest = get_tfdataset(nm_features, nm_labels)

        mtest = mtest.batch(self.batch_size)
        nmtest = nmtest.batch(self.batch_size)

        return mtest, nmtest

