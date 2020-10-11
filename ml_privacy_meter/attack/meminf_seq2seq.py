import enum

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from progress.bar import IncrementalBar
from sklearn import svm
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve


class attacks(enum.Enum):
    ONE = 'ranks'
    TWO = 'rank_histogram'
    THREE = 'pred_histogram'


class meminf:
    def __init__(self, target_model_encoder, target_model_decoder, target_model_optimizer, datahandler, units, dec_input_start, dec_input_end, model_name='sample_seq2seq', attack_type=attacks.ONE) -> None:
        self.encoder = target_model_encoder
        self.decoder = target_model_decoder
        self.optimizer = target_model_optimizer
        self.attack_data = datahandler
        self.model_name = model_name
        self.attack_type = attack_type
        self.units = units
        self.dec_input_start = dec_input_start
        self.dec_input_end = dec_input_end

    def translate_and_get_indices(self, target, pred_probs):
        ### score = sentence_bleu([tr.split()], res.split())
        indices = []
        for word_index, prob in zip(target, pred_probs):
            temp = (-prob).argsort()[:len(prob)]
            ranks = np.empty_like(temp)
            ranks[temp] = np.arange(len(prob))
            indices.append(ranks[word_index])
        return indices

    def attack1(self):
        bar = IncrementalBar('processing in_train_indices',
                             max=len(self.attack_data.in_target_train))
        in_train_indices = []
        for input, target in zip(self.attack_data.in_input_train, self.attack_data.in_target_train):
            _, _, pred_probs = self.predict(input, True)
            indices = self.translate_and_get_indices(target, pred_probs)
            in_train_indices.append(np.mean(indices))
            bar.next()
        bar.finish()

        bar = IncrementalBar('processing out_train_indices',
                             max=len(self.attack_data.out_target_train))
        out_train_indices = []
        for input, target in zip(self.attack_data.out_input_train, self.attack_data.out_target_train):
            _, _, pred_probs = self.predict(input, True)
            indices = self.translate_and_get_indices(target, pred_probs)
            out_train_indices.append(np.mean(indices))
            bar.next()
        bar.finish()

        bar = IncrementalBar('processing in_test_indices',
                             max=len(self.attack_data.in_target_test))
        in_test_indices = []
        for input, target in zip(self.attack_data.in_input_test, self.attack_data.in_target_test):
            _, _, pred_probs = self.predict(input, True)
            indices = self.translate_and_get_indices(target, pred_probs)
            in_test_indices.append(np.mean(indices))
            bar.next()
        bar.finish()

        bar = IncrementalBar('processing out_test_indices',
                             max=len(self.attack_data.out_target_test))
        out_test_indices = []
        for input, target in zip(self.attack_data.out_input_test, self.attack_data.out_target_test):
            _, _, pred_probs = self.predict(input, True)
            indices = self.translate_and_get_indices(target, pred_probs)
            out_test_indices.append(np.mean(indices))
            bar.next()
        bar.finish()

        print("creating x_train and y_train")
        x_train = np.concatenate([in_train_indices, out_train_indices])
        y_train = [1. for _ in range(len(in_train_indices))]
        y_train.extend([0. for _ in range(len(out_train_indices))])

        print("creating x_test and y_test")
        x_test = np.concatenate([in_test_indices, out_test_indices])
        y_test = [1. for _ in range(len(in_test_indices))]
        y_test.extend([0. for _ in range(len(out_test_indices))])

        print("fitting classifier")
        clf = svm.SVC()
        clf.fit(x_train.reshape(-1, 1), y_train)
        y_pred = clf.predict(x_test.reshape(-1, 1))
        print("Attack 1 Accuracy : %.2f%%" %
              (100.0 * accuracy_score(y_test, y_pred)))

        ra_score = roc_auc_score(y_test, y_pred)
        print("Attack 1 ROC_AUC Score : %.2f%%" % (100.0 * ra_score))

        fpr, tpr, thresholds = roc_curve(y_test, y_pred, pos_label=1)
        plt.figure(1)
        plt.plot(fpr, tpr, label='Attack 1')
        plt.xlabel('False positive rate')
        plt.ylabel('True positive rate')
        plt.title('ROC curve')
        plt.savefig('satedrecord_attack1_roc_curve.png')

    def attack(self):
        if self.attack_type == attacks.ONE:
            self.attack1()
        else:
            raise NotImplementedError('Attack type not implemented yet')

    def predict(self, input, tensor=False):
        attention_plot = np.zeros(
            (self.attack_data.max_length_targ, self.attack_data.max_length_inp))
        inputs = tf.keras.preprocessing.sequence.pad_sequences([input],
                                                               maxlen=self.attack_data.max_length_inp,
                                                               padding='post')
        inputs = tf.convert_to_tensor(inputs)

        result = ''

        hidden = [tf.zeros((1, self.units))]
        enc_out, enc_hidden = self.encoder(inputs, hidden)

        dec_hidden = enc_hidden
        dec_input = tf.expand_dims([self.dec_input_start], 0)

        pred_probs = []

        for t in range(self.attack_data.max_length_targ):
            predictions, dec_hidden, attention_weights = self.decoder(dec_input,
                                                                      dec_hidden,
                                                                      enc_out)

            attention_weights = tf.reshape(attention_weights, (-1, ))
            attention_plot[t] = attention_weights.numpy()

            predicted_id = tf.argmax(predictions[0]).numpy()
            pred_probs.append(predictions[0].numpy())

            # finish prediction if <end> token is predicted
            # don't perform check if 0 is predicted as it's reserved for padding (word index won't have key = 0)
            if predicted_id == self.dec_input_end:
                return result, attention_plot, pred_probs

            dec_input = tf.expand_dims([predicted_id], 0)

        return result, attention_plot, pred_probs
