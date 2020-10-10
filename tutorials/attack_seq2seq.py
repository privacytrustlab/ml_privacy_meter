# Approximation Attack 1 : Average Rank Thresholding

import os
import pickle

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from nltk.translate.bleu_score import sentence_bleu
from sklearn import svm
from sklearn.metrics import accuracy_score, roc_curve, roc_auc_score

from seq2seq import UNITS, Decoder, Encoder
from translate import Translate

BATCH_SIZE = 64
###################################

with open('sated_model/inp_lang.pickle', 'rb') as handle, open('sated_model/targ_lang.pickle', 'rb') as handle2:
    inp_lang = pickle.load(handle)
    targ_lang = pickle.load(handle2)


in_train, in_train_label = np.load(
    'sated_model/in_train.npy'), np.load('sated_model/in_train_label.npy')
out_train, out_train_label = np.load(
    'sated_model/out_train.npy'), np.load('sated_model/out_train_label.npy')
in_test, in_test_label = np.load(
    'sated_model/in_test.npy'), np.load('sated_model/in_test_label.npy')
out_test, out_test_label = np.load(
    'sated_model/out_test.npy'), np.load('sated_model/out_test_label.npy')

vocab_inp_size = len(inp_lang.word_index)+1
vocab_tar_size = len(targ_lang.word_index)+1

encoder = Encoder(vocab_inp_size, BATCH_SIZE)
decoder = Decoder(vocab_tar_size, BATCH_SIZE)
spa_eng_max_length_targ, spa_eng_max_length_inp = 11, 16
max_length_targ, max_length_inp = 65, 67
optimizer = tf.keras.optimizers.Adam()

checkpoint_dir = './sated_model/training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(optimizer=optimizer,
                                 encoder=encoder,
                                 decoder=decoder)


minimum = min(len(in_train), len(out_train))


def translate_and_get_indices(tr, tar, pred_probs):
    res = ''
    for word in tar:
        if word != 0:
            res += targ_lang.index_word[word] + ' '
    res = res.split(' ', 1)[1]

    ### score = sentence_bleu([tr.split()], res.split())

    indices = []

    for word, prob in zip(res.split(), pred_probs):
        temp = (-prob).argsort()[:len(prob)]
        ranks = np.empty_like(temp)
        ranks[temp] = np.arange(len(prob))
        ind = targ_lang.word_index[word]
        indices.append(ranks[ind])

    return indices


translator = Translate(encoder, decoder, UNITS,
                       inp_lang, targ_lang, max_length_targ, max_length_inp)
in_train_indices = []
i = 0
print("processing in_train_indices")
for ten, tar in zip(in_train, in_train_label):
    i += 1
    if i > minimum:
        break
    tr, pred_probs = translator.translate(ten, True)
    indices = translate_and_get_indices(tr, tar, pred_probs)
    in_train_indices.append(np.mean(indices))


out_train_indices = []
i = 0
print("processing out_train_indices")
for ten, tar in zip(out_train, out_train_label):
    i += 1
    if i > minimum:
        break
    tr, pred_probs = translator.translate(ten, True)
    indices = translate_and_get_indices(tr, tar, pred_probs)
    out_train_indices.append(np.mean(indices))

in_test_indices = []
i = 0
print("processing in_test_indices")
for ten, tar in zip(in_test, in_test_label):
    i += 1
    if i > minimum:
        break
    tr, pred_probs = translator.translate(ten, True)
    indices = translate_and_get_indices(tr, tar, pred_probs)
    in_test_indices.append(np.mean(indices))

out_test_indices = []
i = 0
print("processing out_test_indices")
for ten, tar in zip(out_test, out_test_label):
    i += 1
    if i > minimum:
        break
    tr, pred_probs = translator.translate(ten, True)
    indices = translate_and_get_indices(tr, tar, pred_probs)
    out_test_indices.append(np.mean(indices))

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
print("Attack 1 Accuracy : %.2f%%" % (100.0 * accuracy_score(y_test, y_pred)))

ra_score = roc_auc_score(y_test, y_pred)
print("Attack 1 ROC_AUC Score : %.2f%%" % (100.0 * ra_score))

fpr, tpr, thresholds = roc_curve(y_test, y_pred, pos_label=1)
plt.figure(1)
plt.plot(fpr, tpr, label='Attack 1')
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('ROC curve')
plt.savefig('satedrecord_attack1_roc_curve.png')
