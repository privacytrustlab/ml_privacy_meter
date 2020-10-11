# Runs the Approximation Attack 1 : Average Rank Thresholding

import os
import pickle

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from nltk.translate.bleu_score import sentence_bleu
from sklearn import svm
from sklearn.metrics import accuracy_score, roc_curve, roc_auc_score

from seq2seq_model import UNITS, Decoder, Encoder

from ml_privacy_meter.utils.attack_data import attack_data_seq2seq
from ml_privacy_meter.attack.meminf_seq2seq import meminf, attacks

ATTACKER_KNOWLEDGE_RATIO = 0.5
BATCH_SIZE = 64
###################################


max_length_targ, max_length_inp = 65, 67

print('Preparing attack data')
attack_data = attack_data_seq2seq('sated_model/in_input.npy', 'sated_model/in_target.npy', 'sated_model/out_input.npy', 'sated_model/out_target.npy', max_length_inp, max_length_targ)

print('Preparing model')
with open('sated_model/inp_lang.pickle', 'rb') as handle, open('sated_model/targ_lang.pickle', 'rb') as handle2:
    inp_lang = pickle.load(handle)
    targ_lang = pickle.load(handle2)
vocab_inp_size = len(inp_lang.word_index)+1
vocab_tar_size = len(targ_lang.word_index)+1
dec_input_start = targ_lang.word_index['<start>']
dec_input_end = targ_lang.word_index['<end>']

encoder = Encoder(vocab_inp_size, BATCH_SIZE)
decoder = Decoder(vocab_tar_size, BATCH_SIZE)
optimizer = tf.keras.optimizers.Adam()
checkpoint_dir = './sated_model/training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt-10") # TODO : Change this path
checkpoint = tf.train.Checkpoint(optimizer=optimizer,
                                 encoder=encoder,
                                 decoder=decoder)
checkpoint.restore(checkpoint_prefix)

print('Creating attack object')
attackobj = meminf(encoder, decoder, optimizer, attack_data, UNITS, dec_input_start, dec_input_end, model_name='sated', attack_type=attacks.ONE)

print('### Attack 1')
attackobj.attack()
