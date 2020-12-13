# Code adapted from https://github.com/csong27/auditing-text-generation

import numpy as np
from collections import Counter, defaultdict
from itertools import chain

import tensorflow.keras.backend as K
from tensorflow.keras.layers import Layer, InputSpec
from tensorflow.keras import activations, initializers, regularizers, constraints
from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Embedding, LSTM, Dropout, Dense, Add
from tensorflow.keras.regularizers import l2

SATED_PATH = 'sated-release-0.9.0/en-fr/'
SATED_TRAIN_ENG = SATED_PATH + 'train.en'
SATED_TRAIN_FR = SATED_PATH + 'train.fr'
SATED_TRAIN_USER = SATED_PATH + 'train.usr'
SATED_DEV_ENG = SATED_PATH + 'dev.en'
SATED_DEV_FR = SATED_PATH + 'dev.fr'
SATED_DEV_USER = SATED_PATH + 'dev.usr'
SATED_TEST_ENG = SATED_PATH + 'test.en'
SATED_TEST_FR = SATED_PATH + 'test.fr'


# ================================ NMT MODEL UTILS ================================ #

class Attention(Layer):
    def __init__(self, units,
                 activation='linear',
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 **kwargs):
        if 'input_shape' not in kwargs and 'input_dim' in kwargs:
            kwargs['input_shape'] = (kwargs.pop('input_dim'),)
        super(Attention, self).__init__(**kwargs)
        self.units = units
        self.activation = activations.get(activation)
        self.use_bias = use_bias
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.activity_regularizer = regularizers.get(activity_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.bias_constraint = constraints.get(bias_constraint)
        self.supports_masking = True

    def build(self, input_shape):
        if not isinstance(input_shape, list) or len(input_shape) != 2:
            raise ValueError('An attention layer should be called '
                             'on a list of 2 inputs.')
        enc_dim = input_shape[0][-1]
        dec_dim = input_shape[1][-1]

        self.W_enc = self.add_weight(shape=(enc_dim, self.units),
                                     initializer=self.kernel_initializer,
                                     name='W_enc',
                                     regularizer=self.kernel_regularizer,
                                     constraint=self.kernel_constraint)

        self.W_dec = self.add_weight(shape=(dec_dim, self.units),
                                     initializer=self.kernel_initializer,
                                     name='W_dec',
                                     regularizer=self.kernel_regularizer,
                                     constraint=self.kernel_constraint)

        self.W_score = self.add_weight(shape=(self.units, 1),
                                       initializer=self.kernel_initializer,
                                       name='W_score',
                                       regularizer=self.kernel_regularizer,
                                       constraint=self.kernel_constraint)

        if self.use_bias:
            self.bias_enc = self.add_weight(shape=(self.units,),
                                            initializer=self.bias_initializer,
                                            name='bias_enc',
                                            regularizer=self.bias_regularizer,
                                            constraint=self.bias_constraint)
            self.bias_dec = self.add_weight(shape=(self.units,),
                                            initializer=self.bias_initializer,
                                            name='bias_dec',
                                            regularizer=self.bias_regularizer,
                                            constraint=self.bias_constraint)
            self.bias_score = self.add_weight(shape=(1,),
                                              initializer=self.bias_initializer,
                                              name='bias_score',
                                              regularizer=self.bias_regularizer,
                                              constraint=self.bias_constraint)

        else:
            self.bias_enc = None
            self.bias_dec = None
            self.bias_score = None

        self.built = True

    def call(self, inputs, **kwargs):
        if not isinstance(inputs, list) or len(inputs) != 2:
            raise ValueError('An attention layer should be called '
                             'on a list of 2 inputs.')
        encodings, decodings = inputs
        d_enc = K.dot(encodings, self.W_enc)
        d_dec = K.dot(decodings, self.W_dec)

        if self.use_bias:
            d_enc = K.bias_add(d_enc, self.bias_enc)
            d_dec = K.bias_add(d_dec, self.bias_dec)

        if self.activation is not None:
            d_enc = self.activation(d_enc)
            d_dec = self.activation(d_dec)

        enc_seqlen = K.shape(d_enc)[1]
        d_dec_shape = K.shape(d_dec)

        stacked_d_dec = K.tile(d_dec, [enc_seqlen, 1, 1])  # enc time x batch x dec time  x da
        stacked_d_dec = K.reshape(stacked_d_dec, [enc_seqlen, d_dec_shape[0], d_dec_shape[1], d_dec_shape[2]])
        stacked_d_dec = K.permute_dimensions(stacked_d_dec, [2, 1, 0, 3])  # dec time x batch x enc time x da
        tanh_add = K.tanh(stacked_d_dec + d_enc)  # dec time x batch x enc time x da
        scores = K.dot(tanh_add, self.W_score)
        if self.use_bias:
            scores = K.bias_add(scores, self.bias_score)
        scores = K.squeeze(scores, 3)  # batch x dec time x enc time

        weights = K.softmax(scores)  # dec time x batch x enc time
        weights = K.expand_dims(weights)

        weighted_encodings = weights * encodings  # dec time x batch x enc time x h
        contexts = K.sum(weighted_encodings, axis=2)  # dec time x batch x h
        contexts = K.permute_dimensions(contexts, [1, 0, 2])  # batch x dec time x h

        return contexts

    def compute_output_shape(self, input_shape):
        assert isinstance(input_shape, list) and len(input_shape) == 2
        assert input_shape[-1]
        output_shape = list(input_shape[1])
        output_shape[-1] = self.units
        return tuple(output_shape)

    def get_config(self):
        config = {
            'units': self.units,
            'activation': activations.serialize(self.activation),
            'use_bias': self.use_bias,
            'kernel_initializer': initializers.serialize(self.kernel_initializer),
            'bias_initializer': initializers.serialize(self.bias_initializer),
            'kernel_regularizer': regularizers.serialize(self.kernel_regularizer),
            'bias_regularizer': regularizers.serialize(self.bias_regularizer),
            'activity_regularizer': regularizers.serialize(self.activity_regularizer),
            'kernel_constraint': constraints.serialize(self.kernel_constraint),
            'bias_constraint': constraints.serialize(self.bias_constraint)
        }
        base_config = super(Attention, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class DenseTransposeTied(Layer):
    def __init__(self, units,
                 tied_to=None,  # Enter a layer as input to enforce weight-tying
                 activation=None,
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 **kwargs):
        if 'input_shape' not in kwargs and 'input_dim' in kwargs:
            kwargs['input_shape'] = (kwargs.pop('input_dim'),)
        super(DenseTransposeTied, self).__init__(**kwargs)
        self.units = units
        # We add these two properties to save the tied weights
        self.tied_to = tied_to
        self.tied_weights = self.tied_to.weights
        self.activation = activations.get(activation)
        self.use_bias = use_bias
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.activity_regularizer = regularizers.get(activity_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.bias_constraint = constraints.get(bias_constraint)
        self.input_spec = InputSpec(min_ndim=2)
        self.supports_masking = True

    def build(self, input_shape):
        assert len(input_shape) >= 2
        input_dim = input_shape[-1]

        # We remove the weights and bias because we do not want them to be trainable
        self.input_spec = InputSpec(min_ndim=2, axes={-1: input_dim})
        if self.use_bias:
            self.bias = self.add_weight(shape=(self.units,),
                                        initializer=self.bias_initializer,
                                        name='bias',
                                        regularizer=self.bias_regularizer,
                                        constraint=self.bias_constraint)
        else:
            self.bias = None
        self.built = True

    def call(self, inputs, **kwargs):
        # Return the transpose layer mapping using the explicit weight matrices
        output = K.dot(inputs, K.transpose(self.tied_weights[0]))
        if self.use_bias:
            output = K.bias_add(output, self.bias, data_format='channels_last')

        if self.activation is not None:
            output = self.activation(output)

        return output

    def compute_output_shape(self, input_shape):
        assert input_shape and len(input_shape) >= 2
        assert input_shape[-1]
        output_shape = list(input_shape)
        output_shape[-1] = self.units
        return tuple(output_shape)

    def get_config(self):
        config = {
            'units': self.units,
            'activation': activations.serialize(self.activation),
            'use_bias': self.use_bias,
            'kernel_initializer': initializers.serialize(self.kernel_initializer),
            'bias_initializer': initializers.serialize(self.bias_initializer),
            'kernel_regularizer': regularizers.serialize(self.kernel_regularizer),
            'bias_regularizer': regularizers.serialize(self.bias_regularizer),
            'activity_regularizer': regularizers.serialize(self.activity_regularizer),
            'kernel_constraint': constraints.serialize(self.kernel_constraint),
            'bias_constraint': constraints.serialize(self.bias_constraint)
        }
        base_config = super(DenseTransposeTied, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


def build_nmt_model(Vs, Vt, demb=128, h=128, drop_p=0.5, tied=True, mask=True, attn=True, l2_ratio=1e-4,
                    training=None, rnn_fn='lstm'):
    """
    Builds the target machine translation model.

    :param demb: Embedding dimension.
    :param h: Number of hidden units.
    :param drop_p: Dropout percentage.
    :param attn: Flag to include attention units.
    :param rnn_fn: Can be 'lstm' or 'gru'.
    """
    if rnn_fn == 'lstm':
        rnn = LSTM
    elif rnn_fn == 'gru':
        rnn = LSTM
    else:
        raise ValueError(rnn_fn)

    # Build encoder
    encoder_input = Input((None,), dtype='float32', name='encoder_input')
    if mask:
        encoder_emb_layer = Embedding(Vs + 1, demb, mask_zero=True, embeddings_regularizer=l2(l2_ratio),
                                      name='encoder_emb')
    else:
        encoder_emb_layer = Embedding(Vs, demb, mask_zero=False, embeddings_regularizer=l2(l2_ratio),
                                      name='encoder_emb')

    encoder_emb = encoder_emb_layer(encoder_input)

    # Dropout for encoder
    if drop_p > 0.:
        encoder_emb = Dropout(drop_p)(encoder_emb, training=training)

    encoder_rnn = rnn(h, return_sequences=True, return_state=True, kernel_regularizer=l2(l2_ratio), name='encoder_rnn')
    encoder_rtn = encoder_rnn(encoder_emb)
    encoder_outputs = encoder_rtn[0]
    encoder_states = encoder_rtn[1:]

    # Build decoder
    decoder_input = Input((None,), dtype='float32', name='decoder_input')
    if mask:
        decoder_emb_layer = Embedding(Vt + 1, demb, mask_zero=True, embeddings_regularizer=l2(l2_ratio),
                                      name='decoder_emb')
    else:
        decoder_emb_layer = Embedding(Vt, demb, mask_zero=False, embeddings_regularizer=l2(l2_ratio),
                                      name='decoder_emb')

    decoder_emb = decoder_emb_layer(decoder_input)

    # Dropout for decoder
    if drop_p > 0.:
        decoder_emb = Dropout(drop_p)(decoder_emb, training=training)

    decoder_rnn = rnn(h, return_sequences=True, kernel_regularizer=l2(l2_ratio), name='decoder_rnn')
    decoder_outputs = decoder_rnn(decoder_emb, initial_state=encoder_states)

    if drop_p > 0.:
        decoder_outputs = Dropout(drop_p)(decoder_outputs, training=training)

    # Taken from https://arxiv.org/pdf/1805.01817.pdf for training with user annotations
    if tied:
        final_outputs = DenseTransposeTied(Vt, kernel_regularizer=l2(l2_ratio), name='outputs',
                                           tied_to=decoder_emb_layer, activation='linear')(decoder_outputs)
    else:
        final_outputs = Dense(Vt, activation='linear', kernel_regularizer=l2(l2_ratio), name='outputs')(decoder_outputs)

    # Add attention units
    if attn:
        contexts = Attention(units=h, kernel_regularizer=l2(l2_ratio), name='attention',
                             use_bias=False)([encoder_outputs, decoder_outputs])
        if drop_p > 0.:
            contexts = Dropout(drop_p)(contexts, training=training)

        contexts_outputs = Dense(Vt, activation='linear', use_bias=False, name='context_outputs',
                                 kernel_regularizer=l2(l2_ratio))(contexts)

        final_outputs = Add(name='final_outputs')([final_outputs, contexts_outputs])

    model = Model(inputs=[encoder_input, decoder_input], outputs=[final_outputs])
    return model


def words_to_indices(data, vocab, mask=True):
    """
    Converts words to indices according to vocabulary.
    """
    if mask:
        return [[vocab[w] + 1 for w in t] for t in data]
    else:
        return [[vocab[w] for w in t] for t in data]


# ================================  DATA LOADER UTILS ================================ #
def load_users(p=SATED_TRAIN_USER):
    users = []
    with open(p, 'r', encoding='UTF-8') as f:
        for line in f:
            users.append(line.replace('\n', ''))
    return users


def load_texts(p=SATED_TRAIN_ENG):
    texts = []
    with open(p, 'r', encoding='UTF-8') as f:
        for line in f:
            arr = ['<sos>'] + line.replace('\n', '').split(' ') + ['<eos>']
            words = []
            for w in arr:
                words.append(w)
            texts.append(words)

    return texts


def process_texts(texts, vocabs):
    for t in texts:
        for i, w in enumerate(t):
            if w not in vocabs:
                t[i] = '<unk>'


def process_vocabs(vocabs, num_words=10000):
    counter = Counter(vocabs)
    count_pairs = sorted(counter.items(), key=lambda x: (-x[1], x[0]))
    print('Loaded {} vocabs'.format(len(count_pairs)))

    if num_words is not None:
        count_pairs = count_pairs[:num_words - 1]

    print(f"Count pairs (first 50): {count_pairs[:50]}".encode('UTF-8'))

    words, _ = list(zip(*count_pairs))
    word_to_id = dict(zip(words, np.arange(len(words))))
    return word_to_id


def load_sated_data_by_user(num_users=100, num_words=10000, test_on_user=False, sample_user=False,
                            seed=12345, user_data_ratio=0.):
    src_users = load_users(SATED_TRAIN_USER)
    train_src_texts = load_texts(SATED_TRAIN_ENG)
    train_trg_texts = load_texts(SATED_TRAIN_FR)

    dev_src_texts = load_texts(SATED_DEV_ENG)
    dev_trg_texts = load_texts(SATED_DEV_FR)

    test_src_texts = load_texts(SATED_TEST_ENG)
    test_trg_texts = load_texts(SATED_TEST_FR)

    user_counter = Counter(src_users)
    all_users = [tup[0] for tup in user_counter.most_common()]

    np.random.seed(seed)
    np.random.shuffle(all_users)
    np.random.seed(None)

    train_users = set(all_users[:num_users])
    test_users = set(all_users[num_users: num_users * 2])

    if sample_user:
        attacker_users = all_users[num_users * 2: num_users * 4]

        train_users = np.random.choice(attacker_users, size=num_users, replace=False)
        print(len(train_users))
        print(train_users[:10])

    user_src_texts = defaultdict(list)
    user_trg_texts = defaultdict(list)

    test_user_src_texts = defaultdict(list)
    test_user_trg_texts = defaultdict(list)

    for u, s, t in zip(src_users, train_src_texts, train_trg_texts):
        if u in train_users:
            user_src_texts[u].append(s)
            user_trg_texts[u].append(t)
        if test_on_user and u in test_users:
            test_user_src_texts[u].append(s)
            test_user_trg_texts[u].append(t)

    if 0. < user_data_ratio < 1.:
        # held out some fraction of data for testing
        for u in user_src_texts:
            l = len(user_src_texts[u])

            l = int(l * user_data_ratio)
            user_src_texts[u] = user_src_texts[u][:l]
            user_trg_texts[u] = user_trg_texts[u][:l]

    src_words = []
    trg_words = []
    for u in train_users:
        src_words += list(chain(*user_src_texts[u]))
        trg_words += list(chain(*user_trg_texts[u]))

    src_vocabs = process_vocabs(src_words, num_words)
    trg_vocabs = process_vocabs(trg_words, num_words)

    for u in train_users:
        process_texts(user_src_texts[u], src_vocabs)
        process_texts(user_trg_texts[u], trg_vocabs)

    if test_on_user:
        for u in test_users:
            process_texts(test_user_src_texts[u], src_vocabs)
            process_texts(test_user_trg_texts[u], trg_vocabs)

    process_texts(dev_src_texts, src_vocabs)
    process_texts(dev_trg_texts, trg_vocabs)

    process_texts(test_src_texts, src_vocabs)
    process_texts(test_trg_texts, trg_vocabs)

    src_words = []
    trg_words = []
    for u in train_users:
        src_words += list(chain(*user_src_texts[u]))
        trg_words += list(chain(*user_trg_texts[u]))

    src_vocabs = process_vocabs(src_words, None)
    trg_vocabs = process_vocabs(trg_words, None)

    if test_on_user:
        return user_src_texts, user_trg_texts, test_user_src_texts, test_user_trg_texts, src_vocabs, trg_vocabs
    else:
        return user_src_texts, user_trg_texts, dev_src_texts, dev_trg_texts, test_src_texts, test_trg_texts, \
               src_vocabs, trg_vocabs
