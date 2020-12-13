import os
import sys
from collections import defaultdict

import tensorflow as tf
import tensorflow.keras.backend as K
import numpy as np
import scipy.stats as ss
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve
from sklearn.linear_model import LogisticRegression


from utils import process_texts, load_texts, load_users, load_sated_data_by_user, \
    build_nmt_model, words_to_indices, \
    SATED_TRAIN_USER, SATED_TRAIN_FR, SATED_TRAIN_ENG


MODEL_PATH = 'checkpoints/model/'
OUTPUT_PATH = 'checkpoints/output/'

tf.compat.v1.disable_eager_execution()

# ================================ GENERATE RANKS ================================ #
# Code adapted from https://github.com/csong27/auditing-text-generation
def load_train_users_heldout_data(train_users, src_vocabs, trg_vocabs, user_data_ratio=0.5):
    src_users = load_users(SATED_TRAIN_USER)
    train_src_texts = load_texts(SATED_TRAIN_ENG)
    train_trg_texts = load_texts(SATED_TRAIN_FR)

    user_src_texts = defaultdict(list)
    user_trg_texts = defaultdict(list)

    for u, s, t in zip(src_users, train_src_texts, train_trg_texts):
        if u in train_users:
            user_src_texts[u].append(s)
            user_trg_texts[u].append(t)

    assert 0. < user_data_ratio < 1.
    # Hold out some fraction of data for testing
    for u in user_src_texts:
        l = len(user_src_texts[u])
        l = int(l * user_data_ratio)
        user_src_texts[u] = user_src_texts[u][l:]
        user_trg_texts[u] = user_trg_texts[u][l:]

    for u in train_users:
        process_texts(user_src_texts[u], src_vocabs)
        process_texts(user_trg_texts[u], trg_vocabs)

    return user_src_texts, user_trg_texts


def rank_lists(lists):
    ranks = np.empty_like(lists)
    for i, l in enumerate(lists):
        ranks[i] = ss.rankdata(l, method='min') - 1
    return ranks


def get_ranks(user_src_data, user_trg_data, pred_fn, save_probs=False):
    indices = np.arange(len(user_src_data))
    """
    Get ranks from prediction vectors.
    """

    ranks = []
    labels = []
    probs = []
    for idx in indices:
        src_text = np.asarray(user_src_data[idx], dtype=np.float32).reshape(1, -1)
        trg_text = np.asarray(user_trg_data[idx], dtype=np.float32)
        trg_input = trg_text[:-1].reshape(1, -1)
        trg_label = trg_text[1:].reshape(1, -1)

        prob = pred_fn([src_text, trg_input, trg_label, 0])[0][0]
        if save_probs:
            probs.append(prob)

        all_ranks = rank_lists(-prob)
        sent_ranks = all_ranks[np.arange(len(all_ranks)), trg_label.flatten().astype(int)]

        ranks.append(sent_ranks)
        labels.append(trg_label.flatten())

    if save_probs:
        return probs

    return ranks, labels


def save_users_rank_results(users, user_src_texts, user_trg_texts, src_vocabs, trg_vocabs, prob_fn, save_dir,
                            member_label=1, cross_domain=False, save_probs=False, mask=False, rerun=False):
    """
    Save user ranks in the appropriate format for attacks.
    """
    for i, u in enumerate(users):
        save_path = save_dir + 'rank_u{}_y{}{}.npz'.format(i, member_label, '_cd' if cross_domain else '')
        prob_path = save_dir + 'prob_u{}_y{}{}.npz'.format(i, member_label, '_cd' if cross_domain else '')

        if os.path.exists(save_path) and not save_probs and not rerun:
            continue

        user_src_data = words_to_indices(user_src_texts[u], src_vocabs, mask=mask)
        user_trg_data = words_to_indices(user_trg_texts[u], trg_vocabs, mask=mask)

        rtn = get_ranks(user_src_data, user_trg_data, prob_fn, save_probs=save_probs)

        if save_probs:
            probs = rtn
            np.savez(prob_path, probs)
        else:
            ranks, labels = rtn[0], rtn[1]
            np.savez(save_path, ranks, labels)

        if (i + 1) % 500 == 0:
            sys.stderr.write('Finishing saving ranks for {} users'.format(i + 1))


def get_target_ranks(num_users=200, num_words=5000, mask=False, h=128, emb_h=128, user_data_ratio=0.,
                     tied=False, save_probs=False):
    """
    Get ranks of target machine translation model.
    """
    user_src_texts, user_trg_texts, test_user_src_texts, test_user_trg_texts, src_vocabs, trg_vocabs \
        = load_sated_data_by_user(num_users, num_words, test_on_user=True, user_data_ratio=user_data_ratio)

    train_users = sorted(user_src_texts.keys())
    test_users = sorted(test_user_src_texts.keys())

    # Get model
    save_dir = OUTPUT_PATH + 'target_{}{}/'.format(num_users, '_dr' if 0. < user_data_ratio < 1. else '')
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    model_path = 'sated_nmt'.format(num_users)

    if 0. < user_data_ratio < 1.:
        model_path += '_dr{}'.format(user_data_ratio)
        heldout_src_texts, heldout_trg_texts = load_train_users_heldout_data(train_users, src_vocabs, trg_vocabs)
        for u in train_users:
            user_src_texts[u] += heldout_src_texts[u]
            user_trg_texts[u] += heldout_trg_texts[u]

    model = build_nmt_model(Vs=num_words, Vt=num_words, mask=mask, drop_p=0., h=h, demb=emb_h, tied=tied)
    model.load_weights(MODEL_PATH + '{}_{}.h5'.format(model_path, num_users))

    src_input_var, trg_input_var = model.inputs
    prediction = model.output
    trg_label_var = K.placeholder((None, None), dtype='float32')

    # Get predictions
    prediction = K.softmax(prediction)
    prob_fn = K.function([src_input_var, trg_input_var, trg_label_var, K.learning_phase()], [prediction])

    # Save user ranks for train and test dataset
    save_users_rank_results(users=train_users, save_probs=save_probs,
                            user_src_texts=user_src_texts, user_trg_texts=user_trg_texts,
                            src_vocabs=src_vocabs, trg_vocabs=trg_vocabs, cross_domain=False,
                            prob_fn=prob_fn, save_dir=save_dir, member_label=1)
    save_users_rank_results(users=test_users, save_probs=save_probs,
                            user_src_texts=test_user_src_texts, user_trg_texts=test_user_trg_texts,
                            src_vocabs=src_vocabs, trg_vocabs=trg_vocabs, cross_domain=False,
                            prob_fn=prob_fn, save_dir=save_dir, member_label=0)


# ================================ ATTACK ================================ #
def avg_rank_feats(ranks):
    """
    Averages ranks to get features for deciding the threshold for membership inference.
    """
    avg_ranks = []
    for r in ranks:
        avg = np.mean(np.concatenate(r))
        avg_ranks.append(avg)

    return avg_ranks


def load_ranks_by_label(save_dir, num_users=300, cross_domain=False, label=1):
    """
    Helper method to load ranks by train/test dataset.
    If label = 1, train set ranks are loaded. If label = 0, test set ranks are loaded.
    Ranks are generated by running sated_nmt_ranks.py.
    """
    ranks = []
    labels = []
    y = []
    for i in range(num_users):
        save_path = save_dir + 'rank_u{}_y{}{}.npz'.format(i, label, '_cd' if cross_domain else '')
        if os.path.exists(save_path):
            f = np.load(save_path, allow_pickle=True)
            train_rs, train_ls = f['arr_0'], f['arr_1']
            ranks.append(train_rs)
            labels.append(train_ls)
            y.append(label)

    return ranks, labels, y


def load_all_ranks(save_dir, num_users=5000, cross_domain=False):
    """
    Loads all ranks generated by the target model.
    Ranks are generated by running sated_nmt_ranks.py.
    """
    ranks = []
    labels = []
    y = []

    # Load train ranks
    train_label = 1
    train_ranks, train_labels, train_y = load_ranks_by_label(save_dir, num_users, cross_domain, train_label)
    ranks = ranks + train_ranks
    labels = labels + train_labels
    y = y + train_y

    # Load test ranks
    test_label = 0
    test_ranks, test_labels, test_y = load_ranks_by_label(save_dir, num_users, cross_domain, test_label)
    ranks = ranks + test_ranks
    labels = labels + test_labels
    y = y + test_y

    return ranks, labels, np.asarray(y)


def run_average_rank_thresholding(num_users=300, dim=100, prop=1.0, user_data_ratio=0.,
                                  top_words=5000, cross_domain=False, rerun=False):
    """
    Runs average rank thresholding attack on the target model.
    """
    result_path = OUTPUT_PATH

    if dim > top_words:
        dim = top_words

    attack1_results_save_path = result_path + 'mi_data_dim{}_prop{}_{}{}_attack1.npz'.format(
        dim, prop, num_users, '_cd' if cross_domain else '')

    if not rerun and os.path.exists(attack1_results_save_path):
        f = np.load(attack1_results_save_path)
        X, y = [f['arr_{}'.format(i)] for i in range(4)]
    else:
        save_dir = result_path + 'target_{}{}/'.format(num_users, '_dr' if 0. < user_data_ratio < 1. else '')
        # Load ranks
        train_ranks, _, train_y = load_ranks_by_label(save_dir, num_users, label=1)
        test_ranks, _, test_y = load_ranks_by_label(save_dir, num_users, label=0)

        # Convert to average rank features
        train_feat = avg_rank_feats(train_ranks)
        test_feat = avg_rank_feats(test_ranks)

        # Create dataset
        X, y = np.concatenate([train_feat, test_feat]), np.concatenate([train_y, test_y])
        np.savez(attack1_results_save_path, X, y)

    # print(X.shape, y.shape)

    # Find threshold using ROC
    clf = LogisticRegression()
    clf.fit(X.reshape(-1, 1), y)
    probs = clf.predict_proba(X.reshape(-1, 1))

    fpr, tpr, thresholds = roc_curve(y, probs[:, 1])

    plt.figure(1)
    plt.plot(fpr, tpr, label='Attack 1')
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title('ROC curve')
    plt.savefig('sateduser_attack1_roc_curve.png')


if __name__ == '__main__':
    num_users = 300
    save_probs = False
    rerun = True

    print("Getting target ranks...")
    get_target_ranks(num_users=num_users, save_probs=save_probs)

    print("Running average rank thresholding attack...")
    run_average_rank_thresholding(num_users=num_users, rerun=True)
