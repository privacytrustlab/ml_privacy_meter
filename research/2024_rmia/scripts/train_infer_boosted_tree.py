import functools
import os
import shutil
from typing import Callable
import json

# from sklearn.ensemble import GradientBoostingClassifier # too slow
from xgboost import XGBClassifier
from objax.util import EasyDict

from dataset import DataSet
import pickle
from absl import app, flags
import numpy as np

from sklearn.model_selection import GridSearchCV

FLAGS = flags.FLAGS


def get_data(seed):
    """
    This is the function to generate subsets of the data for training models.

    First, we get the training dataset either from the numpy cache
    or otherwise we load it from tensorflow datasets.

    Then, we compute the subset. This works in one of two ways.

    1. If we have a seed, then we just randomly choose examples based on
       a prng with that seed, keeping FLAGS.pkeep fraction of the data.

    2. Otherwise, if we have an experiment ID, then we do something fancier.
       If we run each experiment independently then even after a lot of trials
       there will still probably be some examples that were always included
       or always excluded. So instead, with experiment IDs, we guarantee that
       after FLAGS.num_experiments are done, each example is seen exactly half
       of the time in train, and half of the time not in train.

    """
    DATA_DIR = os.path.join(os.environ['HOME'], 'TFDS')

    if os.path.exists(os.path.join(FLAGS.logdir, "x_train.npy")) or FLAGS.dataset == "purchase100": # Modified
        try:
            inputs = np.load(os.path.join(FLAGS.logdir, "x_train.npy"))
            labels = np.load(os.path.join(FLAGS.logdir, "y_train.npy"))
        except:
            print("Please check if x_train.npy and y_train.npy are in the folder. Download the corresponding .npy at https://drive.google.com/drive/folders/1cIJlbLlgqDSJKd8YhTucHwaPiLsh6ZyW?usp=sharing")
    else:
        raise Exception("Dataset not supported.")
            
    nclass = np.max(labels)+1

    np.random.seed(seed)
    if FLAGS.num_experiments is not None:
        np.random.seed(0)
        keep = np.random.uniform(0,1,size=(FLAGS.num_experiments, FLAGS.dataset_size))
        order = keep.argsort(0)
        keep = order < int(FLAGS.pkeep * FLAGS.num_experiments)
        keep = np.array(keep[FLAGS.expid], dtype=bool)
    else:
        keep = np.random.uniform(0, 1, size=FLAGS.dataset_size) <= FLAGS.pkeep

    if FLAGS.only_subset is not None:
        keep[FLAGS.only_subset:] = 0

    xs = inputs[keep]
    ys = labels[keep]
    
    ############################ Added
    if FLAGS.dataset == "purchase100":
        try:
            x_test = np.load(os.path.join(FLAGS.logdir, "x_test.npy"))
            y_test = np.load(os.path.join(FLAGS.logdir, "y_test.npy"))
        except:
            print("Please check if x_test.npy and y_test.npy are in the folder.")
        # train = DataSet.from_arrays(xs, ys)
        # test = DataSet.from_arrays(x_test, y_test)
        # train = train.cache().shuffle(8192).repeat().parse().batch(FLAGS.batch) # no augmentation of tabular data.
        # train = train.one_hot(nclass).prefetch(16)
        # test = test.cache().parse().batch(FLAGS.batch).prefetch(16)
    else:
        raise Exception("Dataset not supported.")
    ############################ Added

    return x_test, y_test, xs, ys, keep, nclass, inputs, labels

def main(argv):
    del argv

    seed = FLAGS.seed
    if seed is None:
        import time
        seed = np.random.randint(0, 1000000000)
        seed ^= int(time.time())
        
    args = EasyDict(lr=FLAGS.lr,
                    seed=seed)

        
    if FLAGS.tunename:
        logdir = '_'.join(sorted('%s=%s' % k for k in args.items()))
    elif FLAGS.expid is not None:
        logdir = "experiment-%d_%d"%(FLAGS.expid,FLAGS.num_experiments)
    else:
        logdir = "experiment-"+str(seed)
    logdir = os.path.join(FLAGS.logdir, logdir)

    if os.path.exists(os.path.join(logdir, "ckpt", "%010d.npz"%FLAGS.epochs)):
        print(f"run {FLAGS.expid} already completed.")
        return
    else:
        if os.path.exists(logdir):
            print(f"deleting run {FLAGS.expid} that did not complete.")
            shutil.rmtree(logdir)

    print(f"starting run {FLAGS.expid}.")
    if not os.path.exists(logdir):
        os.makedirs(logdir)

    x_test, y_test, xs, ys, keep, nclass, inputs, labels = get_data(seed)

    np.save(os.path.join(logdir,'keep.npy'), keep)

    clf = XGBClassifier(
                        learning_rate=FLAGS.lr,
                        n_estimators=FLAGS.n_estimators, 
                        subsample=FLAGS.subsample, 
                        max_depth=FLAGS.max_depth,
                        device="cuda:0"
    )

    clf.fit(xs, ys)

    print("Gradient Boosted Classifier Score on Train :", clf.score(xs, ys))
    print("Gradient Boosted Classifier Score on Test :", clf.score(x_test, y_test))

    epoch = 50
    nb_augmentations = 2

    if not os.path.exists(os.path.join(logdir, "logits")):
        os.mkdir(os.path.join(logdir, "logits"))

    # Infers "logits" right away, will be reconverted to correct logit, loss and softmax from that
    stats = clf.predict_proba(inputs) # probabilities
    stats = np.log(stats+1e-45)-np.log(1-stats+1e-45)
    stats = np.array(stats)[:,None,:].repeat(2, axis=1)
    np.save(os.path.join(logdir, "logits", "%010d_%04d"%(epoch, nb_augmentations)), stats[:,None,:,:])

if __name__ == '__main__':
    flags.DEFINE_string('dataset', 'cifar10', 'Dataset.')
    flags.DEFINE_string('logdir', 'experiments', 'Directory where to save checkpoints and tensorboard data.')
    flags.DEFINE_integer('seed', None, 'Training seed.')
    flags.DEFINE_float('pkeep', .5, 'Probability to keep examples.')
    flags.DEFINE_integer('expid', None, 'Experiment ID')
    flags.DEFINE_integer('num_experiments', None, 'Number of experiments')
    flags.DEFINE_integer('dataset_size', 50000, 'number of examples to keep.')
    flags.DEFINE_bool('tunename', False, 'Use tune name?')
    flags.DEFINE_integer('epochs', 100, 'Do not change.')
    flags.DEFINE_integer('only_subset', None, 'Only train on a subset of images.')

    ### XGBoost parameters
    flags.DEFINE_integer('n_estimators', 250, 'The number of boosting stages to perform.')
    flags.DEFINE_float('lr', 0.1, 'Learning rate shrinks the contribution of each tree by lr.')
    flags.DEFINE_float('subsample', 0.2, 'The fraction of samples to be used for fitting the individual base learners.')
    flags.DEFINE_integer('max_depth', 4, 'Maximum depth of the individual regression estimators.')
    flags.DEFINE_string('loss', 'log_loss', 'Loss Function.')

    app.run(main)