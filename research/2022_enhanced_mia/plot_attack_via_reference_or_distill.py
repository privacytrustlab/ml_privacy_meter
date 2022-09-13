# Copyright 2021 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# modification by yuan74: added suport for plotting performance of Attacks in the enhancedmia paper 
# by perform sweeping on alpha associated with target's loss value rather than on the original LiRA score

import sys
import os
import scipy.stats

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import auc, roc_curve
import functools

from numpy import savez_compressed

# Look at me being proactive!
import matplotlib

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

def sweep(score, x):
    """
    Compute a ROC curve and then return the FPR, TPR, AUC, and ACC.
    """
    fpr, tpr, _ = roc_curve(x, - score)
    acc = np.max(1-(fpr+(1-tpr))/2)
    return fpr, tpr, auc(fpr, tpr), acc


def load_data(p):
    """
    Load our saved losses and then put them into a big matrix.
    """
    global losses, keep
    losses = []
    keep = []

    for f in sorted(os.listdir(os.path.join(p))):
        if not f.startswith("experiment"): continue
        print(f)
        if not os.path.exists(os.path.join(p, f, "losses")): continue
        last_epoch = sorted(os.listdir(os.path.join(p, f, "losses")))
        if len(last_epoch) == 0: continue
        filepath = os.path.join(p, f, "losses", last_epoch[-1])
        print(f"loading losses from: {filepath}")
        losses_f = np.load(os.path.join(p, f, "losses", last_epoch[-1])) 
        print(f"Does this have nan values: {np.count_nonzero(np.isnan(losses_f))}")
        losses.append(losses_f)
        keep_f = np.load(os.path.join(p, f, "keep.npy"))
        print(f"Number of values to keep: {np.sum(keep_f)}")
        keep.append(keep_f)

    losses = np.array(losses)[:,:,0,0]
    keep = np.array(keep)[:, :losses.shape[1]]

    if not os.path.exists(logdir + "_tmp"):
        os.mkdir(logdir + "_tmp")

    return losses, keep


def generate_reference_or_distill_logit(losses, check_keep, check_losses, dat_reference_or_distill_size=100000,
                  fix_variance=False):
    """
    Fit a two predictive models using keep and scores in order to predict
    if the examples in check_scores were training data or not, using the
    ground truth answer from check_keep.
    """
    print(losses.shape)

    dat_reference_or_distill = np.log(
        np.exp(-losses) / (1 - np.exp(-losses))
    )
    # dat_out = []
    mean_reference_or_distill = np.mean(dat_reference_or_distill, 0)
    # mean_out = np.median(dat_out, 1)

    std_reference_or_distill = np.std(dat_reference_or_distill, 0)

    check_losses=np.transpose(check_losses[0])

    prediction = 1 - scipy.stats.norm.cdf(np.log(np.exp(-check_losses)/(1-np.exp(-check_losses))), mean_reference_or_distill, std_reference_or_distill + 1e-30)
    answers = np.transpose(check_keep[0])

    return prediction, answers

def generate_reference_or_distill_linear(losses, check_keep, check_losses, dat_reference_or_distill_size=100000,
                  fix_variance=False):
    """
    Fit a two predictive models using keep and scores in order to predict
    if the examples in check_scores were training data or not, using the
    ground truth answer from check_keep.
    """
    print(losses.shape)
    
    dummy_min = np.zeros((1, len(losses[0])))

    dummy_max = dummy_min + 1000

    dat_reference_or_distill = np.sort(np.concatenate((losses, dummy_max, dummy_min), axis = 0), axis = 0)

    prediction = np.array([])

    discrete_alpha = np.linspace(0, 1, len(dat_reference_or_distill))
    for i in range(len(dat_reference_or_distill[0])):
        losses_i =  dat_reference_or_distill[:, i]

        # Create the interpolator
        pr = np.interp(check_losses[0][i], losses_i, discrete_alpha)
        
        prediction = np.append(prediction, pr)
    
    answers = np.transpose(check_keep[0])

    return prediction, answers

def generate_reference_or_distill_mean_linear_logit(losses, check_keep, check_losses, dat_reference_or_distill_size=100000,
                  fix_variance=False):
    """
    Fit a two predictive models using keep and scores in order to predict
    if the examples in check_scores were training data or not, using the
    ground truth answer from check_keep.
    """
    print(losses.shape)
    
    prediction1, _ = generate_reference_or_distill_linear(losses, check_keep, check_losses, dat_reference_or_distill_size=100000,
                  fix_variance=False)

    prediction2, answers = generate_reference_or_distill_logit(losses, check_keep, check_losses, dat_reference_or_distill_size=100000,
                  fix_variance=False)

    prediction = np.mean( np.array([ prediction1, prediction2 ]), axis=0 )

    return prediction, answers

def generate_reference_or_distill_min_linear_logit(losses, check_keep, check_losses, dat_reference_or_distill_size=100000,
                  fix_variance=False):
    """
    Fit a two predictive models using keep and scores in order to predict
    if the examples in check_scores were training data or not, using the
    ground truth answer from check_keep.
    """
    print(losses.shape)
    
    prediction1, _ = generate_reference_or_distill_linear(losses, check_keep, check_losses, dat_reference_or_distill_size=100000,
                  fix_variance=False)

    prediction2, answers = generate_reference_or_distill_logit(losses, check_keep, check_losses, dat_reference_or_distill_size=100000,
                  fix_variance=False)
    
    # here we perform max operation because we are operating on alpha instead of threshold
    prediction = np.max( np.array([ prediction1, prediction2 ]), axis=0 )

    return prediction, answers

def do_plot(fn, keep, scores, ntest, legend='', metric='auc', sweep_fn=sweep, **plot_kwargs):
    """
    Generate the ROC curves by using ntest models as test models and the rest to train.
    """

    prediction, answers = fn(scores[:-ntest],
                             keep[-ntest:],
                             scores[-ntest:])

    fpr, tpr, auc, acc = sweep_fn(np.array(prediction), np.array(answers, dtype=bool))

    low = tpr[np.where(fpr < .001)[0][-1]]

    print('Attack %s   AUC %.4f, Accuracy %.4f, TPR@0.1%%FPR of %.4f' % (legend, auc, acc, low))

    metric_text = ''
    if metric == 'auc':
        metric_text = 'auc=%.3f' % auc
    elif metric == 'acc':
        metric_text = 'acc=%.3f' % acc

    plt.plot(fpr, tpr, label=legend + metric_text, **plot_kwargs)
    df = pd.DataFrame(data={
        'fpr': fpr,
        'tpr': tpr
    })
    filepath = f"./{logdir}_tmp/{legend}_fpr_tpr_data.csv"
    df.to_csv(filepath)
    return (acc, auc)


def fig_fpr_tpr():

    if not os.path.exists(logdir + "_tmp"): 
        os.mkdir(logdir + "_tmp")

    plt.figure(figsize=(4, 3))

    do_plot(generate_reference_or_distill_logit,
            keep, losses, 1,
            "logit_rescale",
            metric='auc'
            )

    do_plot(generate_reference_or_distill_linear,
            keep, losses, 1,
            "linear_itp",
            metric='auc'
            )
        
    do_plot(generate_reference_or_distill_min_linear_logit,
            keep, losses, 1,
            "min_linear_logit",
            metric='auc'
            )
    
    do_plot(generate_reference_or_distill_mean_linear_logit,
            keep, losses, 1,
            "mean_linear_logit",
            metric='auc'
            )

    plt.semilogx()
    plt.semilogy()
    plt.xlim(1e-5, 1)
    plt.ylim(1e-5, 1)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.plot([0, 1], [0, 1], ls='--', color='gray')
    plt.subplots_adjust(bottom=.18, left=.18, top=.96, right=.96)
    plt.legend(fontsize=8)
    plt.savefig(logdir+"_tmp/fprtpr.png")
    plt.show()

logdir = sys.argv[1]
load_data(logdir)
fig_fpr_tpr()
