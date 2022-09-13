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

# Modification by yuan74: to save model loss values to use for implementing attack R and D in privacy meter tool

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

    losses = np.array(losses)[:,:,:,0]
    keep = np.array(keep)[:, :losses.shape[1]]

    target_member = []
    target_non_member = []
    reference_member = []
    reference_non_member = []

    for j in range(losses.shape[1]):
        if keep[-1,j]>0:
            target_member.append(losses[-1, j, :])
            reference_member.append(losses[:-1,j, :])
        else:
            target_non_member.append(losses[-1, j, :])
            reference_non_member.append(losses[:-1, j, :])

    # np.save(os.path.join(logdir, 'member_non_member_losses'), losses)

    # np.save(os.path.join(logdir, 'member_non_member_keep'), keep)

    if not os.path.exists(logdir + "_tmp"):
        os.mkdir(logdir + "_tmp")

    savez_compressed(os.path.join(logdir, 'ReferenceMetric_reference_member.npz'), np.transpose(reference_member))

    savez_compressed(os.path.join(logdir, 'ReferenceMetric_reference_non_member.npz'), np.transpose(reference_non_member))
    
    savez_compressed(os.path.join(logdir, 'ReferenceMetric_target_member.npz'), np.transpose(target_member))

    savez_compressed(os.path.join(logdir, 'ReferenceMetric_target_non_member.npz'), np.transpose(target_non_member))

    return losses, keep



logdir = sys.argv[1]
load_data(logdir)
