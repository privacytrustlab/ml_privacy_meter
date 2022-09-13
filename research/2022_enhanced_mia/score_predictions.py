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

# Modification by yuan74
# Generate the soft label prediction to use for distillation

import sys
import numpy as np
import os
import multiprocessing as mp


def load_one(base):
    """
    This loads a  logits and converts it to a scored prediction.
    """
    root = os.path.join(logdir, base, 'logits')
    if not os.path.exists(root): return None

    if not os.path.exists(os.path.join(logdir, base, 'predictions')):
        os.mkdir(os.path.join(logdir, base, 'predictions'))

    for f in os.listdir(root):
        try:
            opredictions = np.load(os.path.join(root, f))
        except:
            print("Fail")
            continue

        ## Be exceptionally careful.
        ## Numerically stable everything, as described in the paper.
        predictions = opredictions - np.max(opredictions, axis=3, keepdims=True)
        predictions = np.array(np.exp(predictions), dtype=np.float64)
        predictions = predictions / np.sum(predictions, axis=3, keepdims=True)

        COUNT = predictions.shape[0]
        print(COUNT)

        print('mean acc', np.mean(predictions[:, 0, 0, :].argmax(1) == labels[:COUNT]))

        np.save(os.path.join(logdir, base, 'predictions', f), predictions)


def load_stats():
    with mp.Pool(8) as p:
        p.map(load_one, [x for x in os.listdir(logdir) if 'exp' in x])


logdir = sys.argv[1]
labels = np.load(os.path.join(logdir,"y_train.npy"))
load_stats()
