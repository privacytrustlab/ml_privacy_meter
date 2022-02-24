import os
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, auc, confusion_matrix, roc_auc_score

from ml_privacy_meter.utils.attack_utils import get_predictions, calculate_loss_threshold


class ReferenceAttack:
    def __init__(self, exp_name, x_population, y_population,
                 x_target_train, y_target_train,
                 x_target_test, y_target_test,
                 target_model_filepath, target_model_type,
                 x_ref_train_indices_list,
                 ref_model_filepath_list, ref_model_type_list,
                 loss_fn, seed, x_target_train_indices=None,
                 target_model_class=None,
                 ref_model_class_list=None):

        # reference models are trained on splits of
        # the population data
        self.x_population = x_population
        self.y_population = y_population

        # target model data and architecture
        self.x_target_train = x_target_train
        self.y_target_train = y_target_train
        self.x_target_test = x_target_test
        self.y_target_test = y_target_test
        self.target_model_filepath = target_model_filepath
        self.target_model_type = target_model_type
        self.target_model_class = target_model_class
        self.x_target_train_indices = x_target_train_indices

        # reference model train data splits and architecture
        self.x_ref_train_indices_list = x_ref_train_indices_list
        self.ref_model_filepath_list = ref_model_filepath_list
        self.ref_model_type_list = ref_model_type_list
        self.ref_model_class_list = ref_model_class_list

        # other hyperparameters
        self.loss_fn = loss_fn
        self.seed = seed

        # create results directory
        self.attack_results_dirpath = f'logs/population_attack_{exp_name}/'
        if not os.path.isdir(Path(self.attack_results_dirpath)):
            os.mkdir(Path(self.attack_results_dirpath))

    def prepare_attack(self):
        """
        Compute and save loss distribution of target model and loss values on its train and test data.
        """
        print("Computing and saving train and test loss distributions of the target model...")

        if self.x_target_train_indices:
            loss_dist_shape = (len(self.x_target_train), len(self.ref_model_filepath_list))
            train_loss_dist = np.full(loss_dist_shape, fill_value=np.inf)
            for ref_idx, ref_model_filepath in enumerate(self.ref_model_filepath_list):
                print(f"Computing train loss dist using reference model at {ref_model_filepath}...")
                ref_model_type = self.ref_model_type_list[ref_idx]
                ref_model_class = self.ref_model_class_list[ref_idx]

                ref_model_indices = self.x_ref_train_indices_list[ref_idx]
                remaining_indices = np.setdiff1d(self.x_target_train_indices, ref_model_indices)
                x_remaining, y_remaining = self.x_population[remaining_indices], self.y_population[remaining_indices]

                remaining_losses = self.loss_fn(
                    get_predictions(
                        model_filepath=ref_model_filepath,
                        model_type=ref_model_type,
                        data=x_remaining,
                        model_class=ref_model_class
                    )
                )

                for (point_idx, point_loss) in zip(remaining_indices, remaining_losses):
                    row_position = list(self.x_target_train_indices).index(point_idx)
                    train_loss_dist[row_position, ref_idx] = point_loss

            # remove inf values from train loss distribution
            train_loss_dist = np.array([x[x != np.inf] for x in train_loss_dist], dtype=object)
        else:
            print(f"No overlapping train data used in reference models...")

            train_loss_dist = np.array([0] * len(self.x_target_train))
            for ref_idx, ref_model_filepath in enumerate(self.ref_model_filepath_list):
                print(f"Computing train loss dist using reference model at {ref_model_filepath}...")
                ref_model_type = self.ref_model_type_list[ref_idx]
                ref_model_class = self.ref_model_class_list[ref_idx]
                ref_train_losses = self.loss_fn(
                    y_true=self.y_target_train,
                    y_pred=get_predictions(
                        model_filepath=ref_model_filepath,
                        model_type=ref_model_type,
                        data=self.x_target_train,
                        model_class=ref_model_class
                    )
                )
                train_loss_dist = np.column_stack((train_loss_dist,
                                                   ref_train_losses))
            train_loss_dist = train_loss_dist[:, 1:]  # remove dummy first column

        np.savez(f"{self.attack_results_dirpath}/target_model_train_loss_dist",
                 train_loss_dist=train_loss_dist)

        test_loss_dist = np.array([0] * len(self.x_target_test))
        for ref_idx, ref_model_filepath in enumerate(self.ref_model_filepath_list):
            print(f"Computing test loss dist using reference model at {ref_model_filepath}...")
            ref_model_type = self.ref_model_type_list[ref_idx]
            ref_model_class = self.ref_model_class_list[ref_idx]
            ref_test_losses = self.loss_fn(
                y_true=self.y_target_test,
                y_pred=get_predictions(
                    model_filepath=ref_model_filepath,
                    model_type=ref_model_type,
                    data=self.x_target_test,
                    model_class=ref_model_class
                )
            )
            test_loss_dist = np.column_stack((test_loss_dist,
                                              ref_test_losses))
        test_loss_dist = test_loss_dist[:, 1:]  # remove dummy first column

        np.savez(f"{self.attack_results_dirpath}/target_model_test_loss_dist",
                 test_loss_dist=test_loss_dist)

        print("Computing and saving train and test losses of the target model...")

        train_losses = self.loss_fn(
            y_true=self.y_target_train,
            y_pred=get_predictions(model_filepath=self.target_model_filepath,
                                   model_type=self.target_model_type,
                                   data=self.x_target_train,
                                   model_class=self.target_model_class)
        )
        test_losses = self.loss_fn(
            y_true=self.y_target_test,
            y_pred=get_predictions(model_filepath=self.target_model_filepath,
                                   model_type=self.target_model_type,
                                   data=self.x_target_test,
                                   model_class=self.target_model_class)
        )

        np.savez(f"{self.attack_results_dirpath}/target_model_losses",
                 train_losses=train_losses,
                 test_losses=test_losses)

    def run_attack(self, alphas):
        """
        Runs the reference attack on the target model.
        """

        # get train and test loss distributions, loss values
        train_loss_dist_filepath = f"{self.attack_results_dirpath}/target_model_train_loss_dist.npz"
        test_loss_dist_filepath = f"{self.attack_results_dirpath}/target_model_test_loss_dist.npz"
        losses_filepath = f"{self.attack_results_dirpath}/target_model_losses.npz"
        if os.path.isfile(train_loss_dist_filepath) \
                and os.path.isfile(test_loss_dist_filepath) \
                and os.path.isfile(losses_filepath):
            with np.load(train_loss_dist_filepath, allow_pickle=True) as train_loss_dist_data:
                train_loss_dist = train_loss_dist_data['train_loss_dist'][()]

            with np.load(test_loss_dist_filepath, allow_pickle=True) as test_loss_dist_data:
                test_loss_dist = test_loss_dist_data['test_loss_dist'][()]

            with np.load(losses_filepath, allow_pickle=True) as losses_data:
                train_losses = losses_data['train_losses'][()]
                test_losses = losses_data['test_losses'][()]
        else:
            self.prepare_attack()

        for alpha in alphas:
            print(f"For alpha = {alpha}...")

            # compute threshold for train data
            train_thresholds = []
            for (point_idx, point_loss_dist) in enumerate(train_loss_dist):
                train_thresholds.append(calculate_loss_threshold(alpha, loss_distribution=point_loss_dist))

            # compute threshold for test data
            test_thresholds = []
            for (point_idx, point_loss_dist) in enumerate(test_loss_dist):
                test_thresholds.append(calculate_loss_threshold(alpha, loss_distribution=point_loss_dist))

            # generate predictions: <= threshold, output '1' (member) else '0' (non-member)
            preds = []
            for (loss, threshold) in zip(train_losses, train_thresholds):
                if loss <= threshold:
                    preds.append(1)
                else:
                    preds.append(0)

            for (loss, threshold) in zip(test_losses, test_thresholds):
                if loss <= threshold:
                    preds.append(1)
                else:
                    preds.append(0)

            y_eval = [1] * len(train_losses)
            y_eval.extend([0] * len(test_losses))

            # save attack results
            acc = accuracy_score(y_eval, preds)
            roc_auc = roc_auc_score(y_eval, preds)
            tn, fp, fn, tp = confusion_matrix(y_eval, preds).ravel()
            np.savez(f"{self.attack_results_dirpath}/attack_results_{alpha}",
                     true_labels=y_eval, preds=preds, alpha=alpha,
                     train_thresholds=train_thresholds,
                     test_thresholds=test_thresholds,
                     acc=acc, roc_auc=roc_auc,
                     tn=tn, fp=fp, tp=tp, fn=fn)

            print(
                f"Reference attack performance:\n"
                f"Accuracy = {acc}\n"
                f"ROC AUC Score = {roc_auc}\n"
                f"FPR: {fp / (fp + tn)}\n"
                f"TN, FP, FN, TP = {tn, fp, fn, tp}"
            )

    def visualize_attack(self, alphas):
        alphas = sorted(alphas)

        tpr_values = []
        fpr_values = []

        for alpha in alphas:
            filepath = f'{self.attack_results_dirpath}/attack_results_{alpha}.npz'
            with np.load(filepath, allow_pickle=True) as data:
                tp = data['tp'][()]
                fp = data['fp'][()]
                tn = data['tn'][()]
                fn = data['fn'][()]
            tpr = tp / (tp + fn)
            fpr = fp / (fp + tn)
            tpr_values.append(tpr)
            fpr_values.append(fpr)

        tpr_values.insert(0, 0)
        fpr_values.insert(0, 0)
        tpr_values.append(1)
        fpr_values.append(1)

        auc_value = round(auc(x=fpr_values, y=tpr_values), 5)

        fig, ax = plt.subplots()
        ax.plot(fpr_values,
                tpr_values,
                linewidth=2.0,
                color='b',
                label=f'AUC = {auc_value}')
        ax.set_xlabel("FPR")
        ax.set_ylabel("TPR")
        ax.set_ylim([0.0, 1.1])
        ax.legend(loc='lower right')
        plt.savefig(f'{self.attack_results_dirpath}/tpr_vs_fpr', dpi=250)
        plt.close(fig)
