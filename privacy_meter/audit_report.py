import json
import os
import subprocess
from abc import ABC, abstractmethod
from datetime import date
from typing import Dict, List, Tuple, Union

import jinja2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sn
from PIL import Image
from scipy import interpolate

from privacy_meter.constants import InferenceGame
from privacy_meter.information_source import InformationSource
from privacy_meter.information_source_signal import DatasetSample
from privacy_meter.metric_result import CombinedMetricResult, MetricResult

########################################################################################################################
# GLOBAL SETTINGS
########################################################################################################################

# Temporary parameter pointing to the report_file directory (for compatibility)
REPORT_FILES_DIR = "report_files"

# Configure jinja for LaTex
latex_jinja_env = jinja2.Environment(
    block_start_string='\BLOCK{',
    block_end_string='}',
    variable_start_string='\VAR{',
    variable_end_string='}',
    comment_start_string='\#{',
    comment_end_string='}',
    line_statement_prefix='%%',
    line_comment_prefix='%#',
    trim_blocks=True,
    autoescape=False,
    loader=jinja2.FileSystemLoader(os.path.abspath('.'))
)

########################################################################################################################
# AUDIT_REPORT CLASS
########################################################################################################################


class AuditReport(ABC):
    """
    An abstract class to display and/or save some elements of a metric result object.
    """

    @staticmethod
    @abstractmethod
    def generate_report(
            metric_result: Union[MetricResult, List[MetricResult], dict, CombinedMetricResult],
            inference_game_type: InferenceGame
    ):
        """
        Core function of the AuditReport class that actually generates the report.

        Args:
            metric_result: MetricResult object, containing data for the report.
            inference_game_type: Value from the InferenceGame ENUM type, indicating which inference game was used.
        """
        pass


########################################################################################################################
# ROC_CURVE_REPORT CLASS
########################################################################################################################


class ROCCurveReport(AuditReport):
    """
    Inherits from the AuditReport class, an interface class to display and/or save some elements of a metric result
    object. This particular class is used to generate a ROC (Receiver Operating Characteristic) curve.
    """

    @staticmethod
    def __avg_roc(
            fpr_2d_list: List[List[float]],
            tpr_2d_list: List[List[float]],
            n: int = 200
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Private helper function, to average a ROC curve from non-aligned list.

        Args:
            fpr_2d_list: A 2D list of fpr values.
            tpr_2d_list: A 2D list of fpr values.
            n: Number of points in the resulting lists.

        Returns:
            A tuple of aligned 1D numpy arrays, fpr and tpr.
        """
        functions = [interpolate.interp1d(fpr, tpr) for (
            fpr, tpr) in zip(fpr_2d_list, tpr_2d_list)]
        fpr = np.linspace(0, 1, n)
        tpr = np.mean([f(fpr) for f in functions], axis=0)
        return fpr, tpr

    @staticmethod
    def generate_report(
            metric_result: Union[MetricResult, List[MetricResult], List[List[MetricResult]], CombinedMetricResult],
            inference_game_type: InferenceGame,
            show: bool = False,
            save: bool = True,
            filename: str = 'roc_curve.jpg'
    ):
        """
        Core function of the AuditReport class that actually generates the report.

        Args:
            metric_result: A list of MetricResult objects, containing data for the report.
            inference_game_type: Value from the InferenceGame ENUM type, indicating which inference game was used.
            show: Boolean specifying if the plot should be displayed on screen.
            save: Boolean specifying if the plot should be saved as a file.
            filename: File name to be used if the plot is saved as a file.
        """

        # Read and store the explanation dict
        with open(f'{REPORT_FILES_DIR}/explanations.json', 'r') as f:
            explanations = json.load(f)

        # Check if it is the combined report:
        if not isinstance(metric_result, list):
            metric_result = [metric_result]
        if not isinstance(metric_result[0], CombinedMetricResult) and not isinstance(metric_result[0][0], CombinedMetricResult):
            # Casts type to a 2D list
            if not isinstance(metric_result[0], list):
                metric_result = [metric_result]
            # Computes fpr, tpr and auc in different ways, depending on the available information and inference game
            if inference_game_type == InferenceGame.PRIVACY_LOSS_MODEL:
                if metric_result[0][0].predictions_proba is None:
                    fpr = [mr.fp / (mr.fp + mr.tn) for mr in metric_result[0]]
                    tpr = [mr.tp / (mr.tp + mr.fn) for mr in metric_result[0]]
                    roc_auc = np.trapz(x=fpr, y=tpr)
                else:
                    fpr, tpr, _ = metric_result[0][0].roc
                    roc_auc = metric_result[0][0].roc_auc
            elif inference_game_type == InferenceGame.AVG_PRIVACY_LOSS_TRAINING_ALGO:
                if metric_result[0][0].predictions_proba is None:
                    fpr = [[metric_result[i][j].fp / (metric_result[i][j].fp + metric_result[i][j].tn)
                            for j in range(len(metric_result[0]))] for i in range(len(metric_result))]
                    tpr = [[metric_result[i][j].tp / (metric_result[i][j].tp + metric_result[i][j].fn)
                            for j in range(len(metric_result[0]))] for i in range(len(metric_result))]
                    fpr = np.mean(fpr, axis=0)
                    tpr = np.mean(tpr, axis=0)
                    roc_auc = np.trapz(x=fpr, y=tpr)
                else:
                    fpr, tpr = ROCCurveReport.__avg_roc(
                        fpr_2d_list=[metric_result[i][0].roc[0]
                                     for i in range(len(metric_result))],
                        tpr_2d_list=[metric_result[i][0].roc[1]
                                     for i in range(len(metric_result))]
                    )
                    roc_auc = np.trapz(x=fpr, y=tpr)
            else:
                raise NotImplementedError
        else:
            # Generate report for the combined report
            # Computes fpr, tpr and auc in different ways, depending on the available information and inference game
            if inference_game_type == InferenceGame.PRIVACY_LOSS_MODEL:
                if metric_result[0].predictions_proba is None:
                    mr = metric_result[0]
                    fpr = mr.fp / (mr.fp + mr.tn)
                    tpr = mr.tp / (mr.tp + mr.fn)
                    roc_auc = np.trapz(x=fpr, y=tpr)
            elif inference_game_type == InferenceGame.AVG_PRIVACY_LOSS_TRAINING_ALGO:
                if metric_result[0][0].predictions_proba is None:
                    fpr = [[metric_result[i][j].fp / (metric_result[i][j].fp + metric_result[i][j].tn)
                            for j in range(len(metric_result[0]))] for i in range(len(metric_result))]
                    tpr = [[metric_result[i][j].tp / (metric_result[i][j].tp + metric_result[i][j].fn)
                            for j in range(len(metric_result[0]))] for i in range(len(metric_result))]
                    fpr = np.mean(fpr, axis=0).ravel()
                    tpr = np.mean(tpr, axis=0).ravel()
                    roc_auc = np.trapz(x=fpr, y=tpr)
            else:
                raise NotImplementedError

        # Gets metric ID
        if isinstance(metric_result, list):
            if isinstance(metric_result[0], list):
                metric_id = metric_result[0][0].metric_id
            else:
                metric_id = metric_result[0].metric_id
        else:
            metric_id = metric_result.metric_id

        # Generate plot
        range01 = np.linspace(0, 1)
        plt.fill_between(fpr, tpr, alpha=0.15)
        plt.plot(fpr, tpr, label=explanations["metric"][metric_id]["name"])
        plt.plot(range01, range01, '--', label='Random guess')
        plt.xlim([0, 1])
        plt.ylim([0, 1])
        plt.grid()
        plt.legend()
        plt.xlabel('False positive rate (FPR)')
        plt.ylabel('True positive rate (TPR)')
        plt.title('ROC curve')
        plt.text(
            0.7, 0.3,
            f'AUC = {roc_auc:.03f}',
            horizontalalignment='center',
            verticalalignment='center',
            bbox=dict(facecolor='white', alpha=0.5)
        )
        if save:
            plt.savefig(fname=filename, dpi=1000)
        if show:
            plt.show()
        plt.clf()


########################################################################################################################
# CONFUSION_MATRIX_REPORT CLASS
########################################################################################################################


class ConfusionMatrixReport(AuditReport):
    """
    Inherits from the AuditReport class, an interface class to display and/or save some elements of a metric result
    object. This particular class is used to generate a confusion matrix.
    """

    @staticmethod
    def generate_report(
            metric_result: Union[MetricResult, List[MetricResult]],
            inference_game_type: InferenceGame,
            show: bool = False,
            save: bool = True,
            filename: str = 'confusion_matrix.jpg'
    ):
        """
        Core function of the AuditReport class that actually generates the report.

        Args:
            metric_result: MetricResult object, containing data for the report.
            inference_game_type: Value from the InferenceGame ENUM type, indicating which inference game was used.
            show: Boolean specifying if the plot should be displayed on screen.
            save: Boolean specifying if the plot should be saved as a file.
            filename: File name to be used if the plot is saved as a file.
        """

        if inference_game_type == InferenceGame.PRIVACY_LOSS_MODEL:
            assert isinstance(metric_result, MetricResult)
            cm = np.array([[metric_result.tn, metric_result.fp],
                          [metric_result.fn, metric_result.tp]])
        elif inference_game_type == InferenceGame.AVG_PRIVACY_LOSS_TRAINING_ALGO:
            assert isinstance(metric_result, list)
            cm = np.mean([[[mr.tn, mr.fp], [mr.fn, mr.tp]]
                         for mr in metric_result], axis=0)
        else:
            raise NotImplementedError

        cm = 100 * cm / np.sum(cm)
        index = ["Non-member", "Member"]
        df_cm = pd.DataFrame(cm, index, index)
        sn.heatmap(df_cm, annot=True, cmap=plt.cm.Blues)
        plt.xlabel('Predicted label')
        plt.ylabel('True label')
        plt.title('Confusion matrix (in %)')
        if save:
            plt.savefig(fname=filename, dpi=1000)
        if show:
            plt.show()
        plt.clf()


########################################################################################################################
# SIGNAL_HISTOGRAM_REPORT CLASS
########################################################################################################################


class SignalHistogramReport(AuditReport):
    """
    Inherits from the AuditReport class, an interface class to display and/or save some elements of a metric result
    object. This particular class is used to generate a histogram of the signal values.
    """

    @staticmethod
    def generate_report(
            metric_result: Union[MetricResult, List[MetricResult]],
            inference_game_type: InferenceGame,
            show: bool = False,
            save: bool = True,
            filename: str = 'signal_histogram.jpg'
    ):
        """
        Core function of the AuditReport class that actually generates the report.

        Args:
            metric_result: MetricResult object, containing data for the report.
            inference_game_type: Value from the InferenceGame ENUM type, indicating which inference game was used.
            show: Boolean specifying if the plot should be displayed on screen.
            save: Boolean specifying if the plot should be saved as a file.
            filename: File name to be used if the plot is saved as a file.
        """

        if inference_game_type == InferenceGame.PRIVACY_LOSS_MODEL:
            values = np.array(metric_result.signal_values).ravel()
            labels = np.array(metric_result.true_labels).ravel()
            threshold = metric_result.threshold
        elif inference_game_type == InferenceGame.AVG_PRIVACY_LOSS_TRAINING_ALGO:
            if not isinstance(metric_result[0], list):
                values = np.concatenate(
                    [mr.signal_values for mr in metric_result]).ravel()
                labels = np.concatenate(
                    [mr.true_labels for mr in metric_result]).ravel()
                threshold_list = [mr.threshold for mr in metric_result]
                threshold = None if None in threshold_list else np.mean(
                    threshold_list)
            else:
                values = np.array([[metric_result[i][j].signal_values for j in range(
                    len(metric_result[0]))] for i in range(len(metric_result))]).ravel()
                labels = np.array([[metric_result[i][j].true_labels for j in range(
                    len(metric_result[0]))] for i in range(len(metric_result))]).ravel()
                threshold_list = None
                threshold = None

        else:
            raise NotImplementedError

        histogram = sn.histplot(
            data=pd.DataFrame({
                'Signal': values,
                'Membership': ['Member' if y == 1 else 'Non-member' for y in labels]
            }),
            x='Signal',
            hue='Membership',
            element='step',
            kde=True
        )

        if threshold is not None and type(threshold) == float:
            histogram.axvline(
                x=threshold,
                linestyle='--',
                color="C{}".format(2)
            )
            histogram.text(
                x=threshold - (np.max(values) - np.min(values))/30,
                y=.8,
                s='Threshold',
                rotation=90,
                color="C{}".format(2),
                transform=histogram.get_xaxis_transform()
            )

        plt.grid()
        plt.xlabel('Signal value')
        plt.ylabel('Number of samples')
        plt.title('Signal histogram')
        if save:
            plt.savefig(fname=filename, dpi=1000)
        if show:
            plt.show()
        plt.clf()


########################################################################################################################
# VULNERABLE_POINTS_REPORT CLASS
########################################################################################################################


class VulnerablePointsReport(AuditReport):
    """
    Inherits from the AuditReport class, an interface class to display and/or save some elements of a metric result
    object. This particular class is used to identify the most vulnerable points.
    """

    @staticmethod
    def generate_report(
            metric_results: List[MetricResult],
            inference_game_type: InferenceGame,
            target_info_source: InformationSource,
            target_model_to_train_split_mapping: List[Tuple[int, str, str, str]],
            number_of_points: int = 10,
            save_tex: bool = False,
            filename: str = 'vulnerable_points.tex',
            return_raw_values: bool = True,
            point_type: str = 'any'
    ):
        """Core function of the AuditReport class that actually generates the report.

        Args:
            metric_results: A dict of lists of MetricResult objects, containing data for the report.
            target_info_source: The InformationSource associated with the audited model training.
            target_model_to_train_split_mapping: The mapping associated with target_info_source.
            number_of_points: Number of vulnerable to be selected.
            save_tex: Boolean specifying if a partial .tex file should be generated.
            filename: Filename of the partial .tex file.
            return_raw_values: Boolean specifying if the points indices and scores should be returned.
            point_type: Can be "any" or "image". If "image", then the images are displayed as such in the report.

        Returns:
            Indices of the vulnerable points and their scores.

        """

        if inference_game_type != InferenceGame.PRIVACY_LOSS_MODEL:
            raise NotImplementedError(
                "For now, the only inference_game_type supported is InferenceGame.PRIVACY_LOSS_MODEL"
            )

        # Objects to be returned if return_raw_values is True
        indices, scores = [], []

        # If only one metric was used (i.e. we have access to the prediction probabilities)
        if len(metric_results) == 1:
            mr = metric_results[0]
            # Sort the training points that were identified as such by their prediction probabilities
            adjusted_values = np.where(
                (np.array(mr.predicted_labels) == np.array(mr.true_labels))
                &
                (np.array(mr.true_labels) == 1),
                - mr.predictions_proba,
                10
            )
            indices = np.argsort(adjusted_values)[:number_of_points]
            # Get the associated scores
            scores = mr.predictions_proba[indices]

        # If multiple metrics were used (i.e. we don't have access to the prediction probabilities)
        else:
            # Use the various metric, from the one with lowest fpr to the one with highest fpr
            fp_indices = np.argsort([mr.fp for mr in metric_results])
            for k in range(len(metric_results)):
                mr = metric_results[fp_indices[k]]
                # Get the training points that were identified as such
                new_indices = np.argwhere(
                    (np.array(mr.predicted_labels) == np.array(mr.true_labels))
                    &
                    (np.array(mr.true_labels) == 1)
                )
                indices.extend(list(new_indices.ravel()))
                # Get the associated scores
                fpr = mr.fp / (mr.fp + mr.tn)
                scores.extend([1-fpr] * new_indices.shape[0])
            # Only keep number_of_points points
            indices, scores = indices[:number_of_points], scores[:number_of_points]

        # Map indices stored in the metric_result object to indices in the training set
        indices_to_train_indices = []
        counter = 0
        for k, v in enumerate(metric_results[0].true_labels):
            indices_to_train_indices.append(counter)
            counter += v
        indices = np.array(indices_to_train_indices)[np.array(indices)]

        # If points are images and we are creating a LaTex file, then we read the information source to create image
        # files from the vulnerable
        if save_tex and point_type == "image":
            for k, point in enumerate(indices):
                x = target_info_source.get_signal(
                    signal=DatasetSample(),
                    model_to_split_mapping=target_model_to_train_split_mapping,
                    extra={"model_num": 0, "point_num": point}
                )
                Image.fromarray((x*255).astype('uint8')
                                ).save(f'point{k:03d}.jpg')

        # If we are creating a LaTex
        if save_tex:

            # Load template
            template = latex_jinja_env.get_template(
                f'{REPORT_FILES_DIR}/vulnerable_points_template.tex')

            # Render the template (i.e. generate the corresponding string)
            latex_content = template.render(
                points=[{
                    "index": index,
                    "score": f'{score:.3f}',
                    "type": point_type,
                    "path": f"point{k:03d}.jpg" if point_type == "image" else None
                } for (k, (index, score)) in enumerate(zip(indices, scores))]
            )

            # Write the result (the string) to a .tex file
            with open(filename, 'w') as f:
                f.write(latex_content)

        # If we required the values to be returned
        if return_raw_values:
            return indices, scores


########################################################################################################################
# PDF_REPORT CLASS
########################################################################################################################


class PDFReport(AuditReport):
    """
    Inherits from the AuditReport class, an interface class to display and/or save some elements of a metric result
    object. This particular class is used to generate a user-friendly report, with multiple plots and some explanations.
    """

    @staticmethod
    def generate_report(
            metric_results: Dict[str, Union[MetricResult, List[MetricResult], List[List[MetricResult]]]],
            inference_game_type: InferenceGame,
            figures_dict: dict,
            system_name: str,
            call_pdflatex: bool = True,
            show: bool = False,
            save: bool = True,
            filename_no_extension: str = 'report',
            target_info_source: InformationSource = None,
            target_model_to_train_split_mapping: List[Tuple[int,
                                                            str, str, str]] = None,
            point_type: str = 'any'
    ):
        """
        Core function of the AuditReport class that actually generates the report.

        Args:
            metric_results: A dict of lists of MetricResult objects, containing data for the report.
            inference_game_type: Value from the InferenceGame ENUM type, indicating which inference game was used.
            figures_dict: A dictionary containing the figures to include, for each metric result.
                E.g. {"shadow_metric": ["roc_curve", "confusion_matrix", "signal_histogram"]}.
            system_name: Name of the system being audited. E.g. "Purchase100 classifier".
            call_pdflatex: Boolean to specify if the pdflatex compiler should be called (to get a PDF file from the
                TEX file).
            show: Boolean specifying if the plot should be displayed on screen.
            save: Boolean specifying if the plot should be saved as a file.
            filename_no_extension: File name to be used if the plot is saved as a file, without the file extension.
        """

        for metric in metric_results:
            if not isinstance(metric_results[metric], list):
                metric_results[metric] = [metric_results[metric]]
            if not isinstance(metric_results[metric][0], list):
                metric_results[metric] = [[mr]
                                          for mr in metric_results[metric]]

        # Read and store the explanation dict
        with open(f'{REPORT_FILES_DIR}/explanations.json', 'r') as f:
            explanations = json.load(f)

        # Generate all plots, and save their filenames
        files_dict = {}
        for metric in metric_results:
            files_dict[metric] = {}
            result = metric_results[metric]

            # Select one instance to display when necessary (e.g. for a confusion matrix with a PopulationMetric)
            if inference_game_type == InferenceGame.PRIVACY_LOSS_MODEL:
                best_index = np.argmax([r.accuracy for r in result])
                best_result = result[best_index]
            elif inference_game_type == InferenceGame.AVG_PRIVACY_LOSS_TRAINING_ALGO:
                best_indices = np.argmax(
                    [[r2.accuracy for r2 in r1] for r1 in result], axis=1)
                best_result = [result[k][best_index]
                               for k, best_index in enumerate(best_indices)]
            else:
                raise NotImplementedError

            if 'roc_curve' in figures_dict[metric]:
                figure = 'roc_curve'
                filename = f'{metric}_{figure}.jpg'
                files_dict[metric][figure] = filename
                ROCCurveReport.generate_report(
                    metric_result=result,
                    inference_game_type=inference_game_type,
                    filename=filename
                )
            if 'confusion_matrix' in figures_dict[metric]:
                figure = 'confusion_matrix'
                filename = f'{metric}_{figure}.jpg'
                files_dict[metric][figure] = filename
                ConfusionMatrixReport.generate_report(
                    metric_result=best_result,
                    inference_game_type=inference_game_type,
                    filename=filename
                )
            if 'signal_histogram' in figures_dict[metric]:
                figure = 'signal_histogram'
                filename = f'{metric}_{figure}.jpg'
                files_dict[metric][figure] = filename
                SignalHistogramReport.generate_report(
                    metric_result=best_result,
                    inference_game_type=inference_game_type,
                    filename=filename
                )
            if 'vulnerable_points' in figures_dict[metric]:
                assert target_info_source is not None
                assert target_model_to_train_split_mapping is not None
                figure = 'vulnerable_points'
                filename = f'{metric}_{figure}.tex'
                files_dict[metric][figure] = filename
                VulnerablePointsReport.generate_report(
                    metric_results=result,
                    inference_game_type=inference_game_type,
                    save_tex=True,
                    filename=filename,
                    target_info_source=target_info_source,
                    target_model_to_train_split_mapping=target_model_to_train_split_mapping,
                    point_type=point_type
                )

        # Load template
        template = latex_jinja_env.get_template(
            f'{REPORT_FILES_DIR}/report_template.tex')

        # Render the template (i.e. generate the corresponding string)
        latex_content = template.render(
            bib_file=os.path.abspath(f'{REPORT_FILES_DIR}/citations.bib'),
            image_folder=os.path.abspath('.'),
            name=system_name,
            tool_version='1.0',
            report_date=date.today().strftime("%b-%d-%Y"),
            explanations=explanations,
            figures_dict=figures_dict,
            files_dict=files_dict,
            inference_game_type=inference_game_type.value
        )

        # Write the result (the string) to a .tex file
        with open(f'{filename_no_extension}.tex', 'w') as f:
            f.write(latex_content)

        print(
            f'LaTex file created:\t{os.path.abspath(f"{filename_no_extension}.tex")}')

        if call_pdflatex:

            # Compile the .tex file to a .pdf file. Several rounds are required to get the references (to papers, to
            # page numbers, and to figure numbers)

            process = subprocess.Popen(['pdflatex', os.path.abspath(f'{filename_no_extension}.tex')],
                                       stdout=subprocess.PIPE,
                                       stderr=subprocess.PIPE)
            stdout, stderr = process.communicate()

            process = subprocess.Popen(['biber', os.path.abspath(f'{filename_no_extension}')],
                                       stdout=subprocess.PIPE,
                                       stderr=subprocess.PIPE)
            stdout, stderr = process.communicate()

            process = subprocess.Popen(['pdflatex', os.path.abspath(f'{filename_no_extension}.tex')],
                                       stdout=subprocess.PIPE,
                                       stderr=subprocess.PIPE)
            stdout, stderr = process.communicate()

            process = subprocess.Popen(['pdflatex', os.path.abspath(f'{filename_no_extension}.tex')],
                                       stdout=subprocess.PIPE,
                                       stderr=subprocess.PIPE)
            stdout, stderr = process.communicate()

            print(
                f'PDF file created:\t{os.path.abspath(f"{filename_no_extension}.pdf")}')
