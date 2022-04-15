import os
import subprocess
from abc import ABC, abstractmethod
from typing import List, Union
import json

import jinja2
import numpy as np
import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt
from datetime import date

from privacy_meter.metric_result import MetricResult


REPORT_FILES_DIR = 'report_files'


class AuditReport(ABC):
    """
    An abstract class to display and/or save some elements of a metric result object.
    """

    @staticmethod
    @abstractmethod
    def generate_report(metric_result: Union[MetricResult, List[MetricResult], dict]):
        """
        Core function of the AuditReport class, that actually generates the report.

        Args:
            metric_result: MetricResult object, containing data for the report.
        """
        pass


class ROCCurveReport(AuditReport):
    """
    Inherits of the AuditReport class, an interface class to display and/or save some elements of a metric result
    object. This particular class is used to generate a ROC (Receiver Operating Characteristic) curve.
    """

    @staticmethod
    def generate_report(metric_results: List[MetricResult],
                        show: bool = False,
                        save: bool = True,
                        filename: str = 'roc_curve.jpg'
                        ):
        """
        Core function of the AuditReport class, that actually generates the report.

        Args:
            metric_results: A list of MetricResult objects, containing data for the report.
            show: Boolean specifying if the plot should be displayed on screen.
            save: Boolean specifying if the plot should be saved as a file.
            filename: File name to be used if the plot is saved as a file.
        """

        # Read and store the explanation dict
        with open(f'{REPORT_FILES_DIR}/explanations.json', 'r') as f:
            explanations = json.load(f)

        # Computes fpr, tpr and auc in different ways, depending on the available information
        if len(metric_results) == 1:
            fpr, tpr, _ = metric_results[0].roc
            roc_auc = metric_results[0].roc_auc
        else:
            fpr = [metric_result.fp / (metric_result.fp + metric_result.tn) for metric_result in metric_results]
            tpr = [metric_result.tp / (metric_result.tp + metric_result.fn) for metric_result in metric_results]
            roc_auc = np.trapz(x=fpr, y=tpr)

        # Generate plot
        range01 = np.linspace(0, 1)
        plt.fill_between(fpr, tpr, alpha=0.15)
        plt.plot(fpr, tpr, label=explanations["metric"][metric_results[0].metric_id]["name"])
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


class ConfusionMatrixReport(AuditReport):
    """
    Inherits of the AuditReport class, an interface class to display and/or save some elements of a metric result
    object. This particular class is used to generate a confusion matrix.
    """

    @staticmethod
    def generate_report(metric_result: MetricResult,
                        show: bool = False,
                        save: bool = True,
                        filename: str = 'confusion_matrix.jpg'
                        ):
        """
        Core function of the AuditReport class, that actually generates the report.

        Args:
            metric_result: MetricResult object, containing data for the report.
            show: Boolean specifying if the plot should be displayed on screen.
            save: Boolean specifying if the plot should be saved as a file.
            filename: File name to be used if the plot is saved as a file.
        """
        cm = np.array([[metric_result.tn, metric_result.fp], [metric_result.fn, metric_result.tp]])
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


class SignalHistogramReport(AuditReport):
    """
    Inherits of the AuditReport class, an interface class to display and/or save some elements of a metric result
    object. This particular class is used to generate a histogram of the signal values.
    """

    @staticmethod
    def generate_report(metric_result: MetricResult,
                        show: bool = False,
                        save: bool = True,
                        filename: str = 'signal_histogram.jpg'
                        ):
        """
        Core function of the AuditReport class, that actually generates the report.

        Args:
            metric_result: MetricResult object, containing data for the report.
            show: Boolean specifying if the plot should be displayed on screen.
            save: Boolean specifying if the plot should be saved as a file.
            filename: File name to be used if the plot is saved as a file.
        """

        histogram = sn.histplot(
            data=pd.DataFrame({
                'Signal': np.array(metric_result.signal_values).ravel(),
                'Membership': ['Member' if y == 1 else 'Non-member' for y in np.array(metric_result.true_labels).ravel()]
            }),
            x='Signal',
            hue='Membership',
            element='step',
            kde=True
        )

        if metric_result.threshold is not None:
            threshold = metric_result.threshold
            histogram.axvline(
                x=threshold,
                linestyle='--',
                color="C{}".format(2)
            )
            histogram.text(
                x=threshold - (np.max(metric_result.signal_values) - np.min(metric_result.signal_values))/30,
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


class PDFReport(AuditReport):
    """
    Inherits of the AuditReport class, an interface class to display and/or save some elements of a metric result
    object. This particular class is used to generate a user-friendly report, with multiple plots and some explanations.
    """

    @staticmethod
    def generate_report(metric_results: dict,
                        figures_dict: dict,
                        system_name: str,
                        call_pdflatex: bool = True,
                        show: bool = False,
                        save: bool = True,
                        filename_no_extension: str = 'report'
                        ):
        """
        Core function of the AuditReport class, that actually generates the report.

        Args:
            metric_results: A dict of lists of MetricResult objects, containing data for the report.
            figures_dict: A dictionary containing the figures to include, for each metric result.
                E.g. {"shadow_metric": ["roc_curve", "confusion_matrix", "signal_histogram"]}
            system_name: Name of the system being audited. E.g. "Purchase100 classifier"
            call_pdflatex: Boolean to specify if the pdflatex compiler should be called (to get a PDF file from the
                TEX file)
            show: Boolean specifying if the plot should be displayed on screen.
            save: Boolean specifying if the plot should be saved as a file.
            filename_no_extension: File name to be used if the plot is saved as a file, without the file extension.
        """

        # Read and store the explanation dict
        with open(f'{REPORT_FILES_DIR}/explanations.json', 'r') as f:
            explanations = json.load(f)

        # Generate all plots, and save their filenames
        files_dict = {}
        for metric in metric_results:
            files_dict[metric] = {}
            result = metric_results[metric]
            best_index = np.argmax([r.accuracy for r in result])
            if 'roc_curve' in figures_dict[metric]:
                figure = 'roc_curve'
                filename = f'{metric}_{figure}.jpg'
                files_dict[metric][figure] = filename
                ROCCurveReport.generate_report(result, filename=filename)
            if 'confusion_matrix' in figures_dict[metric]:
                figure = 'confusion_matrix'
                filename = f'{metric}_{figure}.jpg'
                files_dict[metric][figure] = filename
                ConfusionMatrixReport.generate_report(result[best_index], filename=filename)
            if 'signal_histogram' in figures_dict[metric]:
                figure = 'signal_histogram'
                filename = f'{metric}_{figure}.jpg'
                files_dict[metric][figure] = filename
                SignalHistogramReport.generate_report(result[best_index], filename=filename)

        # Configure jinja for LaTex and load template
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
        template = latex_jinja_env.get_template(f'{REPORT_FILES_DIR}/report_template.tex')

        # Render the template (i.e. generate the corresponding string)
        latex_content = template.render(
            bib_file=os.path.abspath(f'{REPORT_FILES_DIR}/citations.bib'),
            image_folder=os.path.abspath('.'),
            name=system_name,
            tool_version='1.0',
            report_date=date.today().strftime("%b-%d-%Y"),
            explanations=explanations,
            figures_dict=figures_dict,
            files_dict=files_dict
        )

        # Write the result (the string) to a .tex file
        with open(f'{filename_no_extension}.tex', 'w') as f:
            f.write(latex_content)

        print(f'LaTex file created:\t{os.path.abspath(f"{filename_no_extension}.tex")}')

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

            print(f'PDF file created:\t{os.path.abspath(f"{filename_no_extension}.pdf")}')
