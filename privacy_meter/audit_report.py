import os
import subprocess
from abc import ABC, abstractmethod
from typing import List
import json

import jinja2
import numpy as np
import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt
from datetime import date

from privacy_meter.metric_result import MetricResult


class AuditReport(ABC):
    """
    An abstract class to display and/or save some elements of a metric result object.
    """

    @staticmethod
    @abstractmethod
    def generate_report(metric_result: MetricResult):
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
    def generate_report(metric_result: MetricResult,
                        show: bool = False,
                        save: bool = True,
                        filename: str = 'roc_curve.jpg'
                        ):
        """
        Core function of the AuditReport class, that actually generates the report.

        Args:
            metric_result: MetricResult object, containing data for the report.
            show: Boolean specifying if the plot should be displayed on screen.
            save: Boolean specifying if the plot should be saved as a file.
            filename: File name to be used if the plot is saved as a file.
        """

        # Read and store the explanation dict
        with open('report_files/explanations.json', 'r') as f:
            explanations = json.load(f)

        # Generate plot
        range01 = np.linspace(0, 1)
        fpr, tpr, thresholds = metric_result.roc
        plt.fill_between(fpr, tpr, alpha=0.15)
        plt.plot(fpr, tpr, label=explanations["metric"][metric_result.metric_id]["name"])
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
            f'AUC = {metric_result.roc_auc:.03f}',
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
        member_signals = metric_result.signal_values[np.array(metric_result.true_labels)]
        non_member_signals = metric_result.signal_values[1 - np.array(metric_result.true_labels)]
        plt.hist(member_signals, label='Members', alpha=0.5)
        plt.hist(non_member_signals, label='Non-members', alpha=0.5)
        plt.grid()
        plt.legend()
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
    def generate_report(metric_results: List[MetricResult],
                        figures_dict: dict,
                        system_name: str,
                        call_pdflatex: bool = True,
                        show: bool = False,
                        save: bool = True,
                        filename_no_extension: str = 'report',
                        ):
        """
        Core function of the AuditReport class, that actually generates the report.

        Args:
            metric_results: A list of MetricResult objects, containing data for the report.
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
        with open('report_files/explanations.json', 'r') as f:
            explanations = json.load(f)

        # Some objects...
        metric_result_names = [m.metric_id for m in metric_results]
        files_dict = {}

        # Generate all plots and save their names
        for metric in figures_dict:
            files_dict[metric] = {}
            result = metric_results[metric_result_names.index(metric)]
            for figure in figures_dict[metric]:
                filename = f'{metric}_{figure}.jpg'
                files_dict[metric][figure] = filename
                if figure == 'roc_curve':
                    ROCCurveReport.generate_report(result, filename=filename)
                elif figure == 'confusion_matrix':
                    ConfusionMatrixReport.generate_report(result, filename=filename)
                elif figure == 'signal_histogram':
                    SignalHistogramReport.generate_report(result, filename=filename)
                else:
                    raise ValueError(f'{figure} is not a valid figure id.')

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
        template = latex_jinja_env.get_template('./report_files/report_template.tex')

        # Render the template (i.e. generate the corresponding string)
        latex_content = template.render(
            bib_file=os.path.abspath('report_files/citations.bib'),
            image_folder=os.path.abspath('.'),
            name='Purchase100 classifier',
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
