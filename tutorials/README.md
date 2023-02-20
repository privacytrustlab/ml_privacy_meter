<!-- # Documentation  -->

# Privacy Meter Tutorials

To help you get started with the Privacy Meter, we have included the following tutorials in the form of Jupyter Notebooks. For a comprehensive understanding, we suggest working through the first five tutorials. Additionally, we have included tutorials on how to utilize the Privacy Meter with OpenVINO models and Hugging Face's Causal Language Models.

You can access these tutorials on Colab for a seamless experience.

| Tutorial                                                                                     | <img src="https://www.tensorflow.org/images/colab_logo_32px.png" />                                                                    |
| -------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------- |
| [Running the Population Metric](population_metric.ipynb)                                     | [Run](https://colab.research.google.com/github/privacytrustlab/ml_privacy_meter/blob/master/tutorials/population_metric.ipynb)         |
| [Running the White-box Attack based on Population Metric](white_box_attack.ipynb)             | [Run](https://colab.research.google.com/github/privacytrustlab/ml_privacy_meter/blob/master/tutorials/white_box_attack.ipynb)           |
| [Running the Reference Metric](reference_metric.ipynb)                                       | [Run](https://colab.research.google.com/github/privacytrustlab/ml_privacy_meter/blob/master/tutorials/reference_metric.ipynb)          |
| [Evaluating the Average Privacy Loss of a Training Algorithm](avg_loss_training_algo.ipynb)  | [Run](https://colab.research.google.com/github/privacytrustlab/ml_privacy_meter/blob/master/tutorials/avg_loss_training_algo.ipynb)    |
| [Running the Shadow Metric ](shadow_metric.ipynb)                                            | [Run](https://colab.research.google.com/github/privacytrustlab/ml_privacy_meter/blob/master/tutorials/shadow_metric.ipynb)             |
| [Extending the Model class for OpenVINO Models](openvino_models.ipynb)                       | [Run](https://colab.research.google.com/github/privacytrustlab/ml_privacy_meter/blob/master/tutorials/openvino_models.ipynb)           |
| [Extending the tool for HuggingFace Causal Language Models](hf_causal_language_models.ipynb) | [Run](https://colab.research.google.com/github/privacytrustlab/ml_privacy_meter/blob/master/tutorials/hf_causal_language_models.ipynb) |

If you are running the tutorials on Colab, please add the following code to install the privacy meter.
```
# Clone the repo.
!git clone https://github.com/changhongyan123/ml_privacy_meter.git
%pip install -r privacy_meter/requirements.txt
exit()
```
Then, add the following code before you import the privacy meter
```
# Add the repo root to the Python path.
import sys, os
sys.path.append(os.getcwd())
```

