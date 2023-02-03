# Privacy Meter

[![PyPI - Python Version](https://img.shields.io/badge/python-3.6%20%7C%203.7%20%7C%203.8-blue)](https://pypi.org/project/privacy-meter/)
[![Downloads](https://static.pepy.tech/badge/privacy-meter)](https://pepy.tech/project/privacy-meter)
[![PyPI version](https://img.shields.io/pypi/v/openfl)](https://pypi.org/project/privacy-meter/)
[<img src="https://img.shields.io/badge/slack-@privacy_meter-blue.svg?logo=slack">](https://join.slack.com/t/privacy-meter/shared_invite/zt-1oge6ovjq-SS4UZnBVB115Tx8Nn3TVhA)
![License](https://img.shields.io/github/license/privacytrustlab/ml_privacy_meter)
[![Citation](https://img.shields.io/badge/cite-citation-brightgreen)](https://arxiv.org/abs/2007.09339)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/privacytrustlab/ml_privacy_meter/blob/master/docs/population_metric.ipynb)
![Contributors](https://img.shields.io/github/contributors/privacytrustlab/ml_privacy_meter?color=dark-green)
![Forks](https://img.shields.io/github/forks/privacytrustlab/ml_privacy_meter?style=social)
![Stargazers](https://img.shields.io/github/stars/privacytrustlab/ml_privacy_meter?style=social)
![License](https://img.shields.io/github/license/privacytrustlab/ml_privacy_meter)

## What is privacy meter?

Privacy Meter is an open-source library to audit data privacy in statistical and machine learning algorithms. The tool can help in data protection impact assessment process by providing a quantitative analysis of fundamental privacy risks of a (machine learning) model. It uses state-of-the-art inference techniques to audit a wide range of machine learning algorithms for classification, regression, computer vision, and natural language processing. Privacy Meter generates extensive reports about the aggregate and individual privacy risks for data records in the training set, at multiple levels of access to the model.

## Overview

## Why Privacy Meter?

Machine learning is playing a central role in automated decision making in a wide range of organization and service providers. The data, which is used to train the models, typically contain sensitive information about individuals. Although the data in most cases cannot be released, due to privacy concerns, the models are usually made public or deployed as a service for inference on new test data. For a safe and secure use of machine learning models, it is important to have a quantitative assessment of the privacy risks of these models, and to make sure that they do not reveal sensitive information about their training data. This is of a great importance as there has been a surge in use of machine learning in sensitive domains such as medical and finance applications.

Data Protection regulations, such as GDPR, and AI governance frameworks require personal data to be protected when used in AI systems, and that the users have control over their data and awareness about how it is being used. For example, [Article 35 of GDPR](https://gdpr-info.eu/art-35-gdpr/) requires organizations to systematically analyze, identify and minimize the data protection risks of a project, especially when the project involves innovative technologies such as Artificial Intelligence, Machine Learning and Deep Learning. Thus, proper mechanisms need to be in place to quantitatively evaluate and verify the privacy of individuals in every step of the data processing pipeline in AI systems.

ML Privacy Meter is a Python library (`privacy_meter`) that enables quantifying the privacy risks of machine learning models. The tool provides privacy risk scores which help in identifying data records among the training data that are under high risk of being leaked through the model parameters or predictions.

## Installation

Privacy Meter supports Python `>=3.6` and works with `tensorflow>=2.4.0` and `torch>=1.10.0`.

You can install `privacy-meter` using `pip` for the latest stable version of the tool:

```bash
pip install privacy-meter
```

Alternatively, one can install it via conda:

```bash
conda install privacy-meter
```

## Quickstart

We provide examples about how to run privacy meter on standard datasets and models in experiments folder. Run the following code for your first membership inference attack on CIFAR10.

```
cd experments
python main.py --cf config_models.yaml
```

## User manual

### Basic Usage

### Auditing privacy risk for a trained model

### Advanced Usage

## White box attack

## Roadmap and Architecture

Roadmap is usually included in the issues, where the tage of the issues is for example `feature requested` or `v1.0`.
[Roadmap](https://github.com/Trusted-AI/adversarial-robustness-toolbox/wiki/ART-Architecture-and-Roadmap#art-19-december-2021)

Architecture shows the module design of the privacy meter.
[Architecture](https://github.com/Trusted-AI/adversarial-robustness-toolbox/wiki/ART-Architecture-and-Roadmap#art-19-december-2021)

## Video (Talks)

- [Auditing Data Privacy in Machine Learning](https://youtu.be/sqCd5A1UTrQ) at USENIX Enigma 2022, by Reza Shokri.
- [Machine Learning Privacy Meter Tool](https://youtu.be/DWqnKNZTz10) at HotPETS 2020, by Sasi Kumar Murakonda.

## Contributing

If you wish to add new ways of analyzing the privacy risk or add new models support, please follow our guildelins.

## Contact / Feedback

Please feel free to join our Slack Channel for providing your feedbacks and your thoughts on the project!

## Authors

The tool is designed and developed at NUS Data Privacy and Trustworthy Machine Learning Lab.

<!-- Current contributers are: Aadyaa Maddi, Jiayuan Ye, Victor Masiak, Fatemehsadat Mireshghallah, Hongyan Chang, Martin Strobel, and Reza Shokri. Earlier contributors were Sasi Kumar Murakonda, Milad Nasr, Shadab Shaikh, and Mihir Harshavardhan Khandekar.

<p float="left">
<img src="https://www.comp.nus.edu.sg/~reza/img/aadyaa.jpg" height="140"/>
<img src="https://www.comp.nus.edu.sg/~reza/img/jiayuan.jpg" height="140"/>
<img src="https://www.comp.nus.edu.sg/~reza/img/victor.jpg" height="140"/>
<img src="https://cseweb.ucsd.edu//~fmireshg/pic.jpg" height="140"/>
<img src="https://www.comp.nus.edu.sg/~reza/img/martin.jpg" height="140"/>
<img src="https://www.comp.nus.edu.sg/~reza/img/hongyan.jpg" height="140"/>
<img src="https://www.comp.nus.edu.sg/~reza/img/reza.jpg" height="140"/>
</p> -->

## Citing Privacy Meter

To cite this repository:

```
@article{murakondaml,
  title={ML Privacy Meter: Aiding Regulatory Compliance by Quantifying the Privacy Risks of Machine Learning},
  author={Murakonda, Sasi Kumar and Shokri, Reza}
}
```
