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

## What is Privacy Meter?

Privacy Meter is an open-source library to audit data privacy in statistical and machine learning algorithms. The tool can help in the data protection impact assessment process by providing a quantitative analysis of the fundamental privacy risks of a (machine learning) model. It uses state-of-the-art inference techniques to audit a wide range of machine learning algorithms for classification, regression, computer vision, and natural language processing. Privacy Meter generates extensive reports about the aggregate and individual privacy risks for data records in the training set, at multiple levels of access to the model.

## Why Privacy Meter?

Machine learning is playing a central role in automated decision-making in a wide range of organizations and service providers. The data, which is used to train the models, typically contain sensitive information about individuals. Although the data in most cases cannot be released, due to privacy concerns, the models are usually made public or deployed as a service for inference on new test data. For a safe and secure use of machine learning models, it is important to have a quantitative assessment of the privacy risks of these models, and to make sure that they do not reveal sensitive information about their training data. This is of a great importance as there has been a surge in use of machine learning in sensitive domains such as medical and finance applications.

Data Protection regulations, such as GDPR, and AI governance frameworks require personal data to be protected when used in AI systems, and that the users have control over their data and awareness about how it is being used. For example, [Article 35 of GDPR](https://gdpr-info.eu/art-35-gdpr/) requires organizations to systematically analyze, identify and minimize the data protection risks of a project, especially when the project involves innovative technologies such as Artificial Intelligence, Machine Learning and Deep Learning. Thus, proper mechanisms need to be in place to quantitatively evaluate and verify the privacy of individuals in every step of the data processing pipeline in AI systems.

ML Privacy Meter is a Python library (`privacy_meter`) that enables quantifying the privacy risks of machine learning models. The tool provides privacy risk scores which help in identifying data records among the training data that are at high risk of being leaked through the model parameters or predictions.

## Overview

The core of the Privacy Meter consists of three parts: `Information Source`, `Metric` and `Metric Results`.

![alt text](https://github.com/privacytrustlab/ml_privacy_meter/blob/master/source/_static/privacy_meter_architecture.png?raw=true)

<!-- Kindly refer to the tutorial on the population attack ([here](advanced/population_metric.ipynb)) to gain familiarity with the utilization of each component. -->

## Installation

Privacy Meter supports Python `>=3.6` and works with `tensorflow>=2.4.0` and `torch>=1.10.0`.

You can install `privacy-meter` using `pip` for the latest stable version of the tool:

<!-- ```bash
pip install git+https://github.com/privacytrustlab/ml_privacy_meter.git
``` -->

```bash
pip install privacy-meter
```

Alternatively, one can install it via conda:

```bash
conda install privacy-meter
```
## User manual

We offer two types of tutorials: basic usage (in the [experiments](https://github.com/privacytrustlab/ml_privacy_meter/tree/master/basic/) folder folder) and advanced usage (in the [advanced](https://github.com/privacytrustlab/ml_privacy_meter/tree/master/advanced/) folder folder). The goal of the basic usage is to provide users with a seamless experience in working with a predefined selection of games, algorithms, and signals. These components represent state-of-the-art auditing attacks and can be configured easily, without requiring users to write code (See instructions [here](https://github.com/privacytrustlab/ml_privacy_meter/tree/master/basic/README.md)). On the other hand, the advanced usage is tailored for professional users who seek to conduct sophisticated auditing. It allows them to utilize both pre-existing and customized algorithms, signals, and models, empowering them to perform advanced auditing tasks at a higher level of complexity and customization. Specifically, we provide the following tutorials for adavanced usage:

1. [Understanding low-level APIs: Acquire a fundamental understanding of the Privacy Meter by executing a population attack on the CIFAR10 dataset.](https://github.com/privacytrustlab/ml_privacy_meter/tree/master/advanced/population_metric.ipynb)
2. [Understanding low-level APIs: Enhance your knowledge by conducting a reference attack on the CIFAR10 dataset.](https://github.com/privacytrustlab/ml_privacy_meter/tree/master/advanced/reference_metric.ipynb)
3. [Implementing a simple white-box attack using the Privacy Meter.](https://github.com/privacytrustlab/ml_privacy_meter/tree/master/advanced/white_box_attack.ipynb)
4. [Expanding the Privacy Meter to encompass OpenVINO models.](https://github.com/privacytrustlab/ml_privacy_meter/tree/master/advanced/openvino_models.ipynb)
5. [Integrating the Privacy Meter with HuggingFace models.](https://github.com/privacytrustlab/ml_privacy_meter/tree/master/advanced/hf_causal_language_models.ipynb)

## Video (Talks)

- [Auditing Data Privacy in Machine Learning: A Comprehensive Introduction](https://www.sigsac.org/ccs/CCS2022/workshops/workshops.html#:~:text=Auditing%20Data%20Privacy%20in%20Machine%20Learning%3A%20A%20Comprehensive%20Introduction) at CCS 2022, by Reza Shokri.
- [Auditing Data Privacy in Machine Learning](https://youtu.be/sqCd5A1UTrQ) at USENIX Enigma 2022, by Reza Shokri.
- [Machine Learning Privacy Meter Tool](https://youtu.be/DWqnKNZTz10) at HotPETS 2020, by Sasi Kumar Murakonda.

## Contributing

If you wish to add new ways of analyzing the privacy risk or add new model support, please follow our [guidelines](CONTRIBUTING.md).

## Contact / Feedback

Please feel free to join our [Slack Channel](https://join.slack.com/t/privacy-meter/shared_invite/zt-1oge6ovjq-SS4UZnBVB115Tx8Nn3TVhA) to provide your feedback and your thoughts on the project!

## Citing Privacy Meter

To cite this repository, please include the following references (or you can download the [bib file](CITATION.bib)).

1. Chang, Hongyan, Aadyaa Maddi, Victor Masiak, Mihir Khandekar, Jiayuan Ye, and Reza Shokri (2023). Privacy Meter. https://github.com/privacytrustlab/ml_privacy_meter.

2. Jiayuan Ye, Aadyaa Maddi, Sasi Kumar Murakonda, Reza Shokri. [Enhanced Membership Inference Attacks against Machine Learning Models](https://arxiv.org/pdf/2111.09679.pdf) in Proceedings of the 2022 ACM SIGSAC Conference on Computer and Communications Security, 2022.

3. Sasi Kumar Murakonda, Reza Shokri. [MLPrivacy Meter: Aiding Regulatory Compliance by Quantifying the Privacy Risks of Machine Learning](https://arxiv.org/pdf/2007.09339.pdf) in Workshop on Hot Topics in Privacy Enhancing Technologies (HotPETs), 2020.

4. Milad Nasr, Reza Shokri, and Amir Houmansadr. [Comprehensive Privacy Analysis of Deep Learning: Stand-alone and Federated Learning under Passive and Active White-box Inference Attacks](https://www.comp.nus.edu.sg/~reza/files/Shokri-SP2019.pdf) in IEEE Symposium on Security and Privacy, 2019.

5. Reza Shokri, Marco Stronati, Congzheng Song, and Vitaly Shmatikov. [Membership Inference Attacks against Machine Learning Models](https://www.comp.nus.edu.sg/~reza/files/Shokri-SP2017.pdf) in IEEE Symposium on Security and Privacy, 2017.

## Authors

The tool is designed and developed at NUS Data Privacy and Trustworthy Machine Learning Lab. We also welcome contributions from the community.

<a href="https://github.com/privacytrustlab/ml_privacy_meter/graphs/contributors">
  <img src="https://stg.contrib.rocks/image?repo=privacytrustlab/ml_privacy_meter" />
</a>
