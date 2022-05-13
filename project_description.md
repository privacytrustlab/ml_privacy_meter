# Privacy Meter

## Introduction

Privacy Meter is an open-source library to audit data privacy in statistical and machine learning algorithms. The tool can help in data protection impact assessment process by providing a quantitative analysis of fundamental privacy risks of a (machine learning) model. It uses state-of-the-art inference techniques to audit a wide range of machine learning algorithms for classification, regression, computer vision, and natural language processing. Privacy Meter generates extensive reports about the aggregate and individual privacy risks for data records in the training set, at multiple levels of access to the model.

## Installation

Privacy Meter supports Python `>=3.6` and works with `tensorflow>=2.4.0` and `torch>=1.10.0`.

You can install `privacy_meter` using `pip` for the latest stable version of the tool:

```bash
pip install privacy_meter
```

Privacy Meter has been tested with other machine learning libraries like HuggingFace and Intel OpenVINO, and can even be extended to be used with a framework of your choice.
