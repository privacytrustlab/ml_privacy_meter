# Privacy Meter

## Introduction

Privacy Meter is an open-source library to audit data privacy in statistical and machine learning algorithms. The tool can help in data protection impact assessment process by providing a quantitative analysis of fundamental privacy risks of a (machine learning) model. It uses state-of-the-art inference techniques to audit a wide range of machine learning algorithms for classification, regression, computer vision, and natural language processing. Privacy Meter generates extensive reports about the aggregate and individual privacy risks for data records in the training set, at multiple levels of access to the model.

## Installation

Privacy Meter supports Python `>=3.6` and works with `tensorflow>=2.4.0` and `torch>=1.10.0`.

You can install `privacy_meter` using `pip` for the latest stable version of the tool:

```bash
pip install privacy_meter
```

Alternatively, one can install it via conda:

```bash
conda install -c conda-forge privacy_meter
```

Privacy Meter has been tested with other machine learning libraries like HuggingFace and Intel OpenVINO, and can even be extended to be used with a framework of your choice. 

## Usage

Examples of using the tool can be found in the `docs/` directory. The tutorials are provided as Jupyter notebooks.

## References

1. Jiayuan Ye, Aadyaa Maddi, Sasi Kumar Murakonda, Reza Shokri. [Enhanced Membership Inference Attacks against Machine Learning Models](https://arxiv.org/pdf/2111.09679.pdf) arXiv preprint arXiv:2111.09679

2. Milad Nasr, Reza Shokri, and Amir Houmansadr. [Comprehensive Privacy Analysis of Deep Learning: Stand-alone and Federated Learning under Passive and Active White-box Inference Attacks](https://www.comp.nus.edu.sg/~reza/files/Shokri-SP2019.pdf) in IEEE Symposiumon Security and Privacy, 2019.

3. Reza Shokri, Marco Stronati, Congzheng Song, and Vitaly Shmatikov. [Membership Inference Attacks against Machine Learning Models](https://www.comp.nus.edu.sg/~reza/files/Shokri-SP2017.pdf) in IEEE Symposium on Security and Privacy, 2017.

## Team

The tool is designed and developed at NUS Data Privacy and Trustworthy Machine Learning Lab. Current contributers are: Aadyaa Maddi, Jiayuan Ye, Victor Masiak, Fatemehsadat Mireshghallah, Hongyan Chang, Martin Strobel, and Reza Shokri. Earlier contributors were Sasi Kumar Murakonda, Milad Nasr, Shadab Shaikh, and Mihir Harshavardhan Khandekar.

<p float="left">
<img src="https://www.comp.nus.edu.sg/~reza/img/aadyaa.jpg" height="140"/>
<img src="https://www.comp.nus.edu.sg/~reza/img/jiayuan.jpg" height="140"/>
<img src="https://www.comp.nus.edu.sg/~reza/img/victor.jpg" height="140"/>
<img src="https://cseweb.ucsd.edu//~fmireshg/pic.jpg" height="140"/>
<img src="https://www.comp.nus.edu.sg/~reza/img/martin.jpg" height="140"/>
<img src="https://www.comp.nus.edu.sg/~reza/img/hongyan.jpg" height="140"/>  
<img src="https://www.comp.nus.edu.sg/~reza/img/reza.jpg" height="140"/>
</p>
