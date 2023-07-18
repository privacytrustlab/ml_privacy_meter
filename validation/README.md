# Reproducing Existing Results

This folder provides configurations (YAML files) for reproducing the results provided in published papers on different datasets. To reproduce the results, please run the `basic/main.py` with the corresponding configuration files.

**Example:**

```
cd ../basic
python main.py --cf ../validation/cifar10/config_population_setup1.yaml
```

# Benchmark results

In the following, we provide the pointer to the configuration files and the expected results, and the results we get based on the Privacy Meter.

## On CIFAR10 dataset

| Paper                                                                                                                       | Attack            | Target Model Structure | Results in the paper (AUC)                                                                                                                                                                                                      | Results produced by Privacy Meter (AUC)                                        | Configuration File                      |
| --------------------------------------------------------------------------------------------------------------------------- | ----------------- | ---------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------ | --------------------------------------- |
| [Enhanced Membership Inference Attacks against Machine Learning Models](https://dl.acm.org/doi/abs/10.1145/3548606.3560675) | Population attack | AlexNet                | 0.857±0.023 (Table 4 of the appendix)                                                                                                                                                                                           | 0.870                                                                          | `cifar10/config_population_setup1.yaml` |
| [Enhanced Membership Inference Attacks against Machine Learning Models](https://dl.acm.org/doi/abs/10.1145/3548606.3560675) | Reference attack  | AlexNet                | 0.874±0.018 (Table 4 of the appendix)                                                                                                                                                                                           | 0.879                                                                          | `cifar10/config_reference_setup1.yaml`  |
| [Membership Inference Attacks From First Principles](https://arxiv.org/pdf/2112.03570.pdf)                                  | Online attack     | WideResNet             | 0.6606 (reproduced without data augmentation using the [official library](https://github.com/tensorflow/privacy/tree/4dd8d0ffde4ddb1575d5c2fc02e0693e08f4f4a1/research/mi_lira_2021); the model achieves 8% generalization gap) | 0.6429 (without data augmentation; the model achieves 7.1% generalization gap) | `cifar10/config_lira_online.yaml`       |
