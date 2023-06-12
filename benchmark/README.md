# Reproducing benchmark results

This folder provides configurations (YAML files) for reproducing the results provided in published papers on different datasets. To reproduce the results, please run the `experiments/main.py` with the corresponding configuration files.

**Example:**

```
cd ../experiments
python main.py --cf ../benchmark/cifar10/config_population_setup1.yaml
```

# Benchmark results

In the following, we provide the pointer to the configuration files and the expected results, and the results we get based on the privacy meter.

## On CIFAR10 dataset

| Paper                                                                                                                       | Attack            | Results in the paper (AUC)                                                                                                                                                                                                      | Results produced by privacy meter (AUC)                                      | Configuration File                      |
| --------------------------------------------------------------------------------------------------------------------------- | ----------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------- | --------------------------------------- | --- |
| [Enhanced Membership Inference Attacks against Machine Learning Models](https://dl.acm.org/doi/abs/10.1145/3548606.3560675) | Population attack | 0.857±0.023 (Table 4 of the appendix)                                                                                                                                                                                           | 0.870                                                                        | `cifar10/config_population_setup1.yaml` |
| [Enhanced Membership Inference Attacks against Machine Learning Models](https://dl.acm.org/doi/abs/10.1145/3548606.3560675) | Reference attack  | 0.874±0.018 (Table 4 of the appendix)                                                                                                                                                                                           | 0.879                                                                        | `cifar10/config_reference_setup1.yaml`  |
| [Membership Inference Attacks From First Principles](https://arxiv.org/pdf/2112.03570.pdf)                                  | Online attack     | 0.6606 (reproduced without data argumentation using the [official library](https://github.com/tensorflow/privacy/tree/4dd8d0ffde4ddb1575d5c2fc02e0693e08f4f4a1/research/mi_lira_2021); the model achieves 91.75% test accuracy) | 0.6429 (without data argumentation; the model achieves 91.52% test accuracy) | `cifar10/config_lira_online.yaml`       |     |
| [Membership Inference Attacks From First Principles](https://arxiv.org/pdf/2112.03570.pdf)                                  | Offline attack    | 0.5529                                                                                                                                                                                                                          | 0.5498                                                                       | `cifar10/config_lira_offline.yaml`      |     |
