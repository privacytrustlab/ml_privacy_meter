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

| Paper                                                                                                                       | Attack            | Results in the paper                  | Results produced by privacy meter | Configuration File                      |
| --------------------------------------------------------------------------------------------------------------------------- | ----------------- | ------------------------------------- | --------------------------------- | --------------------------------------- |
| [Enhanced Membership Inference Attacks against Machine Learning Models](https://dl.acm.org/doi/abs/10.1145/3548606.3560675) | Population attack | 0.857±0.023 (Table 4 of the appendix) | 0.870                             | `cifar10/config_population_setup1.yaml` |
| [Enhanced Membership Inference Attacks against Machine Learning Models](https://dl.acm.org/doi/abs/10.1145/3548606.3560675) | Reference attack  | 0.874±0.018 (Table 4 of the appendix) | 0.879                             | `cifar10/config_reference_setup1.yaml`  |
