# Benchmark Experiments

This folder contains a script for running benchmark experiments with different methods under the same settings.

## Setting

The target model is trained using 30% of the entire CIFAR10 dataset (referred to as "members") and tested with another 30% of the dataset (referred to as "non-members"). The remaining data is considered population data and will be used for population attacks.

## Attack Methods

Both Lira attack and reference attacks require training reference models. Here's how each attack method works:

- Lira Attack: The models are trained on a random subset of the target dataset, including both the training and testing sets. Each target data point (from member and non-member set) is included in half of the reference models (referred to as "IN models") and excluded from the other half (referred to as "OUT models").

- Reference Attacks: Only the OUT models are used for the attack.

- Population Attacks: Only use the population data and the target model for the attack (i.e., do not train any reference models)

## Metrics

In terms of metrics, we consider the rescaled logits for all attacks and fit Gaussian distributions to the signals computed on the IN models and OUT models. Here's how the ROC is generated for each attack method:

- Population Attack: The alpha-percentile of the signal distribution computed on the population data from the target model is used as the threshold for the attack. The attacker sweeps all possible thresholds to get the whole ROC.

- Reference Attacks: The alpha-percentile of the OUT model distribution is used as the threshold for the attack. The attacker sweeps all possible thresholds to get the whole ROC.

- Lira Attack: The likelihood ratio between the IN model distribution and the OUT model distribution is computed and the attacker sweeps all possible thresholds to get the whole ROC.

## Running the Script

To obtain all the attack results under this setting, please run the following command:
    
```
python main.py --cf benchmark.yaml
```

The results will be saved in the `benchmark` folder within the current directory.


## Results
In the following, we present the results for different variants of attacks:
1. reference_in_fixed: This attack uses both the IN and OUT models to compute the likelihood ratio for each target data point. The standard deviation of the OUT signals is the same as the IN signals and is constant for all data points.
2. reference_in: This attack also utilizes both the IN and OUT models to compute the likelihood for each target data point. However, the standard deviation is computed individually for each data point.
3. reference_out_fixed: In this attack, only the OUT models are used to compute the signal and compare it with the alpha-percentile of the OUT signal distribution. The standard deviation of the OUT signals is constant for all data points.
4. reference_out: Similar to the previous attack, the OUT models are used to compute the signal and compare it with the alpha-percentile of the OUT signal distribution. However, the standard deviation is computed individually for each data point.
5. reference_out_offline_fixed: This attack employs the OUT models to compute the probability density function (PDF) of the target signal with respect to the OUT signal distribution. The standard deviation of the OUT signals remains constant for all data points.
6. reference_out_offline: Similar to the previous attack, the OUT models are used to compute the PDF of the target signal with respect to the OUT signal distribution. However, the standard deviation is computed individually for each data point.
7. population: This attack utilizes the target model's performance on population data to construct the OUT signal distribution. Each target point's signal is then compared with the alpha-percentile of the OUT signal distribution.

<p align="center" width="100%">
    <img width="80%" src="benchmark/benchmark/Combined_ROC_log_scaled.png">
</p>