# Privacy Meter Experiments

This folder contains the implementation of an end-to-end membership inference attack, based on various attack games defined by [Ye et al. 2022]. It is recommended that you perform privacy auditing using our automatical pipline. By default, the audit provides a way to evaluate the privacy risks for models, algorithms, and data points using the CIFAR10 dataset. To adapt it to your specific use case, you can specify the model structure in `model.py`, the dataset in `dataset.py`, and the training algorithm in `train.py`. The training configurations can then be specified in the configuration Yaml file. The overall pipeline is illustrated below.

<p align="center" width="100%">
    <img width="80%" src="docs/experiment_pipeline.png">
</p>

In the following, we introduce how to run privacy auditing automatelliy using our default configurations.

# Privacy auditing examples

## Auditing the privacy risk for a trained model.

To start, you can run the following commands

```
python main.py --cf config_models_reference.yaml
```

The Yaml file allows you to specify the hyperparameters for training the model, the method for splitting the data, and the algorithm for the membership inference attack. By default, the program trains a model on the CIFAR10 dataset and trains ten reference models to perform the reference attack algorithm (refer to [Ye et al. 2022] for a more detailed explanation). If you are interested in conducting a population attack, which does not require the training of reference models, you can run the following commands.

```
python main.py --cf config_models_population.yaml
```

For a comprehensive explanation of each parameter, please refer to each Yaml file. Upon completion, you will find the results in the `demo` folder, with the reference attack results saved in `demo/report_reference` and the population attack results saved in `demo/report_population`. Furthermore, we also offer a timing log for each run, which can be found in the file `log_time_analysis.log`.

## Auditing the privacy risk for a training algorithm.

To audit the privacy risk of a training algorithm (training settings), you can run the following commands

```
python main.py --cf config_algorithms.yaml
```

The program will repeat the first game for multiple target models that are trained using the same training algorithm and configuration, but with different training and test datasets. The outcome will be the average privacy risk across all the target models. Upon completion, you will find the results in the `demo/report_reference_algorithm` folder.

## Auditing the privacy risk for a data point.

Moreover, we have also included an implementation for auditing the privacy risk of a single data point. To be specific, we compare the loss distribution difference of an auditing data point $z_a$ when another data $z_t$ is present in the training dataset versus not being in the target model's training dataset. If $z_t$ equals $z_a$, then the measurement essentially becomes memorization (refer to [reference](https://arxiv.org/abs/1906.05271)).

By default, we assess the memorization of the data point with index 1 (each dataset assigns an index to each data point based on its order in the dataset). To obtain the results, simply execute the following command:

```
python main.py --cf config_samples.yaml
```

After the completion, you can locate the privacy risk report in the `demo/report_sample` folder.
