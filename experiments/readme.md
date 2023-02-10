# Privacy Meter Experiments

In this folder, we provide the end-to-end of memebership inference attack implementation based on different attack games defined in [Ye et al 2022].

The structure of the pipline is shown below.

For each run, we will save all the trained models into the model pool. If you want to audit privacy risk in different settings, the models in the model pool can be reused.

# Auditing privacy risk for given attack games

## Auditing the privacy risk for a trained model.

To start, you can run the following commands

```
python main.py --cf config_models.yaml
```

Inside the Yaml file, you can indicate how the hyperparameters for training the model, the way of splitting the data, and the membership inference attack algorithm. In the default setting, the program trains a model on cifar10 and trains 10 reference models to run the reference attack algorithm (see detailed description about the algorithm in [Ye et al. 2022]). If you are interested in attacking the model using the population attack, which does not involve training reference models, you can change the algorithm in the YAML file from `reference` to `population`.

## Auditing the privacy risk for a training algorithm.

To audit the privacy risk of a training algorithm (training settings), you can run the following commands

```
python main.py --cf config_algorithms.yaml
```

The program will repeat the first game for multiple target models, which are trained using the same training algorithm and the same training configuration, but with different training and test dataset. The result is the average privacy risk across all the target models.

## Auditing the privacy risk for a data point.
