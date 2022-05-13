# ML Privacy Meter - Tutorials
This directory consists of some sample attack code with their dependencies.
Following are the descriptions of the files.
- attack_alexnet.py : Sample Whitebox attack on Alexnet model trained on CIFAR-100 dataset (Described in README of project)
- attack_fcn.py : Sample Whitebox attack on Fully connected model on Purchase 100 dataset.
- alexnet.py : Sample code to train target Alexnet model which can be given as input for attack
- models/alexnet_pretrained.zip	: Pretrained Keras model for Alexnet trained on CIFAR-100 dataset, converted from pretrained [Pytorch model](https://github.com/bearpaw/pytorch-classification#pretrained-models) (Described in README of project)

## Running the Alexnet CIFAR-100 Attack
To perform an attack as in Nasr et al [2], we use the Alexnet model trained on the CIFAR-100 dataset. We perform the whitebox attack on the model while exploiting the gradients, final layer outputs, loss values and label values.
First, extract the pretrained model from the `tutorials/models` directory and place it in the root directory of the project. `unzip tutorials/models/alexnet_pretrained.zip -d .`

Note: The user can also train their own model to attack simliar to the example in `tutorials/alexnet.py`

The following libraries need to be imported for this tutorial.

```python
import tensorflow as tf
import numpy as np
import ml_privacy_meter
```

To use the CIFAR100 dataset in Tensorflow, add code similar to this in your script.

```python
def preprocess_cifar100_dataset():
    input_shape = (32, 32, 3)
    num_classes = 100

    # Split the data between train and test sets
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar100.load_data()

    return x_train, y_train, x_test, y_test, input_shape, num_classes
    
x_train, y_train, x_test, y_test, input_shape, num_classes = preprocess_cifar100_dataset()
```

Next, we initialize the datahandler required by the tool.

```python
num_datapoints = 5000  # example of the train dataset size
x_target_train, y_target_train = x_train[:num_datapoints], y_train[:num_datapoints]

# population data (training data is a subset of this here)
x_population = np.concatenate((x_train, x_test))
y_population = np.concatenate((y_train, y_test))

datahandlerA = ml_privacy_meter.utils.attack_data.AttackData(x_population=x_population,
                                                             y_population=y_population,
                                                             x_target_train=x_target_train,
                                                             y_target_train=y_target_train,
                                                             batch_size=100,
                                                             attack_percentage=10, input_shape=input_shape,
                                                             normalization=True)
```

We can then specify the configuration of the attack that will be performed by the tool. The `attackobj` initializes the meminf class and the attack configuration. Following are some examples of configurations that can be customized.

Note: The code in the tutorial explicitly sets the means and standard deviations for normalizing the images, according to the CIFAR-100 distribution.

1. Whitebox attack - Exploit the final layer gradients, final layer outputs, loss values and label values (DEFAULT)
```python
attackobj = ml_privacy_meter.attack.meminf.initialize(
    target_train_model=cmodelA,
    target_attack_model=cmodelA,
    train_datahandler=datahandlerA,
    attack_datahandler=datahandlerA,
    layers_to_exploit=[26],
    gradients_to_exploit=[6],
    device=None)
```

2. Whitebox attack - Exploit final two model layer outputs, loss values and label values
```python
attackobj = ml_privacy_meter.attack.meminf.initialize(
    target_train_model=cmodelA,
    target_attack_model=cmodelA,
    train_datahandler=datahandlerA,
    attack_datahandler=datahandlerA,
    layers_to_exploit=[22, 26],
    device=None)
```

3. Blackbox attack - Exploit final layer output and label values
```python
attackobj = ml_privacy_meter.attack.meminf.initialize(
    target_train_model=cmodelA,
    target_attack_model=cmodelA,
    train_datahandler=datahandlerA,
    attack_datahandler=datahandlerA,
    layers_to_exploit=[26],
	exploit_loss=False,
    device=None)
```

The desired attack code can be run using the command `python tutorials/attack_alexnet.py`.
