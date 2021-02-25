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
Note : The user can also train their own model to attack simliar to the example in `tutorials/alexnet.py`
Then, run the script to download the required data files.
```
cd datasets
chmod +x download_cifar100.sh
./download_cifar100.sh
```
This downloads the dataset file and training set file and converts them into the format required by the tool.
We then run the attack code `python tutorials/attack_alexnet.py`. 
The `attackobj` initializes the meminf class and the attack configuration. Following are some examples of configurations that can be changed in the function.
Note : The code explicitly sets the means and standard deviations for normalizing the images, according to the CIFAR-100 distribution.
1. Whitebox attack - Exploit the final layer gradients, final layer outputs, loss values and label values (DEFAULT)
```
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
```
attackobj = ml_privacy_meter.attack.meminf.initialize(
    target_train_model=cmodelA,
    target_attack_model=cmodelA,
    train_datahandler=datahandlerA,
    attack_datahandler=datahandlerA,
    layers_to_exploit=[22, 26],
    device=None)
```
2. Blackbox attack - Exploit final layer output and label values
```
attackobj = ml_privacy_meter.attack.meminf.initialize(
    target_train_model=cmodelA,
    target_attack_model=cmodelA,
    train_datahandler=datahandlerA,
    attack_datahandler=datahandlerA,
    layers_to_exploit=[26],
	exploit_loss=False,
    device=None)
```
