# Robust Membership Inference Attacks
This directory contains code to reproduce the results in our paper:

"Low-Cost High-Power Membership Inference Attacks"
https://arxiv.org/pdf/2312.03262
by Sajjad Zarifzadeh, Philippe Liu, and Reza Shokri.

### Training/Inference Setup

For training target and reference models and for ease of comparison, we use the tool provided by Carlini et al. in [https://github.com/carlini/privacy/tree/better-mi/research/mi_lira_2021](https://github.com/carlini/privacy/tree/better-mi/research/mi_lira_2021).

For setting up the conda environment, we also use the setup provided by Ye et al. in https://github.com/yuan74/ml_privacy_meter/blob/2022_enhanced_mia/research/2022_enhanced_mia/2022_enhanced_mia.yml

To train models and infer signals, you need to be in the /scripts folder [there](scripts/).

#### Training Models

To train one model idx in a set of N models in the folder named path, you can execute: 
```
python3 -u train.py --dataset=cifar10 --epochs=100 --save_steps=100 --arch wrn28-2 --num_experiments $N --expid\$idx --logdir exp/$path &> 'logs/$path/log_$idx'  
```
For example, to train the whole set of CIFAR-10 models used in the paper (with 50%-50% data partitioning in which each sample is used in training of half of models), you can execute: 
```
bash train_cifar10.sh
```
##### Changing dataset

You can change the dataset to CIFAR-100 by redefining the dataset flag `--dataset=cifar10` to `--dataset=cifar100` in the bash file, e.g for model idx in a set of N models in the folder path you can execute:

```
python3 -u train.py --dataset=cifar100 --epochs=100 --save_steps=100 --arch wrn28-2 --num_experiments $N --expid $idx --logdir exp/$path &> 'logs/$path/log_$idx'
```

For CINIC-10 and Purchase-100, you can download reformatted datasets' files (`x_train.npy`,`y_train.npy`,`x_test.npy` and `y_test.npy`) [there](https://drive.google.com/drive/folders/1cIJlbLlgqDSJKd8YhTucHwaPiLsh6ZyW?usp=sharing). You can then put those files in `exp/cinic10` and `exp/purchase100` (or any other set of models using those datasets) respectively and then use the flag `--dataset=cinic10` or `--dataset=purchase100` in your training and inference commands/bash files (e.g. see in [`train_purchase100.sh`](scripts/train_purchase100.sh)  and [`infer_purchase100.sh`](scripts/infer_purchase100.sh)).
  
#### Inferring Models

To infer the logits from the trained models on their respective datasets, you can execute: 
```
python3 inference.py --logdir=exp/cifar10/ --aug=2  --dataset=cifar10
```
To infer for more augmentations (e.g. 18 or 50), you can change the number by the aug argument:
```
python3 inference.py --logdir=exp/cifar10/ --aug=18  --dataset=cifar10
```

You need to execute this command for the set containing your target model as well as the one containing your reference models: (e.g. see [`infer_cifar10.sh`](scripts/infer_cifar10.sh))

```
# infer for target models
python3 inference.py --logdir=exp/cifar10/ --aug=2 --dataset=cifar10  # contains the original query
python3 inference.py --logdir=exp/cifar10/ --aug=18 --dataset=cifar10

# infer for reference models
python3 inference.py --logdir=exp/cifar10_2/ --aug=2 --dataset=cifar10
python3 inference.py --logdir=exp/cifar10_2/ --aug=18 --dataset=cifar10
```

### Performing the attack
This is how to perform the membership inference attack, after having trained and inferred the models using the code above. 

#### Set up the environment

To be able to run our code, you will need to install the requirements from requirements.txt (e.g. on a separate environment):

```
conda create --name rmia python==3.10.9
conda activate rmia
pip install -r requirements.txt
```
  
#### Set up paths to target and reference models

Tou need to specify in the `.yaml` config file:

Under `run`:
- `log_dir`: it is the folder where all the results and signals are stored for a given dataset or set of target models. To compare different attacks, you can use the same `log_dir` (but not the same `report_log`)
 
Under `audit`:
- `report_log`: name of the folder under `log_dir` in which we save the attack's results

Under `data`:
- `target_dir` and `reference_dir`: the target_dir and reference_dir containing all the models trained using `train.py` in the [/script folder](scripts/). Note: `target_dir` and `reference_dir` can point to the same folder, in which case given a target model in that folder, the other models will be used as reference models.
- `dataset_size`: the total number of samples used to train all the models (e.g. 50000 across all the experiments, each model trained on half of it.)
- `epoch`: the epoch for which you want to attack. (e.g 100)

For example, you can change the .yaml file : 
```
run:
	### some parameters ###
	log_dir: demo_cifar10 # folder where all the results and signals are stored for a given dataset or set of target models
audit:
	### some other parameters ###
	report_log: results_using_4_reference_models # name of the folder under log_dir in which we save the attack's results
data:
	target_dir: scripts/exp/cifar10 
	reference_dir: scripts/exp/cifar10_4 # new set of reference models (2 IN, 2 OUT, or 2 OUT)
	dataset_size: 50000
	epoch: 100
```

Or, if you like to change it from the CLI, you can select a .yaml file and set the flags as follows:
```
python main.py --cf demo_relative_1_query.yaml --data.reference_dir "scripts/exp/cifar10_4" --audit.report_log "results_using_4_reference_models"
```

Note: When you change dataset/the set of target models (`target_dir`), it is better to change the log_dir (e.g. to demo_cifar100 or demo_purchase100) as it is another set of signals and models.


#### Options to set
You can also change some parameters and options to personalize the attack (either in the config file, or by adding flags in the CLI command e.g. --audit.offline "False"):

- `offline` (Boolean: True or False): Indicate whether to use the offline setting (only out models) or the online settings (both in and out models). If set to True, half of the `num_ref_models` are used per sample.
- `offline_a`: floats used to compute $p(x)$ and $p(z)$ from $p_{OUT}(x)$ and $p_{OUT}(z)$. When unknown: `offline_a`=1.
- `num_ref_models` (Int): number of reference models can be between 1 and the max number of reference models in `reference_dir`. For offline attacks, the true number of reference model per sample is `num_ref_models / 2` (OR `(num_ref_models-1)//2` if `target_dir=reference_dir`) . For online attacks, exactly `num_ref_models` (half in, half out) are being used.
- `augmentation` ("augmented" or "none"): Indicate whether to use augmentation for the query points. When set to "none", `nb_augmentation` needs to be set to 2 (that contains both the original query and the mirror flipped query).
- `nb_augmentation` (Int): number of queries used for the attack (should have been computed beforehand using `inference.py`). Can be 2 (original and mirrored), 18 (+shifts), 50...etc.
- `top_k` (-1 or positive Int): number of population z to use for the relative attack, can be -1 (all population is used) anywhere between 1 and around 25K (depends on each training set's size). (default is -1)
- `signal` (str): name of the signal used to perform the relative attack. Used and can be the used for other attacks. For relative attacks, 4 different signals: softmax_relative, taylor_softmax_relative, sm_softmax_relative, sm_taylor_softmax_relative (default)
- `temperature` (float): temperature applied to logits when computing signals (default is 2.0)
- `gamma` (float): gamma threshold for LRT(x,z,theta) (default is 2.0)
- `proportiontocut` (float): float: which proportion of models left and right to remove to compute the trimmed mean for p(x) or p(z)
- `taylor_m` (float): margin of the soft-margin (default is 0.6)
- `taylor_n` (Int): order of the taylor approximation (default is 4)
- `target_idx` (Int or str): idx of the target model, can be int (e.g. 0), "ten", "fifty", or "all" for an average over multiple models (default is "ten", and `target_dir` needs to contain at least 10 models)
- `subset` (str): subset on which the attack is performed, can be "all", "typical", "atypical", "not typical", "not atypical" (default is "all")

#### To execute the attack

Once the .yaml file is set (or the CLI command), you can run the code
```
python main.py --cf demo_relative_1_query.yaml
```

You can also execute the attack bash files to compare between attacks (plots are created inside each dataset folder):
```
bash attack_cifar10.sh
```

To check the results of a particular attack, you can go into the report folder and check the `log_time_analysis.log` file, which should contain the results (AUC and TPR@lowFPR).

#### To plot the results
You can directly change [`plot.py`](plot.py). To compare different results, change `attack_list_and_paths` and the `--log_dir` flag (and rename the .png file name it will be saved to):

```
attack_list_and_paths  = [
("2 models", "report_relative_online_2_ref_model"), # (label, report_log)
("4 models", "report_relative_online_4_ref_model"),
]
```
and execute:
```
python plot.py --log_dir "demo_cifar10"
```

#### To reproduce figures

To reproduce the main figures in the paper, you can first execute the corresponding training/infer bash script in [scripts/figure_scripts/](scripts/figure_scripts/) (e.g. `bash figure_scripts/train_....sh` inside `scripts/`), then execute the corresponding attack script in [figure_scripts/](figure_scripts/). (e.g. `bash figure_scripts/attack_....sh` in the current folder)

### Some particular experiments

The following commands are used to reproduce particular settings and results in the paper.

#### Multiple Architectures (Both Multiple Target Architecture and Fixed Target Architecture)

In the /scripts folder [there](scripts/), just run `bash figure_scripts/train_cifar10_multi_arch.sh` to train the different architectures' target and reference models. Once done, you can return to the root folder and execute `bash figure_scripts/attack_cifar10_multi_arch.sh`.

#### DP-SGD

In the /scripts folder [there](scripts/), just run `bash figure_scripts/dp_train_cifar10.sh` to train the DP-SGD trained target and reference models. Once done, you can go back to the root folder and execute `bash figure_scripts/attack_dp_models.sh`.

#### Gradient Boosted Decision Trees (GBDT)

You may first execute in the CLI `pip install xgboost`. Then, in the /scripts folder [there](scripts/), just run `bash figure_scripts/train_purchase100_gbdt.sh` to train the GBDT target and reference models. Once done, you can return to the root folder and execute `bash figure_scripts/attack_purchase100_gbdt.sh`.

#### Various Hyperparameters (AUC vs. gamma, offline a, etc.)

In the /scripts folder [there](scripts/), you first run `bash figure_scripts/train_cifar10_256_models.sh` (trains 256 models for CIFAR-10). Once done, you can go back to the root folder and execute `bash figure_scripts/stats_and_gammas.sh` (for AUC vs. gamma), `bash figure_scripts/stats_and_offline_as.sh` (for AUC vs. offline a) or `bash figure_scripts/stats_and_hyperparams.sh` (for AUC vs. n, m and T).

### Extra 

#### Regarding / Changing Non-members

To run attacks on different Non-members on a specific set of models, you need to:
- Copy paste the folder containing all the models for which you want to evaluate the MIA, e.g. [scripts/exp/](scripts/exp/), you can copy paste the folder of `cifar10` and rename the new folder to e.g. `cifar10_noisy_members`.
- You then need to replace the `x_train.npy` and `y_train.npy` with the new ones (either another dataset, or containing noise). 
- Then, you should execute the normal steps infer the logits using `inference.py` from Carlini's code, e.g. you can check [scripts/figure_scripts/infer_noisy_members.sh](scripts/figure_scripts/infer_noisy_members.sh).
- Finally, execute the attacks using a modified `main.py`, either use (you may see examples of config files and commands in [figure_scripts/attack_ood_non_members.sh](figure_scripts/attack_ood_non_members.sh)):
	- `main_ood.py` : Use `target_dir` and `reference_dir`'s member signals as True members, and any of `extra_target_dir` and `extra_reference_dir`'s signals as non-members. (e.g. when we evaluate on just noise, or on OOD data). See [attack_configs/cifar10/rmia_offline_127_ref_models_ood.yaml](attack_configs/cifar10/rmia_offline_127_ref_models_ood.yaml) for an example of yaml file.
	- `main_noise_ood.py` : Use `target_dir` and `reference_dir`'s member signals as True members, and the members' perturbed signals of `extra_target_dir` and `extra_reference_dir` as non-members. 
