### Training and Performing Inference

For training target and reference models and for ease of comparison, we use the tool provided by Carlini et al. in [https://github.com/carlini/privacy/tree/better-mi/research/mi_lira_2021](https://github.com/carlini/privacy/tree/better-mi/research/mi_lira_2021).

To train models and infer logits, you need to be in the /scripts folder [there](scripts/). (This folder.)

#### Training/Inference Environment

You can install the conda environment from https://github.com/yuan74/ml_privacy_meter/blob/2022_enhanced_mia/research/2022_enhanced_mia/2022_enhanced_mia.yml and by executing:

```
conda env create -f 2022_enhanced_mia.yml
```

#### Training Models

To train one model idx in a set of N models in the folder named path you can execute: 
```
python3 -u train.py --dataset=cifar10 --epochs=100 --save_steps=100 --arch wrn28-2 --num_experiments $N --expid\$idx --logdir exp/$path &> 'logs/$path/log_$idx'  
```
For example, to train the whole set of CIFAR-10 models used in our paper (Using LiRA's data partitioning), you can execute: 
```
bash train_cifar10.sh
```

Note: For CIFAR-10, CIFAR-100 and CINIC-10 we use Wide Resnet 28-2 and for Purchase-100 we use a simple MLP (see [mlp.py](mlp.py)).

##### Changing the number of reference models

To change the number of reference models, you can train a separate set of reference models that contains exactly the number of reference model you want. For example, if you want to train 4 reference models for the online setting (or 2=4/2 reference models for the offline one), you can use the following bash script (you can just comment/uncomment the relevant lines in the train and infer bash files):
```
prefix="cifar10_4" # example name of the folder containing 4 reference models (2 IN, 2 OUT) or (2 OUT per sample)
if [ !  -d  "logs/${prefix}" ]; then
	# If the folder doesn't exist, create the folder
	mkdir  "logs/${prefix}"
	mkdir  "exp/${prefix}"
	echo  "Folder 'logs/${prefix}' created."
else
	echo  "Folder 'logs/${prefix}' already exists."
fi

n_models_end=3  # train 4=3+1 reference models (starts from 0)
for model in $(seq 0 1 $n_models_end);
do
	train="CUDA_VISIBLE_DEVICES='0' python3 -u train.py --dataset=cifar10 --epochs=100 --save_steps=100 --arch wrn28-2 --num_experiments 4 --expid ${model} --logdir exp/${prefix} &> 'logs/${prefix}/log_${model}'"
	eval ${train}
done
```

##### Changing dataset

You can change the dataset you train on to CIFAR-100 by redefining the dataset flag `--dataset=cifar10` to `--dataset=cifar100` in the bash file, e.g for model idx in a set of N models in the folder path you can execute:

```
python3 -u train.py --dataset=cifar100 --epochs=100 --save_steps=100 --arch wrn28-2 --num_experiments $N --expid $idx --logdir exp/$path &> 'logs/$path/log_$idx'
```

For CINIC-10 and Purchase-100, you can download reformatted datasets' files (`x_train.npy`,`y_train.npy`,`x_test.npy` and `y_test.npy`) [there](https://drive.google.com/drive/folders/1cIJlbLlgqDSJKd8YhTucHwaPiLsh6ZyW?usp=sharing). You can then put those files in `exp/cinic10` and `exp/purchase100` (or any other set of models using those datasets) respectively and then use the flag `--dataset=cinic10` or `--dataset=purchase100` in your training and inference commands/bash files (e.g. see in [`train_purchase100.sh`](train_purchase100.sh)  and [`infer_purchase100.sh`](infer_purchase100.sh)).
  
#### Inferring Models

To infer the logits from the trained models on their respective datasets, you can execute: 
```
CUDA_VISIBLE_DEVICES='0' python3 inference.py --logdir=exp/cifar10/ --aug=2  --dataset=cifar10
```
To infer for more augmentations (e.g. 18 or 50), you can change the number in the aug parameter:
```
CUDA_VISIBLE_DEVICES='0' python3 inference.py --logdir=exp/cifar10/ --aug=18  --dataset=cifar10
```

You need to execute this command for both the set containing your target model and the one containing your reference models: (e.g. see [`infer_cifar10.sh`](infer_cifar10.sh))

```
# infer for target models
CUDA_VISIBLE_DEVICES='0' python3 inference.py --logdir=exp/cifar10/ --aug=2 --dataset=cifar10  # contains the original query
CUDA_VISIBLE_DEVICES='0' python3 inference.py --logdir=exp/cifar10/ --aug=18 --dataset=cifar10

# infer for reference models
CUDA_VISIBLE_DEVICES='0' python3 inference.py --logdir=exp/cifar10_2/ --aug=2 --dataset=cifar10
CUDA_VISIBLE_DEVICES='0' python3 inference.py --logdir=exp/cifar10_2/ --aug=18 --dataset=cifar10
```



## Details of modifications of `inference.py`

Note: for inference it is important to precise the directory to the folder containing all the models (i.e. subfolders of the form experiment-{model_idx}_{number_of_models})

This is a list of changes to https://github.com/carlini/privacy/tree/better-mi/research/mi_lira_2021 so that everything works well:

To be able to specify the number of queries, you have to modify `inference.py` by:

1. adding a "aug" flag in `if __name__ == "__main__"`:

`flags.DEFINE_integer('aug', 0, 'number of queries/augmentations')`

2. add to `main` right after `get_loss` (to set shift, reflect and stride):
```
if FLAGS.aug == 0:
	shift = 0
	reflect = False
	stride = 1
elif FLAGS.aug == 2:
	shift = 0
	reflect = True
	stride = 1
elif FLAGS.aug == 18:
	shift = 1
	reflect = True
	stride = 1
elif FLAGS.aug == 50:
	shift = 2
	reflect = True
	stride = 1

nb_steps = len(range(0, 2*shift+1, stride))
nb_augmentations = nb_steps**2 * 2 if reflect else nb_steps**2
```
3. add those parameters to their `feature` function:
```
def features(model, xbatch, ybatch):
	return get_loss(model, xbatch, ybatch,shift=shift, reflect=reflect, stride=stride)
```
4. modify the name of the output file where the logits are saved (will be used in the MIA):
```
for path in enumerate(sorted(os.listdir(os.path.join(FLAGS.logdir)))):
	# ... some code ...
	np.save(os.path.join(FLAGS.logdir, path, "logits", "%010d_%04d"%(epoch, nb_augmentations)), np.array(stats)[:,None,:,:])
```
5. modify the check that verifies whether the logits are already computed or not. Replace:
```
if os.path.exists(os.path.join(FLAGS.logdir, path, "logits", "%010d.npy"%epoch)):
	print("Skipping already generated file", epoch)
	continue
```
by:
```
if os.path.exists(os.path.join(FLAGS.logdir, path, "logits", "%010d_%04d.npy"%(epoch, nb_augmentations))):
	a = np.load(os.path.join(FLAGS.logdir, path, "logits", "%010d_%04d.npy"%(epoch, nb_augmentations)), allow_pickle=True)
	if a.shape[2] == nb_augmentations:
		print("Skipping already generated file", path, epoch)
		continue
```