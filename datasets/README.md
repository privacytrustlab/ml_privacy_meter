## Datasets

There are download scripts for following datasets:

1. Purchase100 : This dataset is based on Kaggle’s “acquire valued
shopper” challenge. The dataset includes shopping records for
several thousand individuals. The goal of the challenge is to find
offer discounts to attract new shoppers to buy new products. Each data 
record corresponds to one costumer and has 600 binary features (each corresponding 
to one item). Each feature reflects if the item is purchased by the costumer
or not. The data is clustered into 100 classes and the task is to pre-
dict the class for each costumer. The dataset contains 197,324 data
records.

2. CIFAR100 : This is a major benchmark dataset used to evaluate
image recognition algorithms. The dataset contains 60,000 im-
ages, each composed of 32 × 32 color pixels. The records are clus-
tered into 100 classes, where each class represents one object. 

You can find 'tgz' files of some of these datasets over here: [here](https://www.comp.nus.edu.sg/~reza/files/datasets.html).

### Downloading datasets:

To generate a particular dataset (named 'dataset_name') in text format as required by  `ml_privacy_meter's`  data loading mechanisms, run the script `sh download_<dataset_name>`. Note that the given scripts call the data processing programs using `python2`, so you need to have Python 2 and `numpy` with Python 2 support (e.g. `v1.16.6`) installed before running them. Eg: For `purchase100`, run:

```
sh download_purchase100.sh
```
The generated text files are in the format described below. For your own datasets you will need to transform them in the following format. 

### Required format for datasets:

1. For Fully Connected Networks (FCNs): a comma + colon seperated list of features and labels (with the *label* present in last column). Eg:

```
....features...;label
```
where features is a comma seperated list and label and features is seperated by a colon. 

2. For Convolutional Neural Networks (CNNs): a comma + colon seperated list (with the *label* present in the 
last column). Eg: 
```
channel 1 features; channel 2 features; channel 3 features;...; channel n features; label 
``` 
where features of individual channels are a comma seperated list, with each feature list seperated by a colon. 
