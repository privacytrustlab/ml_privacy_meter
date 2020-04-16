import cPickle
import numpy as np
import os

def unpickle(file):
    with open(file, 'rb') as fo:
        dict = cPickle.load(fo)
    return dict

traindict = unpickle("./cifar-100-python/train")
testdict  = unpickle("./cifar-100-python/test")

with open("c100.txt.tmp", "w+") as f:
	for i in xrange(len(traindict['data'])):	
		a = ','.join([str(c) for c in traindict['data'][i][:1024]]) + ';' + \
			','.join([str(c) for c in traindict['data'][i][1024:2048]]) + ';' + \
			','.join([str(c) for c in traindict['data'][i][2048:]]) + ';' + \
			str(traindict['fine_labels'][i])
		f.write(a+"\n")

def extract(filepath):
    """
    """
    with open(filepath, "r") as f:
        dataset = f.readlines()
    dataset = map(lambda i: i.strip('\n').decode("utf-8").split(';'), dataset)
    dataset = np.array(list(dataset)) 
    return dataset

dataset = extract('c100.txt.tmp')
os.remove('c100.txt.tmp')
np.save("cifar100_train.txt.npy", dataset)
