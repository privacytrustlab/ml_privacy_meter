import pickle
import pandas as pd

def unpickle(filepath):
    return pd.read_pickle(filepath)

traindict = unpickle("./cifar-100-python/train")
testdict  = unpickle("./cifar-100-python/test")

with open("cifar100.txt", "w+") as f:
	for i in range(len(traindict['data'])):
		a = ','.join([str(c) for c in traindict['data'][i][:1024]]) + ';' + \
			','.join([str(c) for c in traindict['data'][i][1024:2048]]) + ';' + \
			','.join([str(c) for c in traindict['data'][i][2048:]]) + ';' + \
			str(traindict['fine_labels'][i])
		f.write(a+"\n")

with open("cifar100.txt", "a") as f:
	for i in range(len(testdict['data'])):
		a = ','.join([str(c) for c in testdict['data'][i][:1024]]) + ';' + \
			','.join([str(c) for c in testdict['data'][i][1024:2048]]) + ';' + \
			','.join([str(c) for c in testdict['data'][i][2048:]]) + ';' + \
			str(testdict['fine_labels'][i]) 
		f.write(a+"\n")
