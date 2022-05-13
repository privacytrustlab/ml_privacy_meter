import cPickle
def unpickle(file):
    with open(file, 'rb') as fo:
        dict = cPickle.load(fo)
    return dict

traindict = unpickle("./cifar-100-python/train")
testdict  = unpickle("./cifar-100-python/test")

with open("cifar100.txt", "w+") as f:
	for i in xrange(len(traindict['data'])):	
		a = ','.join([str(c) for c in traindict['data'][i][:1024]]) + ';' + \
			','.join([str(c) for c in traindict['data'][i][1024:2048]]) + ';' + \
			','.join([str(c) for c in traindict['data'][i][2048:]]) + ';' + \
			str(traindict['fine_labels'][i])
		f.write(a+"\n")

with open("cifar100.txt", "a") as f:
	for i in xrange(len(testdict['data'])):	
		a = ','.join([str(c) for c in testdict['data'][i][:1024]]) + ';' + \
			','.join([str(c) for c in testdict['data'][i][1024:2048]]) + ';' + \
			','.join([str(c) for c in testdict['data'][i][2048:]]) + ';' + \
			str(testdict['fine_labels'][i]) 
		f.write(a+"\n")