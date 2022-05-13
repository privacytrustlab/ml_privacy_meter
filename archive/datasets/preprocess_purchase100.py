import numpy as np 
with open("dataset_purchase", "r") as f:
    dataset = f.readlines()
with open("purchase100.txt", "w+") as f:
	for datapoint in dataset: 
		split = datapoint.rstrip().split(",")
		label = int(split[0]) - 1
		rearranged = ','.join(split[1:]) + ";" + str(label) + "\n"
		f.write(rearranged)