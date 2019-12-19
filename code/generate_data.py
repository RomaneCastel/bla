import argparse
import os

import numpy as np
import math
from tensorflow.keras.datasets import mnist

parser = argparse.ArgumentParser(description="generate data for the 10 neural networks we have")
parser.add_argument("--nbexamples", required=True, help="number of examples to generate per neural network")

args = parser.parse_args()

# load mnist dataset with 60k mnist examples
(X_train, y_train),(X_test, y_test) = mnist.load_data()

number_total_examples = 10*int(args.nbexamples)

X_train = X_train[:number_total_examples]
y_train = y_train[:number_total_examples]

# Shuffle X and Y in the same way
p = np.random.permutation(len(X_train))
X_train, y_train = X_train[p], y_train[p]

# flatten X_train
X_train_flattened = np.zeros((len(X_train), 28*28))
for j in range(len(X_train)):
	X_train_flattened[j] = X_train[j].flatten() / 255

epsilon = 0.1
image = 0
folders = ['conv1', 'conv2', 'conv3', 'conv4', 'conv5','fc1','fc2','fc3','fc4','fc5']
folder = 0
for i in range(len(X_train)):
	np.savetxt("../dataset/"+folders[folder]+"/img"+str(math.floor(image/10))+"_{0:.5f}".format(epsilon)+".txt", X_train_flattened[i], delimiter='\n')
	with open("../dataset/"+folders[folder]+"/img"+str(math.floor(image/10))+"_{0:.5f}".format(epsilon)+".txt", 'r+') as file:
		a = file.read()
		with open("../dataset/"+folders[folder]+"/img"+str(math.floor(image/10))+"_{0:.5f}".format(epsilon)+".txt", 'w+') as file:
			file.write(str(y_train[i])+'\n'+a)
	image += 1

	folder += 1
	folder %= 10
	if folder == 0:
		epsilon += (0.2 - 0.1) / (int(args.nbexamples) - 1)
		if epsilon > 0.2:
			epsilon = 0.1
