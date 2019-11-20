import numpy as np
import math
from tensorflow.keras.datasets import mnist

# load mnist dataset with 60k mnist examples
(X_train, y_train),(X_test, y_test) = mnist.load_data()

# flatten X_train
X_train_flattened = np.zeros((len(X_train), 28*28))
for i in range(len(X_train)):
	X_train_flattened[i] = X_train[i].flatten()

epsilon = 0.005
image = 0
folders = ['conv1', 'conv2', 'conv3', 'conv4', 'conv5','fc1','fc2','fc3','fc4','fc5']
folder = 0
for img in range(len(X_train)):
	np.savetxt("../dataset/"+folders[folder]+"/img"+str(math.floor(image/10))+"_{0:.5f}".format(epsilon)+".txt", X_train_flattened[i], delimiter='\n')
	with open("../dataset/"+folders[folder]+"/img"+str(math.floor(image/10))+"_{0:.5f}".format(epsilon)+".txt", 'w+') as file:
		a = file.read()
		with open("../dataset/"+folders[folder]+"/img"+str(math.floor(image/10))+"_{0:.5f}".format(epsilon)+".txt", 'w+') as file:
			file.write(str(y_train[i])+'\n'+a)
	image += 1
	epsilon += 0.005
	if epsilon > 0.2:
		epsilon = 0
	folder += 1
	folder %= 10
