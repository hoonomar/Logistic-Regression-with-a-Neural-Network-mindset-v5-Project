import numpy as np
import matplotlib.pyplot as plt
import h5py 
import scipy
from PIL import Image
from scipy import ndimage
from lr_utils import load_dataset

plt.show()

#Loading the data (cat/non-cat)

train_set_x_orig, train_set_y, test_set_x_orig, test_set_y, classes = load_dataset()

#example picture

index = 25
plt.imshow(train_set_x_orig[index])
print ("y = " + str(train_set_y[:, index]) + ", it's a '" + classes[np.squeeze(train_set_y[:, index])].decode("utf-8") +  "' picture.")

#Finding values 

m_train = train_set_x_orig.shape[0]
m_test = test_set_x_orig.shape[0]
num_px = test_set_x_orig.shape[1] 

print("m_train value = " +  str(m_train) + "\n m_test value = " + str(m_test) + "\n num_px value = " + str(num_px))

print ("Number of training examples: m_train = " + str(m_train))
print ("Number of testing examples: m_test = " + str(m_test))
print ("Height/Width of each image: num_px = " + str(num_px))
print ("Each image is of size: (" + str(num_px) + ", " + str(num_px) + ", 3)")
print ("train_set_x shape: " + str(train_set_x_orig.shape))
print ("train_set_y shape: " + str(train_set_y.shape))
print ("test_set_x shape: " + str(test_set_x_orig.shape))
print ("test_set_y shape: " + str(test_set_y.shape))

#Reshape the training and test examples

train_set_x_flatten = train_set_x_orig.reshape(train_set_x_orig.shape[0], -1).T
test_set_x_flatten = test_set_x_orig.reshape(test_set_x_orig.shape[0], -1).T
print ("train_set_x_flatten shape: " + str(train_set_x_flatten.shape))
print ("train_set_y shape: " + str(train_set_y.shape))
print ("test_set_x_flatten shape: " + str(test_set_x_flatten.shape))
print ("test_set_y shape: " + str(test_set_y.shape))
print ("sanity check after reshaping: " + str(train_set_x_flatten[0:5,0]))

#Standardize the data set
train_set_x = train_set_x_flatten/255.
test_set_x = test_set_x_flatten/255.

# Graded fucntion: sigmoid

def sigmoid(z):

	s = 1 / (1 + np.exp(-z))
	return s

print ("sigmoid([0, 2]) = " + str(sigmoid(np.array([0,2]))))


#Creating a vector of zeros of shape(dim, 1)

def initialize_with_zeros(dim):

	w = np.zeros((2, 1))	
	b = 0

	assert(w.shape ==(dim, 1))
	assert(isinstance(b, float) or isinstance(b, int))
	
	return w, b

dim = 2
w, b = initialize_with_zeros(dim)
print ("w = " + str(w))
print ("b = " + str(b)) 


#Propagation

def propagate(w, b, X, Y):

	m = X.shape[1]
	#Foward Propagation
	A = sigmoid(w*X + b)    
	cost = -1/m * np.sum( np.multiply(np.log(A), Y) + np.multiply(np.log(1-A), (1-Y)))

	#Backward Propagation
	dw = 1/m * X * (A - Y)
	db = 1/m * np.sum(A - Y)

	assert(dw.shape == w.shape)
	assert(db.dtype == float)
	cost = np.squeeze(cost)
	assert(cost.shape == ())

	grads = {"dw" : dw,
		 "db" : db}

	return grads, cost

w, b, X, Y = np.array([[1.],[2.]]), 2., np.array([[1.,2.,-1.],[3.,4.,-3.2]]), np.array([[1,0,1]])
grads, cost = propagate(w, b, X, Y)
print ("dw = " + str(grads["dw"]))
print ("db = " + str(grads["db"]))
print ("cost = " + str(cost))
