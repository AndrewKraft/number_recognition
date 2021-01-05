# python libraries
import matplotlib.pyplot as plt, numpy as np
from sklearn import svm, stats

def sigmoid(z):
	return 1 / (1 + np.exp(-z))

def sigmoid_prime(z):
	sig = sigmoid(z)
	return sig * (1 - sig)

def softmax(inputs):
	denom = 0
	for z in inputs:
		denom += np.exp(z)
	
	outputs = np.array(inputs.shape)
	for i in range(inputs.size):
		outputs[i] = np.exp(inputs[i]) / denom

	return outputs

class nNetwork():

	def __init__(lr, depth, width):
		self.lr = lr # learning rate
		self.depth = depth # number of layers excluding input layer
		self.width = width # number of nodes at each layer except input and output

	def fit(X, y):
		#
		# Takes input of X, a numpy array of shape (N, f) and y, a numpy array of shape (N, l)
		#
		# N is the number of training samples, f is the number of features per sample, and l is the number of values the output label could take
		#
		
		N, f = x.shape
		l = y.shape[1]
		rng = np.random.default_rng()

		# initialize the weights (and biases) used in the network, each will start as a random number in the range [0, 1)
		# 
		# weights and biases will be kept in np arrays whos dimensions match the dimensions of their input and output vectors
		self.weights = []
		for i in range(depth):
			input_size, output_size = self.width + 1, self.width
			if i == 0:
				input_size = f + 1
			if i == self.depth - 1:
				output_size = l

			tmp = rng.random((output_size, input_size))
			self.weights.append(tmp)

		# now that we have the initial weight matrices all generated randomly, we can begin training
		for index in range(N):
			dzdw = [ X[index] ]
			for i in range(self.depth):

				# calculate values at next layer and add to array containing dz/dw
				tmp = sigmoid(self.weights[i][:,:-1] @ dzdw[i] + self.weights[i][:,-1])
				dzdw.append(tmp)
			
			# backpropagation time
			for i in reversed(range(self.depth)):
				


				
	
		

		

