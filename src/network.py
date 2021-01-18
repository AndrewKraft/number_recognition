# python libraries
import matplotlib.pyplot as plt, numpy as np, random


def sigmoid(z): # calculate sigmoid function
	return 1 / (1 + np.exp(-z))

def sigmoid_prime(z): # calculates derivative of sigmoid function with respect to z
	sigz = sigmoid(z)
	return sigz * (1 - sigz)

class Network():

	def __init__(self, widths):

		self.depth = len(widths) # number of layers (including the input layer)
		self.widths = widths # number of nodes at each layer with widths[0] representing the input layer and widths[-1] the output layer
		self.biases = [ np.random.randn(y, 1) for y in widths[1:] ]
		self.weights = [ np.random.randn(y, x) for x,y in zip(widths[:-1], widths[1:]) ]

	def feedforward(self, a):
		for w, b in zip(self.weights, self.biases):
			a = sigmoid(w @ a + b)
		return a

	def predict(self, X):
		# takes input vector x of size N, returns an output vector of size (N,) containing the predicted labels for each sample
		N = len(X)
		output = np.empty(N)
		for i in range(N):
			output[i] = np.argmax(self.feedforward(X[i]))
		return output

	def fit(self, X, y, nepoch=32, lr=1, batch_size=16):
		#
		# Takes input of X, a numpy array of shape (N, f) and y, a numpy array of shape (N, m)
		#
		# N is the number of training samples 
		# f is the size of the input layer and m is the size of the output layer, but these values aren't stored because we already have them in self.widths
		#

		N = len(X)

		for nepoch in range(nepoch):
			print("Running epoch %s" % (nepoch))
			# randomly shuffle the training data
			samples = list(zip(X, y))
			random.shuffle(samples)

			minibatches = [ samples[k:k+batch_size] for k in range(0, N, batch_size) ]

			# for each training sample we run the training algorithm
			for minibatch in minibatches:
				# these will store the derivatives of C with respect to each weight and bias
				grad_w = [ np.zeros(w.shape) for w in self.weights ]
				grad_b = [ np.zeros(b.shape) for b in self.biases ]

				for sample, label in minibatch:

					a = [ sample ] # holds the value of the activation layer, needed for each successive layer and for backpropagation
					z = [] # holds the value of the weighted output layer, needed for backpropagation

					# feedforward while updating a, z
					for w, b in zip(self.weights, self.biases):
						# calculate a, z for each successive layer
						z_i = w @ a[-1] + b
						a_i = sigmoid(z_i)
						a.append(a_i)
						z.append(z_i)

					# backpropagation time
					delta = (a[-1] - label) * sigmoid_prime(z[-1])
					grad_w[-1] = np.dot(delta, a[-2].T)
					grad_b[-1] = delta

					for i in range(2, self.depth):
						delta = np.dot(self.weights[1-i].T, delta) * sigmoid_prime(z[-i])
						grad_w[-i] += np.dot(delta, a[-i-1].T)
						grad_b[-i] += delta
				
				# update self.weights and self.biases after each minibatch
				self.weights = [ w - lr * d_w / batch_size for w, d_w in zip(self.weights, grad_w) ]
				self.biases = [ b - lr * d_b / batch_size for b, d_b in zip(self.biases, grad_b) ]

	
		

		

