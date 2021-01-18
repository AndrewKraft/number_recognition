# python libraries
import numpy as np

# local libraries
import loader
from network import Network

# working with greyscale images, so each pixel value is between 0 and 255 (inclusive)
MIN, MAX = 0, 255
def normalize(x): # fit np.array x to normal (0,1) distribution
	N = len(x)
	mean = sum(x) / N
	stdev = 0
	for x_i in x:
		stdev += (x_i - mean)**2
	return (x - mean) / stdev**(1/2)

def standardize(x): # scale each element of np.array x such that x fits on interval [0,1]
	return (x - MIN) / MAX


print("Loading Dataset...", end="", flush=True)
trainX, trainy, testX, testy = loader.load_data()
print("Complete!")

print("Processing Data...", end="", flush=True)
standardized_trainX = [ standardize(x) for x in trainX ]
standardized_testX = [ standardize(x) for x in testX ]

# These functions could be used if we wanted to fit the data to a normal (0,1) distribution
# 
# normalized_trainX = [ normalize(x) for x in trainX ]
# normalized_testX = [ normalize(x) for x in testX ]
print("Complete!")

print("Training Network...")
sizes = [ 784, 50, 10 ]
clf = Network(sizes)
clf.fit(standardized_trainX, trainy, 32, 1, 16)
print("Complete!")

print("Testing...", end="", flush=True)
predictions = clf.predict(standardized_testX)
correct = 0
total = len(testX)

for l, y in zip(predictions, testy):
	if l == y:
		correct+=1

print("Complete!\nRESULTS:\nAccuracy:\t%0.3f (%s/%s)" % (correct/total, correct, total))