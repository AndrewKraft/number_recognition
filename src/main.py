# python libraries
import matplotlib.pyplot as plt
from sklearn import datasets, svm, metrics

# local libraries
import loader

trainX, shape = loader.load_idx("./data/train-images-idx3-ubyte.gz")
print("shape:\t{shape}")