# python libraries
import matplotlib.pyplot as plt
from sklearn import datasets, svm, metrics
from sklearn.model_selection import train_test_split

# local libraries
import loader

n, t, d = loader.load_idx("./data/train-images-idx3-ubyte.gz")
print("n:\t%s\nt:\t%s\nd:\t%s\n" % (n, t, d))