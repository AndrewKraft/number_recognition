# local libraries
import loader
from network import nNetwork

trainX, trainy, train_size, testX, testy, test_size = loader.load_data()

clf = nNetwork(3)
clf.fit(trainX, trainY)
predictions = clf.predict(testX)

print("")