# python libraries
import gzip, numpy as np

def load_idx(fname):

	# open and unzip file with gzip
	with gzip.open(fname, 'rb') as f:

		# idx file format starts with magic num, detailing the type of data contained and the shape of the container
		magic_num = int.from_bytes(f.read(4), "big")
		data_type = (magic_num >> 8) & 0xff
		dimensions = magic_num & 0xff

		# next come the dimensions
		shape = ()
		size = 1
		for i in range(dimensions):
			# create 1 dimensional tuple of size for easy concatenation
			size_i = (int.from_bytes(f.read(4), "big"),)
			shape += size_i
			if i > 0:
				size *= size_i[0]

		data = []

		if data_type == 0x08: # This is the only data type in the file format I want to load so im not gonna add any others
			segment_length = 1
		else: # you could add support for other data types here
			segment_length = 1

		# initialize index to iterate through dataset
		count = 0
		data_i = []
		while(True):
				chunk = f.read(segment_length)
				if not chunk:
					break
				
				if size==1:
					data.append(int.from_bytes(chunk, "big"))
				else:
					if count==size:
						data.append(data_i)
						count = 0
						data_i = []
					
					data_i.append(int.from_bytes(chunk, "big"))
					count+=1

	return data, shape

def vectorize(x):
	l = np.zeros((10,1))
	l[x] = 1
	return l

def load_data():

	# put data from the files into arrays
	trainX, shape_trainX = load_idx("./data/train-images-idx3-ubyte.gz")
	trainy, shape_trainY = load_idx("./data/train-labels-idx1-ubyte.gz")
	testX, shape_testX = load_idx("./data/t10k-images-idx3-ubyte.gz")
	testy, shape_testy = load_idx("./data/t10k-labels-idx1-ubyte.gz")

	trainX = [ np.reshape(x, (784, 1)) for x in trainX ]
	testX = [ np.reshape(x, (784, 1)) for x in testX ]
	trainy = [ vectorize(y) for y in trainy ]

	return trainX, trainy, testX, testy