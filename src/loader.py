# python libraries
import gzip, numpy as np

def load_idx(fname):

	# open and unzip file with gzip
	with gzip.open(fname, 'rb') as f:

		# idx file format starts with magic num, detailing the type of data contained and the shape of the container
		magic_num = int.from_bytes(f.read(4), "big")
		data_type = (magic_num >> 8) & 0xff
		dimensions = magic_num & 0xff

		# empty tuple for the shape of the data for use with np.array
		shape = ()
		size = 1
		for i in range(dimensions):

			# create 1 dimensional tuple for size for easy concatenation
			size_i = (int.from_bytes(f.read(4), "big"),)
			shape += size_i
			if i > 0:
				size *= size_i[0]

		data = np.empty((shape[0], size))

		segment_length = 1
		
		# initialize index to iterate through dataset
		index1, index2 = 0, 0
		while True:
			chunk = f.read(segment_length)
			if not chunk:
				break
			data[index1, index2] = int.from_bytes(chunk, "big")
			index2 += 1
			if index2 == size:
				index2 = 0
				index1 += 1

	return data, shape

def load_data():

	# put data from the files into arrays
	training_samples, samples_shape = load_idx("./data/train-images-idx3-ubyte.gz")
	training_labels, _ = load_idx("./data/train-labels-idx1-ubyte.gz")
	testing_samples, test_shape = load_idx("./data/t10k-images-idx3-ubyte.gz")
	testing_labels, _ = load_idx("./data/t10k-labels-idx1-ubyte.gz")

	return training_samples, training_labels, samples_shape, testing_samples, testing_labels, test_shape