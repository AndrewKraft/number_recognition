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
		for i in range(dimensions):

			# create 1 dimensional tuple for size for easy concatenation
			size_i = (int.from_bytes(f.read(4), "big"),)
			shape += size_i

		data = np.empty(shape)
		
		# initialize index to iterate through dataset
		index = (0,) * dimensions
		while True:
			data[index] = int.from_bytes(f.read(1), "big")
			
			# iterate counter, if count in dimension equals size of dimension, reset and increment next dimension
			ndim = dimensions - 1
			while ndim > -1:
				index[ndim] += 1
				if index[ndim] == shape[ndim]:
					index[ndim] = 0
					ndim -= 1
				else:	break
			else:	break

	return data, shape
		

