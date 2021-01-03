import gzip

def load_idx(fname):
	with gzip.open(fname, 'rb') as f:
		magic_num = f.read(4)
		data_type = magic_num[2]
		dimensions = magic_num[3]
	return magic_num, data_type, dimensions
