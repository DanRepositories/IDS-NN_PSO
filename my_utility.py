import numpy as np

def get_one_hot(y):
	K = np.unique(y).shape[0]
	res = np.eye(K)[(y-1).reshape(-1)]
	return res.reshape(list(y.shape)+[K]).astype(int)

def load_data(file_data, file_label):
	X = np.loadtxt(file_data, delimiter=',', dtype=float)
	y = np.loadtxt(file_label, dtype=float)

	data_set = np.concatenate((X, y.reshape((-1, 1))), axis=1)
	np.random.shuffle(data_set)

	y = data_set[:, -1].astype(int)
	X = np.delete(data_set, -1, axis=1)
	return X.T, get_one_hot(y).T

def get_mse(y_predict, y_true):
	e = y_predict - y_true
	mse = np.sum(sp.sqrt(e))
	return mse
	
