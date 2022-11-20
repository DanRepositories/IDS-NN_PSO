import numpy as np
from act_functions import get_function

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

def get_mse(y_true, y_pred):
	N = y_true.shape[1]
	e = y_pred - y_true
	mse = np.sum(e ** 2) / N
	return mse
	
def readConfigAndData():
	#Leer los datos para training
	FILE_DATA_TRAIN = 'dtrn.csv'
	FILE_LABEL_TRAIN = 'etrn.csv'
	X_train, y_train = load_data(FILE_DATA_TRAIN, FILE_LABEL_TRAIN)

	#Leer y obtener la configuracion del PSO
	configPSO = np.loadtxt("cnf_ann_pso.csv", dtype=int)
	nFunc, L, Np, max_iter_pso = configPSO[0], configPSO[1], configPSO[2], configPSO[3]

	#Leer y obtener la configuracion del BP
	configBP = np.loadtxt("cnf_ann_bp.csv", dtype=float)
	max_iter_ann, mu = int(configBP[0]), configBP[1]

	#Obtener m y K
	m = X_train.shape[0]		# Dimension del vector de entrada
	K = y_train.shape[0]		# Cantidad de nodos de la capa de salida

	#Obtener la funcion de activacion y derivada correspondiente
	fun, fun_ = get_function(nFunc)
	outfun, outfun_ = get_function(5)
	cnf = {'m':m, 'L':L, 'K':K, 'mu':mu, 'Np':Np, 'max_iter_ann': max_iter_ann, 'max_iter_pso':max_iter_pso}
	cnf['fun'] = fun
	cnf['fun_'] = fun_
	cnf['outfun'] = outfun
	cnf['outfun_'] = outfun_
	cnf['X_train'] = X_train
	cnf['y_train'] = y_train
	return cnf
