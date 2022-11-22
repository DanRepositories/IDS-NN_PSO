import numpy as np
from my_utility import get_one_hot
from sv_utility import normalize, substract_mean, filter_data

# Se cargan los archivos con las caracteristicas escogidas
index = np.loadtxt('index.csv', dtype=int)

# Se carga la matriz v para realizar el filtrado
matrix_v = np.loadtxt('filter_v.csv', dtype=float, delimiter=',')
k, d = matrix_v.shape


def prepro_data(file_data, file_data_out, file_label_out):
	# Se cargan los datos y se obtienen dividen la data de la etiqueta
	raw_data = np.loadtxt(file_data, delimiter=',', dtype=float)
	X = np.delete(raw_data, -1, axis=1).T
	y = get_one_hot(raw_data[:, -1].astype(int))

	# Se obtienen las caracteristicas seleccionadas
	X = normalize(X[index - 1])

	X_0 = substract_mean(X, k)
	X = filter_data(X, matrix_v, d)

	np.savetxt(file_data_out, X.T, delimiter=',')
	np.savetxt(file_label_out, y, delimiter=',', fmt='%d')

prepro_data('dtrn.csv', 'xtrn.csv', 'ytrn.csv')
prepro_data('dtst.csv', 'xtst.csv', 'ytst.csv')

