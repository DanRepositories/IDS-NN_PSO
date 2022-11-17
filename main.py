import numpy as np
from ann import *
from my_utility import *
from act_functions import get_function
from pso import *
from metrics import get_metrics, get_confusion_matrix
from pso import ann_train_pso

FILE_DATA_TRAIN = 'dtrn.csv'
FILE_LABEL_TRAIN = 'etrn.csv'
FILE_DATA_TEST = 'dtst.csv'
FILE_LABEL_TEST = 'etst.csv'

# Se cargan los datos
X_train, y_train = load_data(FILE_DATA_TRAIN, FILE_LABEL_TRAIN)
X_test, y_test = load_data(FILE_DATA_TEST, FILE_LABEL_TEST)

# Se carga la configuracion
m = X_train.shape[0]		# Dimension del vector de entrada
L = 10				# Cantidad de nodos de la capa oculta
K = y_train.shape[0]		# Cantidad de nodos de la capa de salida
mu = 0.05			# Tasa de aprendizaje
Np = 20				# Cantidad de particulas del enjambre
max_iter_ann = 1500		# Cantidad de iteraciones de la etapa de train
max_iter_pso = 1000		# Cantidad de iteraciones del pso

# Se obtienen las funciones de activacion a utilizar
fun, fun_ = get_function(5)
outfun, outfun_ = get_function(5)

cnf = {'m':m, 'L':L, 'K':K, 'mu':mu, 'Np':Np, 'max_iter_ann': max_iter_ann, 'max_iter_pso':max_iter_pso}
cnf['fun'] = fun
cnf['fun_'] = fun_
cnf['outfun'] = outfun
cnf['outfun_'] = outfun_

f1_scores = []
for i in range(5):
	w, v = ann_train_pso(X_train, y_train, cnf)
	y_predict = np.round(forward(X_test, w, v, fun, outfun)).astype(int)

	#w = np.random.rand(L, m)
	#v = np.random.rand(K, L)

	# Se entrena la red
	#w, v = ann_train(X_train, y_train, w, v, cnf)

	#y_predict = np.round(forward(X_test, w, v, fun, outfun)).astype(int)

	cm = get_confusion_matrix(y_test, y_predict)
	f1 = get_metrics(cm)[-1]
	f1_scores.append(f1)

	print(cm)
	print(get_metrics(cm))
	print()

print(cm)
mean_f1 = np.mean(f1_scores)
std_f1 = np.std(f1_scores)

print(mean_f1, std_f1)
