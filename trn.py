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
mu = 0.001			# Tasa de aprendizaje
Np = 10				# Cantidad de particulas del enjambre
max_iter_ann = 800		# Cantidad de iteraciones de la etapa de train
max_iter_pso = 200		# Cantidad de iteraciones del pso

# Se obtienen las funciones de activacion a utilizar
fun, fun_ = get_function(5)
outfun, outfun_ = get_function(5)

cnf = {'m':m, 'L':L, 'K':K, 'mu':mu, 'Np':Np, 'max_iter_ann': max_iter_ann, 'max_iter_pso':max_iter_pso}
cnf['fun'] = fun
cnf['fun_'] = fun_
cnf['outfun'] = outfun
cnf['outfun_'] = outfun_

#Se realiza PSO
w, v, pso_MSE = ann_train_pso(X_train, y_train, cnf)
#Se guarda la lista de los MSE del PSO por iteración
np.savetxt("costo_pso.csv", pso_MSE)

# Se entrena la red
w, v, ann_MSE = ann_train(X_train, y_train, w, v, cnf)
#Se guarda la lista de los MSE de la ANN por iteración
np.savetxt("costo_gd.csv", ann_MSE)

#Se guardan los pesos de la capa oculta(w) y de la capa de salida(v)
np.savez("pesos.npz", w, v)