import numpy as np
from ann import *
from my_utility import *
from act_functions import get_function
from pso import *
from metrics import get_metrics, get_confusion_matrix

FILE_DATA_TRAIN = 'dtrn.csv'
FILE_LABEL_TRAIN = 'etrn.csv'
FILE_DATA_TEST = 'dtst.csv'
FILE_LABEL_TEST = 'etst.csv'

# Se cargan los datos
X_train, y_train = load_data(FILE_DATA_TRAIN, FILE_LABEL_TRAIN)
X_test, y_test = load_data(FILE_DATA_TEST, FILE_LABEL_TEST)

# Se carga la configuracion
m = X_train.shape[0]		# Dimension del vector de entrada
L = 5				# Cantidad de nodos de la capa oculta
K = y_train.shape[0]		# Cantidad de nodos de la capa de salida
mu = 0.2			# Tasa de aprendizaje
Np = 10				# Cantidad de particulas del enjambre
max_iter = 2000			# Cantidad de iteraciones de la etapa de train

# Se inicializan los pesos
w = np.random.rand(L, m)
v = np.random.rand(K, L)


# Se obtienen las funciones de activacion a utilizar
fun, fun_ = get_function(5)
outfun, outfun_ = get_function(5)

# Se entrena la red
w, v = ann_train(X_train, y_train, w, v, fun, fun_, outfun, outfun_, mu, max_iter)

y_predict = np.round(forward(X_test, w, v, fun, outfun)).astype(int)

cm = get_confusion_matrix(y_test, y_predict)
print(cm)
print(get_metrics(cm))


i = 10
print(y_predict[:, :i].T)
print(y_test[:, :i].T)
