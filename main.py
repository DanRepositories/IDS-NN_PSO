import numpy as np
from ann import *
from my_utility import *
from act_functions import get_function
from pso import *

FILE_DATA_TRAIN = 'dtrn.csv'
FILE_LABEL_TRAIN = 'etrn.csv'
FILE_DATA_TEST = 'dtst.csv'
FILE_LABEL_TEST = 'etst.csv'

# Se cargan los datos
X_train, y_train = load_data(FILE_DATA_TRAIN, FILE_LABEL_TRAIN)
X_test, y_test = load_data(FILE_DATA_TEST, FILE_LABEL_TEST)

# Se carga la configuracion
m = X_train.shape[1]		# Dimension del vector de entrada
L = 5				# Cantidad de nodos de la capa oculta
K = y_train.shape[1]		# Cantidad de nodos de la capa de salida
mu = 0.1			# Tasa de aprendizaje
Np = 10				# Cantidad de particulas del enjambre

# Se inicializan los pesos
w = np.random.rand(L, m)
v = np.random.rand(K, L)


swarm = init_swarm(m, L, K, Np)
print(swarm.shape) 

"""
# Se obtienen las funciones de activacion a utilizar
fun, der_fun = get_function(1)

y_predict = forward(X_train[0], w, v, fun, fun)

dE_dv, dE_dw = calc_gradient(X_train[0], y_train[0], w, v, fun, der_fun, fun, der_fun)

u = u - mu * dE_dv
w = w - mu * dE_dw
"""
