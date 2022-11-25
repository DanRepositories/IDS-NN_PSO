import numpy as np

# Constantes
_alpha_elu = 0.1
_alpha_selu = 1.6732
_lambda = 1.0507

# Funcion que obtiene una funcion de activacion y su derivada dependiendo del numero ingresado
def activation_function(num_function, x, derivate=False):
	if 1 == num_function:	# Relu
		if not derivate:
			return np.maximum(0, x)
		else:
			return np.greater(x, 0).astype(float)
	if 2 == num_function:	# L-Relu
		if not derivate:
			return np.maximum(0.01 * x, x)
		else:
			return np.piecewise(x, [x <= 0, x > 0], [lambda e: 0.01, lambda e: 1])
	if 3 == num_function: 	# ELU
		if not derivate:
			return np.maximum(_alpha_elu * (np.exp(x) - 1), x)
		else:
			return np.piecewise(x, [x <= 0, x > 0], [lambda e: 0.1 * np.exp(e), lambda e: 1])
	if 4 == num_function:	# SELU
		if not derivate:
			return np.maximum(_lambda * _alpha_selu * (np.exp(x) - 1), _lambda * x)
		else:
			return np.piecewise(x, [x <= 0, x > 0], [lambda e: _lambda * _alpha_selu * np.exp(e), lambda e: _lambda])
	if 5 == num_function:	# Sigmoide
		if not derivate:
			return sigmoid(x)
		else:
			return dev_sigmoid(x)
	else:
		return None

def sigmoid(x):
	f_x = 1 / (1 + np.exp(-x))
	return f_x

def dev_sigmoid(x):
	f_x = sigmoid(x)
	return f_x * (1 - f_x)
