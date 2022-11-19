import numpy as np

# Constantes
_alpha = 1.6732
_lambda = 1.0507

# Funcion que obtiene una funcion de activacion y su derivada dependiendo del numero ingresado
def get_function(num_function):
	if 1 == num_function:
		return np.vectorize(relu), np.vectorize(dev_relu)
	if 2 == num_function:
		return np.vectorize(l_relu), np.vectorize(dev_l_relu)
	if 3 == num_function:
		return np.vectorize(elu), np.vectorize(dev_elu)
	if 4 == num_function:
		return np.vectorize(selu), np.vectorize(dev_selu)
	if 5 == num_function:
		return np.vectorize(sigmoid), np.vectorize(dev_sigmoid)
	else:
		return None

def relu(x):
	if x > 0:
		return x
	else:
		return 0

def dev_relu(x):
	if x > 0:
		return 1
	else:
		return 0

def l_relu(x):
	if x >= 0:
		return x
	else:
		return 0.01 * x

def dev_l_relu(x):
	f_x = l_relu(x)
	return 0  #aquí

def elu(x):
	if x > 0:
		return x
	else:
		return _alpha*(np.exp(x) - 1)

def dev_elu(x):
	f_x = elu(x)
	return 0 #aquí

def selu(x):
	if x > 0:
		return _lambda * x
	else:
		return _lambda * (_alpha*(np.exp(x) - 1))

def dev_selu(x):
	f_x = selu(x)
	return 0 #aquí

def sigmoid(x):
	f_x = 1 / (1 + np.exp(-x))
	return f_x

def dev_sigmoid(x):
	f_x = sigmoid(x)
	return f_x * (1 - f_x)
