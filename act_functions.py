import numpy as np

# Funcion que obtiene una funcion de activacion y su derivada dependiendo del numero ingresado
def get_function(num_function):
	if 1 == num_function:
		return np.vectorize(relu), np.vectorize(dev_relu)
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

def sigmoid(x):
	f_x = 1 / (1 + np.exp(-x))
	return f_x

def dev_sigmoid(x):
	f_x = sigmoid(x)
	return f_x * (1 - f_x)
