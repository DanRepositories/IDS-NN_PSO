import numpy as np

# Funcion que obtiene una funcion de activacion y su derivada dependiendo del numero ingresado
def get_function(num_function):
	if 1 == num_function:
		return np.vectorize(relu), np.vectorize(dev_relu)
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
