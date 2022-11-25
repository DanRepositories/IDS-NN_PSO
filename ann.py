import numpy as np
from my_utility import get_mse
from act_functions import activation_function

def forward(x, w, v, H, f, return_nodes=False):
	z1 = np.matmul(w, x)
	h = activation_function(1, z1)
	
	z2 = np.matmul(v, h)
	y = activation_function(f, z2)

	if return_nodes is False:
		return y
	else:
		return y, z1, h, z2
	
def calc_gradient(x, y_true, w, v, H, f):
	y, z1, h, z2 = forward(x, w, v, H, f, return_nodes=True)
	e = y - y_true

	mse = get_mse(y_true, y)

	d_0 = e * activation_function(f, z2, derivate=True)
	dE_dv = np.matmul(d_0, h.T)

	d_h = np.matmul(v.T, d_0) * activation_function(1, z1, derivate=True)
	dE_dw = np.matmul(d_h, x.T)

	return dE_dw, dE_dv, mse

def ann_train(w, v, cnf):
	x = cnf['X_train']
	y_true = cnf['y_train']
	max_iter = cnf['max_iter_ann']
	mu = cnf['mu']
	H = cnf['fun']
	f = cnf['outfun']

	ann_MSE = [0] * max_iter

	for i in range(max_iter):
		dE_dw, dE_dv, mse = calc_gradient(x, y_true, w, v, H, f,)

		v = v - mu * dE_dv
		w = w - mu * dE_dw
		ann_MSE[i] = mse

	return w, v, ann_MSE
