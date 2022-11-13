import numpy as np

def forward(x, w, v, H, f, return_nodes=False):
	z1 = np.matmul(w, x.reshape((-1, 1)))
	h = H(z1)
	
	z2 = np.matmul(v, h)
	y = f(z2)

	if return_nodes is False:
		return y
	else:
		return y, z1, h, z2
	
def calc_gradient(x, y_true, w, v, H, H_, f, f_):
	y, z1, h, z2 = forward(x, w, v, H, f, return_nodes=True)
	e = y - y_true.reshape((-1, 1))

	d_0 = e * f_(z2)
	dE_dv = np.matmul(d_0, h.T)

	d_h = np.matmul(v.T, d_0) * H_(z1)
	dE_dw = np.matmul(d_h, x.reshape((1, -1)))

	return dE_dv, dE_dw

