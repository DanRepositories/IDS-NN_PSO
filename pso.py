import numpy as np
from my_utility import get_mse
from ann import forward

def init_weights(m, L, K):
	r1 = np.sqrt(6 / (L + m))
	r2 = np.sqrt(6 / (K + L))

	w = np.random.rand(L, m) * 2 * r1 - r1
	v = np.random.rand(K, L) * 2 * r2 - r2
	
	return w, v

def init_swarm(m, L, K, Np):
	swarm = []
	for i in range(Np):
		w, v = init_weights(m, L, K)
		new_particle = np.concatenate((w, v), axis=None)
		swarm.append(new_particle)

	return np.array(swarm)

def init_pso(m, L, K, Np):
	S = init_swarm(m, L, K, Np)
	P = np.zeros(S.shape)
	Pg = np.zeros(S.shape[1])
	P_fit = np.zeros(Np)
	Pg_fit = 0

	return S, P, Pg, P_fit, Pg_fit

def extract_weights(particle, m, L, K):
	w = particle[:m * L].reshape((L, m))
	v = particle[m * L:].reshape((K, L))
	return w, v

def get_alpha(current_iter, max_iter, a_min=0.1, a_max=0.95):
	a = a_max - ((a_max - a_min) / max_iter) * current_iter
	return a

def swarm_fitness(X, y_true, m, L, K, H, f):
	pass

def particle_fitness(particle, X, y_true, m, L, K, H, f):
	w, v = extract_weights(particle, m, L, K)
	y_predict = forward(X, w, v, H, f)

	mse = get_mse(y_predict, y_true)
	return mse 

def update_fitness():
	pass

def update_veloc():
	pass

def ann_train_pso(m, L, K, Np, max_iter):
	S, P, pg, P_fit, Pg_fit = init_pso(m, L, K, Np)

	for i in range(max_iter):
		pass
		
