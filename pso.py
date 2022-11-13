import numpy as np

def init_weights(m, L, K):
	r1 = np.sqrt(6 / (L + m))
	r2 = np.sqrt(6 / (K + L))

	w = np.random.rand(L, m) * 2 * r1
	v = np.random.rand(K, L) * 2 * r2
	
	return w, v

def init_swarm(m, L, K, Np):
	swarm = []
	for i in range(Np):
		w, v = init_weights(m, L, K)
		new_particle = np.concatenate((w, v), axis=None)
		swarm.append(new_particle)

	return np.array(swarm)

def get_alpha(current_iter, max_iter, a_min=0.1, a_max=0.95):
	a = a_max - ((a_max - a_min) / max_iter) * current_iter
	return a

def ann_fitness():
	pass

def update_fitness():
	pass

def update_veloc():
	pass

def ann_train_pso(m, L, K, Np, max_iter):
	S = init_swarm(m, L, K, Np)

	for i in range(max_iter):
		pass
		
