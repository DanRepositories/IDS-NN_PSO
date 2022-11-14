import numpy as np
from my_utility import get_mse
from ann import forward

# Constantes
C1 = 1.05
C2 = 2.95

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
	P_fit = np.zeros(Np)		# TODO: Inicializar fitness
	Pg_fit = P_fit[0]

	return S, P, Pg, P_fit, Pg_fit


def extract_weights(particle, m, L, K):
	w = particle[:m * L].reshape((L, m))
	v = particle[m * L:].reshape((K, L))
	return w, v


def get_alpha(max_iter, current_iter, a_min=0.1, a_max=0.95):
	a = a_max - ((a_max - a_min) / max_iter) * current_iter
	return a


def ann_fitness(S):
	pass


def update_fitness(S, S_fit, P, P_fit, Pg, Pg_fit, Np):
	i_min = 0
	for i in range(Np):
		if S_fit[i] < P_fit[i]:
			P[i] = S[i].copy()
			P_fit[i] = S_fit[i]
		
		if S_fit[i] < S_fit[i_min]:
			i_min = i

	if S_fit[i_min] < Pg_fit:
		Pg = S[i_min].copy()
		Pg_fit = S_fit[i_min]

	return P, P_fit, Pg, Pf_fit


def update_veloc(S, V, P, Pg, max_iter, current_iter):
	a = get_alpha(max_iter, current_iter)
	r1 = np.random.rand()
	r2 = np.random.rand()

	V_k = a*V + C1*r1*(S - P) + C2*r2*(S - Pg)
	return V_k


def ann_train_pso(m, L, K, Np, max_iter):
	S, P, pg, P_fit, Pg_fit = init_pso(m, L, K, Np)

	for i in range(max_iter):
		S_fit = ann_fitness(S)
		P, P_fit, Pg, Pg_fit = update_fitness(S, S_fit, P, P_fit, Pg, Pg_fit, Np)
		V = update_veloc(S, V, P, Pg, max_iter, current_iter)
		S = S + V
		
	w, v = extract_weights(Pg, m, L, K)
	return w, v

