import numpy as np
from my_utility import get_mse
from ann import forward

# Constantes
C1 = 1.05
C2 = 2.05

def get_r(next, prev):
	r = np.sqrt(6 / (next + prev))
	return r

def init_weights(m, L, K):
	r1 = get_r(L, m)
	r2 = get_r(K, L)

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


def init_pso(x, y, m, L, K, Np, H, f):
	S = init_swarm(m, L, K, Np)
	V = np.zeros(S.shape)
	P = S.copy()
	P_fit = ann_fitness(P, x, y, m, L, K, H, f)	

	i_min = np.argmin(P_fit)
	Pg = P[i_min].copy()
	Pg_fit = P_fit[i_min]

	return S, V, P, Pg, P_fit, Pg_fit


def extract_weights(particle, m, L, K):
	w = particle[:m * L].reshape((L, m))
	v = particle[m * L:].reshape((K, L))
	return w, v


def get_alpha(max_iter, current_iter, a_min=0.1, a_max=0.95):
	a = a_max - ((a_max - a_min) / max_iter) * current_iter
	return a


def ann_fitness(S, x, y, m, L, K, H, f):
	Np = S.shape[0]
	fit = np.array([0.0] * Np)
	for i in range(Np):
		fit[i] = particle_fitness(S[i], x, y, m, L, K, H, f)

	return fit


def particle_fitness(part, x, y, m, L, K, H, f):
	w, v = extract_weights(part, m, L, K)
	y_pred = forward(x, w, v, H, f)

	mse = get_mse(y, y_pred)
	return mse


def update_fitness(S, S_fit, P, P_fit, Pg, Pg_fit, Np):
	i_min = 0
	for i in range(Np):
		if S_fit[i] <= P_fit[i]:
			P[i] = S[i].copy()
			P_fit[i] = S_fit[i]
		
		if S_fit[i] <= S_fit[i_min]:
			i_min = i

	if S_fit[i_min] <= Pg_fit:
		Pg = S[i_min].copy()
		Pg_fit = S_fit[i_min]

	return P, P_fit, Pg, Pg_fit


def update_veloc(S, V, P, Pg, max_iter, current_iter):
	a = get_alpha(max_iter, current_iter)

	D = V.shape[1]
	r1 = np.random.rand(D)
	r2 = np.random.rand(D)

	V_k = a*V + C1*r1*(P - S) + C2*r2*(Pg - S)
	return V_k


def ann_train_pso(x, y, cnf):
	m = cnf['m']
	L = cnf['L']
	K = cnf['K']
	max_iter = cnf['max_iter_pso']
	Np = cnf['Np']
	H = cnf['fun']
	f = cnf['outfun']
	
	S, V, P, Pg, P_fit, Pg_fit = init_pso(x, y, m, L, K, Np, H, f)
	r_clip = get_r(L, m)

	for i in range(max_iter):
		V = update_veloc(S, V, P, Pg, max_iter, i)
		V = np.clip(V, -r_clip, r_clip)
		S = S + V
		S_fit = ann_fitness(S, x, y, m, L, K, H, f)
		P, P_fit, Pg, Pg_fit = update_fitness(S, S_fit, P, P_fit, Pg, Pg_fit, Np)
		
	w, v = extract_weights(Pg, m, L, K)
	return w, v

