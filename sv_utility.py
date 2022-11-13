import numpy as np

class_1 = ['normal']
class_2 = ['neptune', 'teardrop', 'smurf', 'pod', 'back', 'land', 'apache2', 'processtable', 'mailbomb', 'udpstorm']
class_3 = ['ipsweep', 'portsweep', 'nmap', 'satan', 'saint', 'mscan']

def readConfigFile(filename):
	config_data = np.loadtxt(filename, dtype=int)
	N_train, N_test = config_data[0], config_data[1]
	k, d = config_data[2], config_data[3]
	classes_to_consider = config_data[4:].tolist()
	return N_train, N_test, k, d, classes_to_consider

def convert_class(class_name, class_values, array):
	array[np.isin(array, class_values)] = class_name
	return array

def extract_categories(x_train, x_test, col):
	cat = np.union1d(np.unique(x_train[:, col]), np.unique(x_test[:, col]))
	return cat

def convert_categorical(x, categorical_list):
	x_converted = x.copy()
	for i in range(len(categorical_list)):
		x_converted[x_converted == categorical_list[i]] = i

	return x_converted.astype(int)

def delete_extra_classes(X, labels, class_values):
	index_extra_classes = np.where(np.isin(labels, class_values, invert=True))

	labels = np.delete(labels, index_extra_classes)
	X = np.delete(X, index_extra_classes, axis=0)
	return X, labels

def preprocess(raw_data, classes_to_consider, cat_1, cat_2, cat_3):
	global class_1
	global class_2
	global class_3

	raw_data = np.delete(raw_data, -1, axis=1)	# Se elimina la ultima columna

	labels = raw_data[:, -1]
	categoricals = raw_data[:, 1:4]

	# Se convierten las variables categoricas
	categorical_1 = convert_categorical(categoricals[:, 0], cat_1)
	categorical_2 = convert_categorical(categoricals[:, 1], cat_2)
	categorical_3 = convert_categorical(categoricals[:, 2], cat_3)

	# Se eliminan las variables categoricas y las etiquetas de la clase
	preprocess_data = np.delete(raw_data, [1,2,3,41], 1).astype(float)

	# Se agregan las variables categoricas convertidas
	preprocess_data = np.insert(preprocess_data, 1, categorical_1, axis=1)
	preprocess_data = np.insert(preprocess_data, 2, categorical_2, axis=1)
	preprocess_data = np.insert(preprocess_data, 3, categorical_3, axis=1)

	# Se dejan solo las clases que establecio el usuario en el archivo de config
	if 0 == classes_to_consider[0]:
		class_1 = ['']
	elif 0 == classes_to_consider[1]:
		class_2 = ['']
	elif 0 == classes_to_consider[2]:
		class_3 = ['']

	# Se eliminan las muestras con las clases extras que no se consideran
	preprocess_data, labels = delete_extra_classes(preprocess_data, labels, class_1 + class_2 + class_3)

	# Se mapean las etiquetas a las 3 clases correspondientes
	labels = convert_class('1', class_1, labels)
	labels = convert_class('2', class_2, labels)
	labels = convert_class('3', class_3, labels)

	# Se transforman las etiqueras a su valor entero
	labels = labels.astype(int)

	if 0 == classes_to_consider[1]:
		labels[labels == 3] = 2

	return preprocess_data.T, labels

def normalize(X, a=0.01, b=0.99):
	D = X.shape[0]
	for i in range(D):
		X[i] = normalize_var(X[i], a, b)

	return X

def normalize_var(x, a=0.01, b=0.99):
	x_min = x.min()
	x_max = x.max()
	if x_max > x_min:
		x = ((x - x_min) / (x_max - x_min)) * (b - a) + a
	else:
		x = a
	return x

# TODO: Revisar funcion
def sample_data(X, y, N):
	X_t = X.copy().T
	index_class_1 = np.where(y == 1)
	index_class_2 = np.where(y == 2)
	index_class_3 = np.where(y == 3)
	
	X_sample = X_t[index_class_1][:N]
	y_sample = y[index_class_1][:N]

	X_sample = np.concatenate((X_sample, X_t[index_class_2][:N]))
	y_sample = np.concatenate((y_sample, y[index_class_2][:N]))

	X_sample = np.concatenate((X_sample, X_t[index_class_3][:N]))
	y_sample = np.concatenate((y_sample, y[index_class_3][:N]))
	return X_sample.T, y_sample

# Obtiene la cantidad de muestras de cada clase para el array 'y' de etiquetas ingresado
def get_class_counts(y):
	_, di = np.unique(y, return_counts=True)
	return di

# I(Y)
def calculate_entropy(y, N):
	class_counts = get_class_counts(y)
	p = class_counts / N
	entropy = -1 * np.sum(p * np.log2(p))
	return entropy

# E(x)
def calculate_cross_entropy(x, y, N, Ix):
	interval_range, x_min, x_max = calculate_range(x, Ix)

	cross_entropy = 0
	for i in range(Ix):
		lower_bound = x_min + interval_range * i
		upper_bound = lower_bound + interval_range

		# Se obtienen los indices de las muestras dentro del intervalo actual
		index_inside_interval = np.where(np.logical_and(x >= lower_bound, x < upper_bound))

		# Se obtienen las clases de las muestras seleccionadas
		yy = y[index_inside_interval]

		entropy = calculate_entropy(yy, N)
		weight_class = yy.shape[0] / N
		cross_entropy += weight_class * entropy

	return cross_entropy

def calculate_range(x, Ix, x_min=0.01, x_max=1):
	range = (x_max - x_min) / Ix
	return range, x_min, x_max

def calculate_info_gain(X, y, Ix, D, N):
	I_y = calculate_entropy(y, N)
	cross_entropy = [0] * D
	for i in range(D):
		E_x = calculate_cross_entropy(X[i], y, N, Ix)
		cross_entropy[i] = (i, I_y - E_x)

	cross_entropy.sort(reverse=True, key=lambda e: e[1])
	return cross_entropy

def substract_mean(X, k):
	for i in range(k):
		mean_row = X[i].mean()
		X[i] = X[i] - mean_row

	return X

def calculate_V_matrix(X_d, N):
	Y_d = X_d.T / np.sqrt(N - 1)
	u, s, v = np.linalg.svd(Y_d)
	return v

def filter_data(X, V, d):
	X_d = np.matmul(V[:, :d].T, X)
	X_d = normalize(X_d)
	return X_d

