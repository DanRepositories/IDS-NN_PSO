from sv_utility import *
import numpy as np

CONFIG_FILE = 'cnf_sv.csv'
TRAIN_DATA_FILE = 'KDDTrain.txt'
TEST_DATA_FILE = 'KDDTest.txt'

N_train, N_test, k, d, classes_to_consider = readConfigFile(CONFIG_FILE)

# ------- Carga de los datos ---------
print('Cargando los datos de entrenamiento desde:', TRAIN_DATA_FILE)
raw_train_data = np.loadtxt(TRAIN_DATA_FILE, dtype=str, delimiter=',')

print('Cargando los datos de prueba desde:', TEST_DATA_FILE)
raw_test_data = np.loadtxt(TEST_DATA_FILE, dtype=str, delimiter=',')

# Se extraen todos los valores posibles de las variables categoricas
cat_1 = extract_categories(raw_train_data, raw_test_data, 1)
cat_2 = extract_categories(raw_train_data, raw_test_data, 2)
cat_3 = extract_categories(raw_train_data, raw_test_data, 3)

# -------- Preprocesado de los datos --------
X_train, y_train = preprocess(raw_train_data, classes_to_consider, cat_1, cat_2, cat_3)
print('\nNormalizando la data de entrenamiento')
X_train, y_train = sample_data(X_train, y_train, N_train)
X_train = normalize(X_train)
D, N = X_train.shape

# -------- Calculo de la ganancia de informacion --------
print('Calculando la ganancia de la informacion')
# Se calcula la entropia cruzada de todas las caracteristicas
Ix = int(np.ceil(np.sqrt(N)))
cross_entropy = calculate_info_gain(X_train, y_train, Ix, D, N)

# Se seleccionan los indices de los top-k atributos de la base de datos
index_top_k = [e[0] for e in cross_entropy[:k]]
print('Extrayendo las caracteristicas mas relevantes')
X_k = X_train[index_top_k]
np.savetxt('index.csv', index_top_k, delimiter=',', fmt='%d')

# -------- Calculo de SVD --------
print('Calculando la matrix V')
X_d = substract_mean(X_k, k)
v = calculate_V_matrix(X_d, N)
print('Filtrando la data de entrenamiento con la matriz V')
X_d = filter_data(X_d, v, d)
np.savetxt('filter_v.csv', v[:, :d], delimiter=',')


print('\nGuardando los datos y etiquetas de entrenamiento')
np.savetxt('dtrn.csv', X_d.T, delimiter=',')
np.savetxt('etrn.csv', y_train, delimiter=',', fmt='%d')


# ------- Procesado de los datos de test -----------
X_test, y_test = preprocess(raw_test_data, classes_to_consider, cat_1, cat_2, cat_3)
print('\nNormalizando la data de test')
X_test, y_test = sample_data(X_test, y_test, N_test)
X_test = normalize(X_test)
print('Extrayendo las caracteristicas mas relevantes')
X_test = X_test[index_top_k]
print('Filtrando la data con la matriz V')
X_test = substract_mean(X_test, k)
X_test = filter_data(X_test, v, d)

print('\nGuardando los datos y etiquetas de prueba')
np.savetxt('dtst.csv', X_test.T, delimiter=',')
np.savetxt('etst.csv', y_test, delimiter=',', fmt='%d')

