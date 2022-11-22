import numpy as np
from my_utility import *
from metrics import get_metrics, get_confusion_matrix
from pso import *

# Se carga la configuracion de PSO y BP, junto con cargar datos dtrn y dtst 
cnf = readConfigAndData()

# Se obtienen las variables a utilizar
X_test = cnf['X_test']
y_test = cnf['y_test']
fun = cnf['fun']
outfun = cnf['outfun']

# Se cargan los pesos de la capa de salida y los de la capa oculta
pesos = np.load("pesos.npz")
w = pesos['arr_0']
v = pesos['arr_1']

# Se obtiene el vector con las predicciones
y_predict = get_one_hot(np.argmin(forward(X_test, w, v, fun, outfun), axis=0)).T

# Se genera la matriz de confusion y
cm = get_confusion_matrix(y_test, y_predict)

# Se guarda la matriz de confusión 
np.savetxt("cmatriz.csv", np.array(cm), fmt='%i')

# Se obtienen los fscores y su media
f1 = get_metrics(cm)

# Se guardan los f-scores 
np.savetxt("fscores.csv", np.array(f1))

