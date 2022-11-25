import numpy as np
from ann import *
from my_utility import *
from pso import *
from pso import ann_train_pso

# Se carga la configuracion de PSO y BP, junto con cargar datos dtrn y dtst 
cnf = readConfigAndData()

#Se realiza PSO
max_iter = 10
for i in range(max_iter):
    print(f'Ejecutando iteracion de train numero', i)
    w, v, pso_MSE = ann_train_pso(cnf)
    #Se guarda la lista de los MSE del PSO por iteración
    file_costo = f'costo_pso{i}.csv'
    np.savetxt(file_costo, np.array(pso_MSE))

    # Se entrena la red
    w, v, ann_MSE = ann_train(w, v, cnf)
    #Se guarda la lista de los MSE de la ANN por iteración
    file_costo = f'costo_gd{i}.csv'
    np.savetxt(file_costo, np.array(ann_MSE))

    #Se guardan los pesos de la capa oculta(w) y de la capa de salida(v)
    file_pesos = f'pesos{i}.npz'
    np.savez(file_pesos, w, v)
