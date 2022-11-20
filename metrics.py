import numpy as np

#Funcion encargada de calcular la presición 
def precision(i, cm, k):
	suma = np.sum(cm[i])

	if (suma > 0):
		prec = cm[i][i] / suma
	else:
		prec = 0
    
	return prec

#Funcion encargada de calcular el recall
def recall(j, cm, k):
	suma = np.sum(cm[:, j])
    
	if (suma > 0):
		rec = cm[j][j] / suma
	else: 
		rec = 0
    
	return rec

#Funcion encargada de calcular el fscore
def fscore(j, cm, k):
	numerator = precision(j, cm, k) * recall(j, cm, k)
	denominator = precision(j, cm, k) + recall(j, cm, k)

	if 0 == denominator:
		return 0
  
	fscore = 2 * (numerator / denominator) 
	return fscore

def get_metrics(cm):
	k = cm.shape[0]
	fscore_result = [0] * (k + 1)
  
	for j in range(k):
		fscore_result[j] = fscore(j, cm, k)

	return fscore_result[:-1]

def get_confusion_matrix(y_true, y_pred):
	k, N = y_pred.shape
	cm = np.zeros((k, k), dtype=int)

	for i in range(k):
		for j in range(k):
			for n in range(N):
				if y_pred[j, n] == 1 and y_true[i, n] == 1:
					cm[i, j] += 1

	return cm

