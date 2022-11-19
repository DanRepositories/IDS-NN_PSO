import numpy as np
from metrics import get_metrics, get_confusion_matrix

f1_scores = []
y_predict = np.array([[1,0,0], [0,0,1],[0,0,1],[1,0,0], [0,1,0],[0,0,1]]).T
y_test = np.array([[1,0,0], [0,0,1],[0,1,0],[1,0,0], [1,0,0], [0,1,0]]).T


