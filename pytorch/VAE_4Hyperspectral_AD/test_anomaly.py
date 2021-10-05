import os
from sklearn import metrics

anomaly_path = './Error_Ab.txt'
nor_path = './Error_Nor.txt'

label = []
ab_error_list = []
with open(anomaly_path, 'r') as file_ab:
    lines = file_ab.readlines()
    for i in range(len(lines)):
        ab_error_list.append(100 * float(lines[i]))
        label.append(1)
        
nor_error_list = []
with open(nor_path, 'r') as file_nor:
    lines = file_nor.readlines()
    for i in range(len(lines)):
        nor_error_list.append(100 * float(lines[i]))
        label.append(0)
        
all_list = []
all_list += ab_error_list
all_list += nor_error_list

pred = []
for idx in range(len(all_list)):
    if all_list[idx] >= 2.5:
        pred.append(1)
    else:
        pred.append(0)
        
print("Kappa  ", metrics.cohen_kappa_score(label, pred))
print("Recall ", metrics.recall_score(label, pred))
print("Acc >> ", metrics.accuracy_score(label, pred))
print("Confuse", metrics.confusion_matrix(label, pred))
