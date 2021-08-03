# writen by Yurong Chen 2021-08-03

from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
import numpy as np
import heapq
from scipy import stats


def KNN(X_train, y_train, X_test, k):
    res = []
    for test_idx in range(len(X_test)):
        temp_test_data = np.expand_dims(X_test[test_idx], axis=0).T
        temp_distance = np.dot(X_train, temp_test_data)
        k_neigh = heapq.nlargest(k, range(len(temp_distance)), temp_distance.take)
        k_neigh = [y_train[_] for _ in k_neigh]
        res.append(stats.mode(k_neigh)[0][0])
    return res

digits = load_digits() # digtis.data [1797, 64] (image shage: 8x8)
digits_data = digits.data
digits_label = digits.target

# randomly select 1000 samples as the databse
X_train, X_test, y_train, y_test = train_test_split(digits_data, digits_label, test_size=0.3)
res = KNN(X_train, y_train, X_test, 20)

true_positve = 0
for test_idx in range(len(y_test)):
    if y_test[test_idx] == res[test_idx]:
        true_positve += 1
print(">>>>>>  acc:", true_positve/len(y_test))