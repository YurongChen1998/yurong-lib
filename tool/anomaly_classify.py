import os
import numpy as np

from sklearn.metrics import roc_curve
from sklearn.metrics import auc

from functools import reduce
def str2float(f):
    s = f.split('.')
    s1 = list(s[0])
    s2 = list(s[1])
    s2.reverse()
    s2.append('0')
    to_digit = lambda x: int(x)
    c10 = lambda x,y: x*10+y
    d10 = lambda x,y: x*0.1 + y
    ii = reduce(c10, map(to_digit, s1))
    ff = reduce(d10, map(to_digit, s2))



path = 'image_level_classification'
num_txt = os.listdir(path)

all_list = []
all_label = []

for i in range(len(num_txt)):
    txt_path = os.path.join(path, num_txt[i])
    print(">>>>>>>>>>>>>>>>>>>>>>>>>>>", num_txt[i], ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
    file_txt = open(txt_path, 'r')
    for idx_line in file_txt: 
        all_list.append(idx_line)
        if num_txt[i] == "good.txt":
            all_label.append(0)
        else: 
            all_label.append(1)

all_list_ = []
for idx in range(len(all_list)):
    all_list[idx] = all_list[idx].replace('\n','')
    all_list[idx] = all_list[idx].replace(' ','')

    all_list_.append(all_list[idx].split(','))

all_num = np.ones([len(all_list_), 10])
for idx in range(len(all_list_)):
    for patch_idx in range(len(all_list_[idx])):
        all_num[idx][patch_idx] = float(all_list_[idx][patch_idx])
all_num = all_num / all_num.max(axis=0)

print(all_num)

predict = []
for idx in range(len(all_num)):
    if all_num[idx][0] <= 0.2:
        num_good_patch = 0
        for patch_idx in range(1, 10): 
            if all_num[idx][patch_idx] <= 0.8:
                num_good_patch += 1
        if num_good_patch == 9:
            predict.append(0)
            #continue
        else:
            predict.append(1)
    else:
        predict.append(1) 
  
print(len(predict), len(all_label))
print(predict)
print(all_label)


fpr, tpr, thersholds = roc_curve(all_label, predict, pos_label=1)
roc_auc = auc(fpr, tpr)
print(roc_auc)
