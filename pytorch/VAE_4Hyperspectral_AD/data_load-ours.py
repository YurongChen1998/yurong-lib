import scipy.io as sio
import torch.utils.data as data
import numpy as np
import torch
from dataset_utils import *
from sklearn.preprocessing import normalize

w = 200
h = 200
b = 101
path = 'vaccine_anomaly.mat'

########################################################################
class Mat_DataLoader(data.Dataset):
    def __init__(self):
        H_data = path
        H_data = sio.loadmat(H_data)
        label = H_data['map']  # 100, 100
        self.label = np.reshape(label, (w * h, 1))
        data_ = H_data['data']  # 100, 100, 205
        self.data = np.reshape(data_, (w * h, b))
        self.data = normalize(self.data, axis=1, norm='max')

        self.normal_data = []
        self.abnormal_data = []
        for idx in range(w * h):
            if self.label[idx] == 0:
                self.normal_data.append(self.data[idx, :])
            else:
                self.abnormal_data.append(self.data[idx, :])

        self.normal_data = np.array(self.normal_data)
        self.abnormal_data = np.array(self.abnormal_data)
        print("normal_data_number:", len(self.normal_data),
              "abnormal_data_number:", len(self.abnormal_data))

    def __getitem__(self, index):
        pixel = np.array(self.normal_data[index,:])
        sample = {'data': pixel}
        return sample

    def __len__(self):
        #print("Total Load Pixel >>>>>>", len(self.normal_data))
        return len(self.normal_data)
########################################################################


########################################################################
class Test_DataLoader(data.Dataset):
    def __init__(self):
        H_data = path
        H_data = sio.loadmat(H_data)
        label = H_data['map']  # 100, 100
        self.label = np.reshape(label, (w * h, 1))
        data_ = H_data['data']  # 100, 100, 205
        self.data = np.reshape(data_, (w * h, b))
        self.data = normalize(self.data, axis=1, norm='max')

        self.normal_data = []
        self.abnormal_data = []
        for idx in range(w * h):
            if self.label[idx] == 0:
                self.normal_data.append(self.data[idx, :])
            else:
                self.abnormal_data.append(self.data[idx, :])

        self.normal_data = np.array(self.normal_data)
        self.abnormal_data = np.array(self.abnormal_data)

    def __getitem__(self, index):
        pixel = np.array(self.abnormal_data[index,:])
        label = np.array(self.label[index,:])
        sample = {'data': pixel, 'label': label}
        return sample

    def __len__(self):
        #print("Total Load Pixel >>>>>>", len(self.normal_data))
        return len(self.abnormal_data)
########################################################################

class ALL_DataLoader(data.Dataset):
    def __init__(self):
        H_data = path
        H_data = sio.loadmat(H_data)
        label = H_data['map']  # 100, 100
        self.label = np.reshape(label, (w * h, 1))
        data_ = H_data['data']  # 100, 100, 205
        self.data = np.reshape(data_, (w * h, b))
        self.data = normalize(self.data, axis=1, norm='max')

    def __getitem__(self, index):
        pixel = np.array(self.data[index,:])
        label = np.array(self.label[index,:])
        sample = {'data': pixel, 'label': label}
        return sample

    def __len__(self):
        #print("Total Load Pixel >>>>>>", len(self.normal_data))
        return len(self.data)

def test_dataloader():
    datasets = []
    #train_dataset = Mat_DataLoader()
    train_dataset = Test_DataLoader()
    datasets.append(train_dataset)
    dataset = ConcatDataset(datasets)
    loader = torch.utils.data.DataLoader(dataset, batch_size=128, shuffle=True, num_workers=0, pin_memory=True,
                                         drop_last=False)
    return loader
    
def get_dataloader():
    datasets = []
    train_dataset = Mat_DataLoader()
    #train_dataset = Test_DataLoader()
    
    train_set, val_set = torch.utils.data.random_split(train_dataset, 
        [10000, len(train_dataset) - 10000])
   
    datasets.append(train_set)
    dataset = ConcatDataset(datasets)
    loader = torch.utils.data.DataLoader(dataset, batch_size=512, shuffle=True, num_workers=0, pin_memory=True,
                                         drop_last=False)
                                         
    val_datasets = []
    val_datasets.append(val_set)
    val_datasets = ConcatDataset(val_datasets)
    val_loader = torch.utils.data.DataLoader(val_datasets, batch_size=512, shuffle=True, num_workers=0, pin_memory=True,
                                         drop_last=False)
    return loader, val_loader

def all_dataloader():
    datasets = []
    #train_dataset = Mat_DataLoader()
    train_dataset = ALL_DataLoader()
    datasets.append(train_dataset)
    dataset = ConcatDataset(datasets)
    loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True, num_workers=0, pin_memory=True,
                                         drop_last=False)
    return loader


'''
loader, val_loader = get_dataloader()
for i, sample in enumerate(loader):
    train_data = sample[0]
    print(train_data['data'].shape)
print(">>>>>>>>>>>>>>>>>>>>>>>>>>", i)
print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>")
for i, sample in enumerate(val_loader):
    train_data = sample[0]
    print(train_data['data'].shape)
print(">>>>>>>>>>>>>>>>>>>>>>>>>>", i)
'''
