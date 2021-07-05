import torchvision.transforms as transforms
import torch.utils.data as data
import numpy as np
from torch.utils.data import Dataset
import torch
import torchvision
from PIL import Image
from random import sample, random
from os.path import join, dirname
from dataset_utils import *
import os
from torchvision.utils import save_image
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

class JigsawDataset(data.Dataset):
    def __init__(self, test_path, img_transformer, tile_transformer, jig_classes=1000, bias_whole_image=0.7):
        self.data_path = test_path
        self.permutations = self.__retrieve_permutations(jig_classes)
        self.grid_size = 3
        self.bias_whole_image = bias_whole_image
        self._image_transformer = img_transformer
        self._augment_tile = tile_transformer
        self.returnFunc = self.make_grid
        
        all_names = []
        for each_image in os.listdir(self.data_path):
            each_image_path = os.path.join(self.data_path, each_image)
            all_names.append(each_image_path)
        self.names = all_names
        
    def get_tile(self, img, n):
        w = float(img.size[0]) / self.grid_size
        h = float(img.size[1]) / self.grid_size
        y = int(n / self.grid_size)
        x = n % self.grid_size
        tile = img.crop([x * w, y * h, (x + 1) * w, (y + 1) * h])
        tile = self._augment_tile(tile)
        return tile

    def get_image(self, index):
        framename = self.names[index]
        img = Image.open(framename).convert('RGB')
        return self._image_transformer(img)

    def __getitem__(self, index):
        img = self.get_image(index)
        n_grids = self.grid_size ** 2
        tiles = [None] * n_grids
        for n in range(n_grids):
            tiles[n] = self.get_tile(img, n)

        order = np.random.randint(len(self.permutations) + 1)  # added 1 for class 0: unsorted
        if self.bias_whole_image:
            if self.bias_whole_image > random():
                order = 0
        if order == 0:
            data = tiles
        else:
            data = [tiles[self.permutations[order - 1][t]] for t in range(n_grids)]

        data = torch.stack(data, 0)
        sample = {'images': data,
                'aux_labels': int(order),
                'label': self.permutations[int(order) - 1]}
        return sample
        
    def get_single_image(self, image):
        img = transforms.ToPILImage()(image).convert('RGB')
        n_grids = self.grid_size ** 2
        tiles = [None] * n_grids
        for n in range(n_grids):
            tiles[n] = self.get_tile(img, n)

        order = np.random.randint(len(self.permutations) + 1)  # added 1 for class 0: unsorted
        if self.bias_whole_image:
            if self.bias_whole_image > random():
                order = 0
        if order == 0:
            data = tiles
        else:
            data = [tiles[self.permutations[order - 1][t]] for t in range(n_grids)]

        data = torch.stack(data, 0).cuda()
        return data
    def __len__(self):
        print("Total Load Number of Image >>>>>>", len(self.names))
        return len(self.names)
        
    def __retrieve_permutations(self, classes):
        all_perm = np.load('permutations_%d.npy' % (jig_classes))
        # from range [1,9] to [0,8]
        if all_perm.min() == 1:
            all_perm = all_perm - 1
        return all_perm

    def make_grid(self, x):
        return torchvision.utils.make_grid(x, 1, padding=0)

def get_transformers():
    size = [300, 300]
    scale = [1.0, 1.0]
    #img_transormer = [transforms.Resize((int(size[0]), int(size[1]))),
    #            transforms.ColorJitter(0.2, 0.2, 0.2, 0.2)]
    img_transormer = [transforms.Resize((int(size[0]), int(size[1])))]
    tile_tr = []
    tile_tr.append(transforms.RandomGrayscale(0.1))
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    tile_tr = tile_tr + [transforms.ToTensor()]
    return transforms.Compose(img_transormer), transforms.Compose(tile_tr)

def get_dataloader(test_path, jig_classes, batch_size):
    datasets = []
    img_transformer, tile_transformer = get_transformers()
    train_dataset = JigsawDataset(test_path, img_transformer, tile_transformer, jig_classes, bias_whole_image=0.7)
    datasets.append(train_dataset)
    dataset = ConcatDataset(datasets)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True, drop_last=True)
    return loader

if __name__ == '__main__':
    path = r'fig'
    loader = get_dataloader(path)
    for i, (train_data, _) in enumerate(loader):
        train_image = train_data['images']
        output = []
        for n in range(train_image.size(1)):
            ip = train_image[:,n,:,:,:]
        #intput_image = torchvision.utils.make_grid(train_image.squeeze(), 3, padding=0)
            save_image(ip, 'fig/output2_{}.jpg'.format(n))
            output.append(ip)
        recon_image = torch.stack(output, 1).squeeze()
        recon_image = torchvision.utils.make_grid(recon_image, 3, padding=0).unsqueeze(0) 
        save_image(recon_image, 'fig/out.jpg')