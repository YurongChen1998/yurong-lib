import torchvision.transforms as transforms
import torch.utils.data as data
import torch
from dataset_utils import *
import os
from torchvision.utils import save_image
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

class Dataset(data.Dataset):
    def __init__(self, test_path, img_transformer):
        self.data_path = test_path
        self._image_transformer = img_transformer
        
        all_names = []
        for each_image in os.listdir(self.data_path):
            each_image_path = os.path.join(self.data_path, each_image)
            all_names.append(each_image_path)
        self.names = all_names

    def get_image(self, index):
        framename = self.names[index]
        img = Image.open(framename).convert('RGB')
        return self._image_transformer(img)

    def __getitem__(self, index):
        img = self.get_image(index)
        sample = {'images': img}
        return sample

    def __len__(self):
        print("Total Load Number of Image >>>>>>", len(self.names))
        return len(self.names)

def get_transformers():
    size = [300, 300]
    scale = [1.0, 1.0]
    #img_transormer = [transforms.Resize((int(size[0]), int(size[1]))),
    #            transforms.ColorJitter(0.2, 0.2, 0.2, 0.2)]
    img_transormer = [transforms.Resize((int(size[0]), int(size[1])))]
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    return transforms.Compose(img_transormer)

def get_dataloader(test_path, batch_size):
    datasets = []
    img_transformer = get_transformers()
    train_dataset = Dataset(test_path, img_transformer)
    datasets.append(train_dataset)
    dataset = ConcatDataset(datasets)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True, drop_last=True)
    return loader

if __name__ == '__main__':
    path = r'fig'
    loader = get_dataloader(path, batch_size=1)
    for i, (train_data, _) in enumerate(loader):
        train_image = train_data['images']
        save_image(train_image, 'fig/out.jpg')