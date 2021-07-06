from dataloader.CommonDataset import get_dataloader
from model.AE import resnet50, Recon_Head
import torch
from torch.optim import Adam
import torch.nn as nn
import torch.nn.functional as F
import torch.optim.lr_scheduler as LR
import torchvision
from torch import distributions as dist
from torchvision.utils import save_image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def test(test_path):
    n = 0
    train_loader = get_dataloader(test_path, batch_size=1)
    model = resnet50().to(device)
    recon_head = Recon_Head().to(device)
    
    
    PATH = 'checkpoint_150.pth'
    checkpoint = torch.load(PATH)
    model.load_state_dict(checkpoint['autoencoder'])
    recon_head.load_state_dict(checkpoint['recon_head'])
    
    recon_loss = torch.nn.L1Loss()

    for epoch in range(1):
        for i, (train_data, _) in enumerate(train_loader):
            train_image = train_data['images'].to(device)
            recon_image = recon_head(model(train_image))
            sum_recon_loss = recon_loss(recon_image, train_image)
            sum_recon_loss += F.mse_loss(recon_image, train_image)
            save_image_recon = recon_image
            
            total_loss = 10e2 * sum_recon_loss
    
            save_image(train_image, 'test/input/in_{}.jpg'.format(n))
            save_image(save_image_recon, 'test/output/out_{}.jpg'.format(n))
            print("Epoch", epoch, "Total_loss:", "%.2f" % total_loss.item())

            n += 1
            

