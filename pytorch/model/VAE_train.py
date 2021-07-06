from dataloader.CommonDataset import get_dataloader
from model.VAE import resnet50, Recon_Head
import torch
from torch.optim import Adam
import torch.nn as nn
import torch.nn.functional as F
import torch.optim.lr_scheduler as LR
import torchvision
from torch import distributions as dist
from torchvision.utils import save_image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train():
    n = 0
    path = './images/good'
    train_loader = get_dataloader(path, batch_size=1)
    model = resnet50().to(device)
    recon_head = Recon_Head().to(device)
    
    '''
    PATH = 'checkpoint_750.pth'
    checkpoint = torch.load(PATH)
    model.load_state_dict(checkpoint['autoencoder'])
    aux_head.load_state_dict(checkpoint['aux_head'])
    recon_head.load_state_dict(checkpoint['recon_head'])
    '''
    
    all_parameter = list(model.parameters()) + list(recon_head.parameters())
    optimizer = Adam(all_parameter, lr=0.001)
    scheduler = LR.StepLR(optimizer, step_size=30)
    recon_loss = torch.nn.L1Loss()

    for epoch in range(2001):
        for i, (train_data, _) in enumerate(train_loader):
            optimizer.zero_grad()

            train_image = train_data['images'].to(device)
            mu, log_var = model(train_image)
            recon_image = recon_head(mu, log_var)
            sum_recon_loss = recon_loss(recon_image, train_image)
            sum_recon_loss += F.mse_loss(recon_image, train_image)
            kld_loss = torch.mean(torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim = 1), dim = 0))
            save_image_recon = recon_image
            
            total_loss = 10e2 * sum_recon_loss + 0.1*kld_loss

            if n % 100 == 0:       
                save_image(train_image, 'images/VAE_input/in_{}.jpg'.format(n))
                save_image(save_image_recon, 'images/VAE_output/out_{}.jpg'.format(n))
                print("Epoch", epoch, "Total_loss:", "%.2f" % total_loss.item(), "KL_loss:", "%.2f" % kld_loss)

            total_loss.backward()
            optimizer.step()
            n += 1
            
        if epoch % 50 == 0:
            torch.save({
                "autoencoder": model.state_dict(),
                "recon_head": recon_head.state_dict(),
            }, 'checkpoint_VAE_{}.pth'.format(epoch))

train()
