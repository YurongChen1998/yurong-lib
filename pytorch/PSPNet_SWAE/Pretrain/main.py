from JigsawDataset import get_dataloader, JigsawDataset, get_transformers
from model import resnet50, Aux_Head, Recon_Head, compute_swd, compute_lower
import torch
from torch.optim import Adam
import torch.nn as nn
import torch.nn.functional as F
import torch.optim.lr_scheduler as LR
import torchvision
from torchvision.utils import save_image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train():
    n = 0
    
    img_transformer, tile_transformer = get_transformers()
    dataset = JigsawDataset(img_transformer, tile_transformer, jig_classes=100, bias_whole_image=0.7)
    
    train_loader = get_dataloader()
    model = resnet50().to(device)
    aux_head = Aux_Head().to(device)
    recon_head = Recon_Head().to(device)
    #contra_loss = compute_contra_loss().to(device)
    
    #PATH = 'checkpoint_25.pth'
    #checkpoint = torch.load(PATH)
    #model.load_state_dict(checkpoint['autoencoder'])
    #aux_head.load_state_dict(checkpoint['aux_head'])
    #recon_head.load_state_dict(checkpoint['recon_head'])

    all_parameter = list(model.parameters()) + list(aux_head.parameters()) + list(recon_head.parameters())
    optimizer = Adam(all_parameter, lr=0.001)
    scheduler = LR.StepLR(optimizer, step_size=30)
    aux_class_loss_func = nn.CrossEntropyLoss()
    recon_loss = torch.nn.L1Loss()

    for epoch in range(501):
        for i, (train_data, _) in enumerate(train_loader):
            optimizer.zero_grad()

            train_image = train_data['images'].to(device)
            intput_image = torchvision.utils.make_grid(train_image.squeeze(), 3, padding=0)
            target = train_data['aux_labels'].to(device)
            tem_feature = []
            recon_image = []

            recon_error = 0.0
            swd = 0.0
            for i in range(train_image.size(1)):
                ip = train_image[:,i,:,:,:]

                aux_op, latent, tsne = model(ip)
                swd += compute_swd(latent)
                print(aux_op.shape)
                tem_feature.append(aux_op)
                recon_image.append(latent)
                
            #c_loss = contra_loss(tem_feature)
            
            tem_feature = torch.stack(tem_feature, 1).view(1, -1)
            final_out = aux_head(tem_feature)
            aux_loss = aux_class_loss_func(final_out, target)
            
            recon_image = torch.stack(recon_image, 1).squeeze()
            recon_image = torchvision.utils.make_grid(recon_image, 3, padding=0).unsqueeze(0)     
            recon_image = recon_head(recon_image)
            recon_image = recon_image.squeeze()
            sum_recon_loss = recon_loss(recon_image, intput_image)
            sum_recon_loss += F.mse_loss(recon_image, intput_image)

            total_loss = 10e2 * aux_loss + 10e2 * sum_recon_loss + 10 * swd

            if n % 100 == 0:       
                save_image(intput_image, 'images/input/in_{}.jpg'.format(n))
                save_image(recon_image, 'images/output/out_{}.jpg'.format(n))
                print("Epoch", epoch, "Total_loss:", "%.2f" % total_loss.item(), "Aux_loss:", "%.2f" % aux_loss.item(),"Recon_sum_loss:", "%.2f" % sum_recon_loss.item(), "swd_loss:", "%.2f" % swd.item())

            total_loss.backward()
            optimizer.step()
            n += 1
        if epoch % 5 == 0:
            torch.save({
                "autoencoder": model.state_dict(),
                "aux_head": aux_head.state_dict(),
                "recon_head": recon_head.state_dict(),
            }, 'checkpoint_{}.pth'.format(epoch))

train()
