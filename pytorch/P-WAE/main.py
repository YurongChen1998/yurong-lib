from JigsawDataset import get_dataloader, JigsawDataset, get_transformers
from model import resnet50, Aux_Head, Recon_Head, compute_swd, compute_lower, compute_swd_final, compute_hingle
import torch
from torch.optim import Adam
import torch.nn as nn
import torch.nn.functional as F
import torch.optim.lr_scheduler as LR
import torchvision
import torch.backends.cudnn as cudnn
#from torchvision.utils import save_image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train():
    n = 0
    cudnn.benchmark = True
    #img_transformer, tile_transformer = get_transformers()
    #dataset = JigsawDataset(img_transformer, tile_transformer, jig_classes=100, bias_whole_image=0.7)
    
    train_loader = get_dataloader()
    model = resnet50().to(device)
    aux_head = Aux_Head().to(device)
    recon_head = Recon_Head().to(device)
    
    PATH = 'checkpoint.pth'
    checkpoint = torch.load(PATH)
    model.load_state_dict(checkpoint['autoencoder'])
    aux_head.load_state_dict(checkpoint['aux_head'])
    recon_head.load_state_dict(checkpoint['recon_head'])

    encoder_parameter = list(model.parameters()) + list(aux_head.parameters()) + list(recon_head.parameters())
    #encoder_parameter = model.parameters()
    #decoder_parameter = list(recon_head.parameters())
    
    optimizer = Adam(encoder_parameter, lr=0.001)
    #optimizer_decodr = Adam(decoder_parameter, lr=0.001)
    
    scheduler = LR.StepLR(optimizer, step_size=30)
    #scheduler_decodr = LR.StepLR(optimizer_decodr, step_size=30)
    
    aux_class_loss_func = nn.CrossEntropyLoss()
    recon_loss = torch.nn.L1Loss()

    for epoch in range(501):
        for i, (train_data, _) in enumerate(train_loader):
            optimizer.zero_grad()
            #optimizer_decodr.zero_grad()

            train_image = train_data['images'].to(device)
            intput_image = torchvision.utils.make_grid(train_image.squeeze(), 3, padding=0)
            target = train_data['aux_labels'].to(device)
            label = train_data['label'][0].to(device)
            tem_feature = []
            recon_image = []
            
            tmp_latent = []
            tmp_piror = []

            recon_error = 0.0
            for i in range(train_image.size(1)):
                ip = train_image[:,i,:,:,:]

                aux_op, latent, tsne = model(ip)
                tmp_latent += compute_swd(latent, label[i])[0]
                tmp_piror += compute_swd(latent, label[i])[1]
                
                tem_feature.append(aux_op)
                recon_image.append(latent)
                
            swd = compute_swd_final(torch.stack(tmp_latent, 1).view(1, -1), torch.stack(tmp_piror, 1).view(1, -1))
            
            tem_feature = torch.stack(tem_feature, 1).view(1, -1)
            final_out = aux_head(tem_feature)
            aux_loss = aux_class_loss_func(final_out, target)
            latent_list = recon_image
            
            recon_image = torch.stack(recon_image, 1).squeeze()
            recon_image = torchvision.utils.make_grid(recon_image, 3, padding=0).unsqueeze(0)     
            recon_image = recon_head(recon_image)
            recon_image = recon_image.squeeze()
            recon_image.clone()
            sum_recon_loss = recon_loss(recon_image, intput_image)
            sum_recon_loss += F.mse_loss(recon_image, intput_image)
            
            #save_image_recon = recon_image
            
            '''
            recon_swd = 0.0
            recon_image = dataset.get_single_image(recon_image.detach().cpu())
            recon_image = recon_image.cuda().unsqueeze(0)
            
            recon_tmp_latent = []
            recon_tmp_piror = []
            
            for i in range(recon_image.size(1)):
                recon_ip = recon_image[:,i,:,:,:]
                _, recon_latent, _ = model(recon_ip)
                
                recon_tmp_latent += compute_swd(recon_latent, label[i])[0]
                recon_tmp_piror += compute_swd(recon_latent, label[i])[1]
 
            #hingle_loss = compute_hingle(torch.stack(recon_tmp_latent, 1).view(1, -1), torch.stack(recon_tmp_piror, 1).view(1, -1)) 
            recon_swd =  compute_swd_final(torch.stack(recon_tmp_latent, 1).view(1, -1), torch.stack(recon_tmp_piror, 1).view(1, -1)) 
            #recon_swd = torch.clamp(10.0 - recon_swd, min=0.0)
            '''
            
            total_loss = 10e2 * aux_loss + 10e2 * sum_recon_loss + 10 * swd
            #decoder_loss = 10e2 * sum_recon_loss - recon_swd


            if n % 100 == 0:       
                #save_image(intput_image, 'images/input/in_{}.jpg'.format(n))
                #save_image(save_image_recon, 'images/output/out_{}.jpg'.format(n))
                print("Epoch", epoch, "Total_loss:", "%.2f" % total_loss.item(), "Aux_loss:", "%.2f" % aux_loss.item(),"Recon_sum_loss:", "%.2f" % sum_recon_loss.item(), "swd_loss:", "%.2f" % swd.item())
            n += 1 
                    
            total_loss.backward(retain_graph=True)
            #decoder_loss.backward()
            
            optimizer.step()
            scheduler.step()
            #optimizer_decodr.step()
            
        if epoch % 5 == 0:
            torch.save({
                "autoencoder": model.state_dict(),
                "aux_head": aux_head.state_dict(),
                "recon_head": recon_head.state_dict(),
            }, 'checkpoint.pth')

train()
