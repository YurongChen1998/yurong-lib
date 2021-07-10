from JigsawDataset import get_dataloader
from model import resnet50, Aux_Head, Recon_Head, compute_swd
import torch
from torch.optim import Adam
import torch.nn as nn
import torch.nn.functional as F
import torch.optim.lr_scheduler as LR
import torchvision
from torchvision.utils import save_image
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from sklearn import manifold, datasets
import seaborn as sns

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def test():
    n = 0
    train_loader = get_dataloader()
    model = resnet50().to(device)
    aux_head = Aux_Head().to(device)
    recon_head = Recon_Head().to(device)

    PATH = 'checkpoint_950.pth'
    checkpoint = torch.load(PATH)
    model.load_state_dict(checkpoint['autoencoder'])
    aux_head.load_state_dict(checkpoint['aux_head'])
    recon_head.load_state_dict(checkpoint['recon_head'])

    aux_class_loss_func = nn.CrossEntropyLoss()
    recon_loss = torch.nn.L1Loss()
    
    Loss = []
    
    with torch.no_grad():
        data_tsne = []
        target_tsne = []
        for i, (train_data, _) in enumerate(train_loader):
            
            train_image = train_data['images'].to(device)
            intput_image = torchvision.utils.make_grid(train_image.squeeze(), 3, padding=0)
            target = train_data['aux_labels'].to(device)
            label = train_data['label'].to(device)
            tem_feature = []
            recon_image = []

            recon_error = 0.0
            swd = 0.0
            for i in range(train_image.size(1)):
                ip = train_image[:,i,:,:,:]
                aux_op, latent, tsne = model(ip)
                swd += compute_swd(latent)
                tem_feature.append(aux_op)
                recon_image.append(latent)
                data_tsne += list(np.array(tsne.cpu()))
                target_tsne += list([str(i)])
            tem_feature = torch.stack(tem_feature, 1).view(1, -1)
            final_out = aux_head(tem_feature)
            aux_loss = aux_class_loss_func(final_out, target)
            recon_image = torch.stack(recon_image, 1).squeeze()
            recon_image = torchvision.utils.make_grid(recon_image, 3, padding=0).unsqueeze(0)
            recon_image = recon_head(recon_image)
            recon_image = recon_image.squeeze()
            sum_recon_loss = recon_loss(recon_image, intput_image)
            sum_recon_loss += F.mse_loss(recon_image, intput_image)
            total_loss = recon_error + 10e2 * aux_loss + 10e2 * sum_recon_loss + swd
            Loss.append(total_loss.item())
            
            save_image(intput_image, 'test/input/in_{}.jpg'.format(n))
            save_image(recon_image, 'test/output/out_{}.jpg'.format(n))
            n += 1
            print("Epoch", 1, "Total_loss:", "%.2f" % total_loss.item(), "Aux_loss:", "%.2f" % aux_loss.item(),
                "Recon_sum_loss:", "%.2f" % sum_recon_loss.item() , "swd_loss:", "%.2f" % swd.item())
                
        tsne = manifold.TSNE(n_components=2, init='pca', random_state=501)
        X_tsne = tsne.fit_transform(data_tsne)
        
        colors = ['#713e5a', '#63a375', '#edc79b', '#d57a66', '#ca6680', '#395B50', '#92AFD7', '#b0413e', '#4381c1']
        sns.set_palette(sns.color_palette(colors))
        
        ax = sns.jointplot(x=X_tsne[:,1], y=X_tsne[:,0], hue=target_tsne)

        plt.show()
        #cluster.tsneplot(score=X_tsne, colorlist=y, legendanchor=(1.15, 1), colordot=('#713e5a', '#63a375', '#edc79b', '#d57a66', '#ca6680', '#395B50', '#92AFD7', '#b0413e', '#4381c1'))
        
        file_ = open('loss.txt', 'w')
        for i in range(len(Loss)):
            s = str(Loss[i]) + '\n'
            file_.write(s)
        file_.close()
   

def mainfold():
    n = 0
    train_loader = get_dataloader('./fig1')
    train_loader2 = get_dataloader('./fig2')
    model = resnet50().to(device)
    aux_head = Aux_Head().to(device)
    recon_head = Recon_Head().to(device)

    PATH = 'checkpoint_950.pth'
    checkpoint = torch.load(PATH)
    model.load_state_dict(checkpoint['autoencoder'])
    aux_head.load_state_dict(checkpoint['aux_head'])
    recon_head.load_state_dict(checkpoint['recon_head'])

    aux_class_loss_func = nn.CrossEntropyLoss()
    recon_loss = torch.nn.L1Loss()
    
    Loss = []
    
    with torch.no_grad():
        data_tsne = []
        target_tsne = []
        
        train_data, _ = next(iter(train_loader))
        train_image = train_data['images'].to(device)
        
        train_data_2, _ = next(iter(train_loader2))
        train_image_2 = train_data_2['images'].to(device)
        
        intput_image = torchvision.utils.make_grid(train_image.squeeze(), 3, padding=0)
        intput_image_2 = torchvision.utils.make_grid(train_image_2.squeeze(), 3, padding=0)

        tem_feature = []
        recon_image = []

        for i in range(train_image.size(1)):
            ip = train_image[:,i,:,:,:]
            aux_op, latent, tsne = model(ip)

            tem_feature.append(aux_op)
            recon_image.append(latent)
               
            data_tsne += list(np.array(tsne.cpu()))
            target_tsne += list([str(i)])

        tem_feature = torch.stack(tem_feature, 1).view(1, -1)
        mainfold_feature  = recon_image
        recon_image = torch.stack(recon_image, 1).squeeze()
        recon_image = torchvision.utils.make_grid(recon_image, 3, padding=0).unsqueeze(0)
        recon_image = recon_head(recon_image)
        recon_image = recon_image.squeeze()
       
            
        save_image(intput_image, 'test/input/in_{}.jpg'.format(n))
        save_image(recon_image, 'test/output/out_{}.jpg'.format(n))
        n += 1
        

        tem_feature = []
        recon_image = []
        for i in range(train_image_2.size(1)):
            ip = train_image_2[:,i,:,:,:]
            aux_op, latent, tsne = model(ip)

            tem_feature.append(aux_op)
            recon_image.append(latent)
               
            data_tsne += list(np.array(tsne.cpu()))
            target_tsne += list([str(i)])

        tem_feature = torch.stack(tem_feature, 1).view(1, -1)
        mainfold_feature_2  = recon_image
        recon_image = torch.stack(recon_image, 1).squeeze()
        recon_image = torchvision.utils.make_grid(recon_image, 3, padding=0).unsqueeze(0)
        recon_image = recon_head(recon_image)
        recon_image = recon_image.squeeze()
       
            
        save_image(intput_image_2, 'test/input/in_{}.jpg'.format(n))
        save_image(recon_image, 'test/output/out_{}.jpg'.format(n))
        n += 1

        
        patch_6 = 4
        residual_6 = mainfold_feature_2[patch_6] - mainfold_feature[patch_6]
        #patch_7 = 1
        #residual_7 = mainfold_feature_2[patch_7] - mainfold_feature[patch_7]
        #patch_8 = 2
        #residual_8 = mainfold_feature_2[patch_8] - mainfold_feature[patch_8]
        for sli in range(0,100):
            mainfold_feature[patch_6] = mainfold_feature_2[patch_6] - residual_6*(sli/ 100)
            #mainfold_feature[patch_7] = mainfold_feature_2[patch_7] - residual_7*(sli/ 100)
            #mainfold_feature[patch_8] = mainfold_feature_2[patch_8] - residual_8*(sli/ 100)

            recon_image = torch.stack(mainfold_feature, 1).squeeze()
            recon_image = torchvision.utils.make_grid(recon_image, 3, padding=0).unsqueeze(0)
            #print(">>>>>>>>>>", recon_image.shape)
            recon_image = recon_head(recon_image)
            recon_image = recon_image.squeeze()

            #save_image(mainfold_feature_2, 'test/input/main_in_{}.jpg'.format(sli))
            save_image(recon_image, 'test/output/main_out_{}.jpg'.format(sli))
        
        
        '''
        residual = []
        for idx in range(len(mainfold_feature_2)):
            residual.append(mainfold_feature_2[idx] - mainfold_feature[idx])

        for sli in range(100):
            for idx in range(len(mainfold_feature_2)):
                mainfold_feature[idx] = mainfold_feature_2[idx] - residual[idx]*(sli/ 100)
        
            recon_image = torch.stack(mainfold_feature, 1).squeeze()
            recon_image = torchvision.utils.make_grid(recon_image, 3, padding=0).unsqueeze(0)
            #print(recon_image.shape, ">>>>>>>>>>>>")
            recon_image = recon_head(recon_image)
            recon_image = recon_image.squeeze()

            #save_image(mainfold_feature_2, 'test/input/main_in_{}.jpg'.format(sli))
            save_image(recon_image, 'test/output/main_out_{}.jpg'.format(sli))
        '''

 
#test()
mainfold()
