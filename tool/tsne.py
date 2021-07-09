from JigsawDataset import get_dataloader, JigsawDataset, get_transformers
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
from plot import plot_diversity

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def test(test_path):
    train_loader = get_dataloader(test_path)
    model = resnet50().to(device)
    aux_head = Aux_Head().to(device)
    recon_head = Recon_Head().to(device)

    PATH = 'checkpoint_1250.pth'
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
            filename = train_data['framename'][0][-7:-4]
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
            
            total_loss = 10e2 * sum_recon_loss
            Loss.append(total_loss.item())
            
            save_image(intput_image, 'test/input/{}.jpg'.format(filename))
            save_image(recon_image, 'test/output/{}.jpg'.format(filename))
            n += 1
            print("Epoch", 1, "Total_loss:", "%.2f" % total_loss.item(), "Recon_sum_loss:", "%.2f" % sum_recon_loss.item())

        tsne = manifold.TSNE(n_components=2, init='pca', random_state=501)
        X_tsne = tsne.fit_transform(data_tsne)
        
        colors = ['#713e5a', '#63a375', '#edc79b', '#d57a66', '#ca6680', '#395B50', '#92AFD7', '#b0413e', '#4381c1']
        sns.set_palette(sns.color_palette(colors))
        
        ax = sns.jointplot(x=X_tsne[:,1], y=X_tsne[:,0], hue=target_tsne)
        plt.show()
        #cluster.tsneplot(score=X_tsne, colorlist=y, legendanchor=(1.15, 1), colordot=('#713e5a', '#63a375', '#edc79b', '#d57a66', '#ca6680', '#395B50', '#92AFD7', '#b0413e', '#4381c1'))
        
test('./images/test/aug_good')
