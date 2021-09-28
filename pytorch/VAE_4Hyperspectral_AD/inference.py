from model import VAEFC, loss_function
import torch
from data_load import all_dataloader
import torch.optim.lr_scheduler as LR
import matplotlib.pyplot as plt
from sklearn import manifold
import seaborn as sns
import numpy as np


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

''' Demo
model  = VAEFC(z_dim = 64)
ip = torch.rand(128, 205)
op = model(ip)
print(op[0].shape, op[1].shape, op[2].shape)
'''

def test():
    train_loader = all_dataloader()
    model = VAEFC(z_dim = 16).to(device)
    ckpt = torch.load("logs/checkpoint_400.pt")
    model.load_state_dict(ckpt["model"])
    
    data_tsne = []
    label_tsne = []
        
    model.eval()
    file_nor = open('Error_Nor.txt', 'a')
    file_ab = open('Error_Ab.txt', 'a')

    for i, sample in enumerate(train_loader):
        t_data = sample[0]
        test_data = t_data['data'].float().to(device)
        test_label = t_data['label'].float().to(device)
        recon_data, z = model(test_data)
        recon_loss, _ = loss_function(test_data, recon_data, z, z)
        data_tsne += list(np.array(z.detach().cpu()))
        label_tsne += list([str(test_label.item())])
        
        if test_label.item() == 0:
            print('Normal .... :{:.3f}'.format(recon_loss))
            s = (str(float(recon_loss)) + '\n')
            file_nor.write(s)
            
            plt.figure()
            plt.plot(test_data[0, :].cpu(), label = "input", color='blue')
            plt.plot(recon_data[0, :].cpu().detach().numpy(), label = "recon", color='coral')
            plt.savefig("test/nor/plot_{}.png".format(i))
            plt.close()
        else:
            print('Abnormal ... :{:.3f}'.format(recon_loss))
            s = (str(float(recon_loss)) + '\n')
            file_ab.write(s)
            
            plt.figure()
            plt.plot(test_data[0, :].cpu(), label = "input", color='blue')
            plt.plot(recon_data[0, :].cpu().detach().numpy(), label = "recon", color='coral')
            plt.savefig("test/ab/plot_{}.png".format(i))
            plt.close()
            
    tsne = manifold.TSNE(n_components=2, init='pca', random_state=501)
    X_tsne = tsne.fit_transform(data_tsne)
    ax = sns.jointplot(x = X_tsne[:, 1], y = X_tsne[:, 0], hue=label_tsne)
    plt.show()
      
test()
