from model import VAEFC, loss_function
import torch
from data_load import get_dataloader, test_dataloader
import torch.optim.lr_scheduler as LR
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

''' Demo
model  = VAEFC(z_dim = 64)
ip = torch.rand(128, 205)
op = model(ip)
print(op[0].shape, op[1].shape, op[2].shape)
'''

def train():
    train_loader, val_loader = get_dataloader()
    test_loader = test_dataloader()
    model = VAEFC(z_dim = 16).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    scheduler = LR.StepLR(optimizer, step_size=30)

    for epoch in range(1000):
        model.train()
        scheduler.step()
        all_loss = 0.
        for i, sample in enumerate(train_loader):
            train_data = sample[0]
            train_data = train_data['data'].float().to(device)
            recon_data, mu, logvar = model(train_data)

            recon_loss, KL_diver = loss_function(train_data, recon_data, mu, logvar)
            loss = 50*recon_loss + KL_diver
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            all_loss += loss.item()

            if i % 50 == 0:
                print('Epoch {}, recon loss:{:.6f}, KL:{:.6f}, loss: {:.6f}'.format(epoch + 1, recon_loss, KL_diver,
                                                                                    all_loss/(i+1)))
        plt.figure()
        plt.plot(train_data[0, :].cpu(), label = "input", color='blue')
        plt.plot(recon_data[0, :].cpu().detach().numpy(), label = "recon", color='coral')
        plt.savefig("fig/plot_{}.png".format(epoch))
        plt.close()
        
        model.eval()
        
        evl_loss = 0.
        for nor_i, sample in enumerate(val_loader):
            t_data = sample[0]
            test_data = t_data['data'].float().to(device)
            recon_data, mu, logvar = model(test_data)
            recon_loss, _ = loss_function(test_data, recon_data, mu, logvar)
            evl_loss += recon_loss
            
        plt.figure()
        plt.plot(test_data[0, :].cpu(), label = "input", color='blue')
        plt.plot(recon_data[0, :].cpu().detach().numpy(), label = "recon", color='coral')
        plt.savefig("test/nor/plot_{}.png".format(epoch))
        plt.close()

        ab_evl_loss = 0.
        for i, sample in enumerate(test_loader):
            t_data = sample[0]
            test_data = t_data['data'].float().to(device)
            recon_data, mu, logvar = model(test_data)
            recon_loss, _ = loss_function(test_data, recon_data, mu, logvar)
            ab_evl_loss += recon_loss
            
        plt.figure()
        plt.plot(test_data[0, :].cpu(), label = "input", color='blue')
        plt.plot(recon_data[0, :].cpu().detach().numpy(), label = "recon", color='coral')
        plt.savefig("test/ab/plot_{}.png".format(epoch))
        plt.close()
                
        print('>>> Abnormal Error:{:.3f}'.format(ab_evl_loss/(i+1)),'>>> Normal Error:{:.3f}'.format(evl_loss/(nor_i+1)), '>>> Ratio:{:.3f}'.format((ab_evl_loss/(i+1)) / (evl_loss/(nor_i+1))),)
        
        if epoch % 50 == 0: 
            ckpt = {"model": model.state_dict(),
                "optimizer": optimizer.state_dict()}
            torch.save(ckpt, "logs/checkpoint_{}.pt".format(epoch))
train()
