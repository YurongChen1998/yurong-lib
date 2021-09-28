import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def loss_function(recon_x, x, mu, logvar):
    l1_loss = nn.L1Loss()
    reconstruction_loss = F.mse_loss(recon_x, x) + l1_loss(recon_x, x)
    KL_divergence = -0.5 * torch.sum(1 + logvar - torch.exp(logvar) - mu ** 2)
    return reconstruction_loss, KL_divergence

def loss_function_AE(recon_x, x):
    reconstruction_loss = F.mse_loss(recon_x, x)
    KL_divergence = 0.
    return reconstruction_loss, KL_divergence

b = 101

class VAEFC(nn.Module):
    def __init__(self, z_dim):
        super(VAEFC, self).__init__()
        self.fc1 = nn.Sequential(
            nn.Linear(b, 400),
            nn.BatchNorm1d(400),
            nn.ReLU(),
            nn.Linear(400, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
        )

        self.fc2_mean = nn.Linear(128, z_dim)
        self.fc2_logvar = nn.Linear(128, z_dim)

        self.fc3 = nn.Sequential(
            nn.Linear(z_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 400),
            nn.BatchNorm1d(400),
            nn.ReLU(),
        )
        self.fc4 = nn.Linear(400, b)

    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        return self.fc2_mean(h1), self.fc2_logvar(h1)

    def reparametrization(self, mu, logvar):
        std = 0.5 * torch.exp(logvar)
        z = torch.randn(std.size()).to(device) * std + mu
        return z

    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        return torch.sigmoid(self.fc4(h3))

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparametrization(mu, logvar)
        return self.decode(z), z#mu, logvar


class VAEFC_AE(nn.Module):
    def __init__(self, z_dim):
        super(VAEFC, self).__init__()
        self.fc1 = nn.Sequential(
            nn.Linear(200, 400),
            nn.BatchNorm1d(400),
            nn.ReLU(),
            nn.Linear(400, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
        )
        self.fc2= nn.Linear(128, z_dim)

        self.fc3 = nn.Sequential(
            nn.Linear(z_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 400),
            nn.BatchNorm1d(400),
            nn.ReLU(),
        )
        self.fc4 = nn.Linear(400, 200)

    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        return self.fc2(h1)

    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        return torch.sigmoid(self.fc4(h3))

    def forward(self, x):
        z = self.encode(x)
        return self.decode(z)
