from torch import nn
from torch.utils import model_zoo
from torchvision.models.resnet import BasicBlock, model_urls, Bottleneck
import torch
from torch import distributions as dist
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Aux_Head(nn.Module):
    def __init__(self, aux_classes=100, latent_dim=128):
        super(Aux_Head, self).__init__()
        self.aux_classes = aux_classes
        self.latent_dim = latent_dim
        self.fc = nn.Linear(self.latent_dim * 9, aux_classes+1)

    def forward(self, x):
        return self.fc(x)

class Recon_Head(nn.Module):
    def __init__(self, block = Bottleneck):
        super(Recon_Head, self).__init__()
        
        self.decoder = nn.Sequential(
            UpBlock(512*block.expansion, 512, upsample=True),
            nn.Conv2d(512, 512, 3, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            UpBlock(512, 256, upsample=True),
            nn.Conv2d(256, 256, 7, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            UpBlock(256, 128, upsample=True),
            UpBlock(128, 64, upsample=True),
            UpBlock(64, 32, upsample=True),
            nn.Conv2d(32, 32, 3, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 3, 3, 1),
            nn.Tanh()
            )

    def forward(self, x):
        return self.decoder(x)

class UpBlock(nn.Module):
    def __init__(self, inplanes, planes, upsample=False):
        super(UpBlock, self).__init__()
        self.conv = nn.Conv2d(inplanes, planes, 1, 1)
        self.bn = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.will_ups = upsample

    def forward(self, x):
        if self.will_ups:
            x = nn.functional.interpolate(x,
                scale_factor=2, mode="bilinear", align_corners=True)

        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)

        return x

class ResNet(nn.Module):
    def __init__(self, block, layers, aux_classes=128):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(3, stride=1)
        #self.aux_classifier = nn.Linear(512 * block.expansion, aux_classes)
        self.aux_classifier = nn.Linear(8192, aux_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def is_patch_based(self):
        return False

    def forward(self, x, **kwargs):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        z = self.layer4(x)

        x = self.avgpool(z)
        x = x.view(x.size(0), -1)
        tsne = x
        aux_out = self.aux_classifier(x)
        return aux_out, z, tsne
 
                
def resnet18(pretrained=True, **kwargs):
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet18']), strict=False)
    return model

def resnet50(pretrained=True, **kwargs):
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet50']), strict=False)
    return model

def kernel(a, b):
    kernel_type = 'Linear'
    
    batch_a, batch_b = a.shape[0], b.shape[0]
    feature = a.shape[1]
    
    if kernel_type == 'Gauss':
        a = a.view(batch_a, 1, feature)
        b = b.view(1, batch_b, feature)
        a_core = a.expand(batch_a, batch_b, feature)
        b_core = b.expand(batch_a, batch_b, feature)
        numerator = (a_core - b_core).pow(2).mean(2) / feature
        kernel = torch.exp(-numerator)
    elif kernel_type == 'Linear':
        kernel = a.matmul(b.t())
        
    #oneN = (torch.ones([batch_a, batch_a])/batch_a).to(device)
    #kernel = kernel - oneN.matmul(kernel) - kernel.matmul(oneN) + (oneN.matmul(kernel)).matmul(oneN)
    return kernel

def mmd(a, b):
    return kernel(a, a).mean() + kernel(b, b).mean() - 2 * kernel(a, b).mean()
    
def compute_swd(z):
    _, batch, feature = z.shape
    z = z.view(batch, feature)
    prior_z = torch.randn_like(z)
    rand_sample = torch.randn_like(z).to(device)
    rand_proj = rand_sample / rand_sample.norm(dim=1).view(-1, 1)
    proj_matrix = rand_proj.transpose(0, 1).to(device)

    latent_projections = z.matmul(proj_matrix)  # [N x S]
    prior_projections = prior_z.matmul(proj_matrix)  # [N x S]

    w_dist = torch.sort(latent_projections.t(), dim=1)[0] - \
             torch.sort(prior_projections.t(), dim=1)[0]
    sw_dist = w_dist.pow(2)
    return sw_dist.mean(), mmd(z, prior_z)

