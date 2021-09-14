from torch import nn
from torch.utils import model_zoo
from torchvision.models.resnet import BasicBlock, model_urls, Bottleneck
import torch
from torch import distributions as dist
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Aux_Head(nn.Module):
    def __init__(self, aux_classes=1000, latent_dim=128):
        super(Aux_Head, self).__init__()
        self.aux_classes = aux_classes
        self.latent_dim = latent_dim
        self.fc = nn.Linear(self.latent_dim * 9, aux_classes)

    def forward(self, x):
        return self.fc(x)

class Recon_Head(nn.Module):
    def __init__(self, block = Bottleneck):
        super(Recon_Head, self).__init__()
        
        self.decoder = nn.Sequential(
            UpBlock(512*block.expansion, 512, upsample=True),
            UpBlock(512, 256, upsample=True),
            nn.Conv2d(256, 256, 7, 1),
            UpBlock(256, 128, upsample=True),
            UpBlock(128, 64, upsample=True),
            UpBlock(64, 32, upsample=True),
            nn.Conv2d(32, 3, 1, 1),
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
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.aux_classifier = nn.Linear(512 * block.expansion, aux_classes)

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

        #reconstruction = self.decoder(z)
        #return aux_out, reconstruction, z
        return aux_out, z, tsne
 
                
def resnet18(pretrained=True, **kwargs):
    """Constructs a ResNet-18 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet18']), strict=False)
    return model

def resnet50(pretrained=True, **kwargs):
    """Constructs a ResNet-50 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet50']), strict=False)
    return model


def compute_swd(z, i):
    batch, channel, h, w = z.shape
    z = z.view(batch, -1)
    prior_z = i + torch.randn_like(z)
    #prior_z = dist.StudentT(torch.tensor([20.0]), torch.tensor([int(i)]), torch.tensor([1.0])).sample((1, channel * h * w)).view(1,
    #                                                                                                          -1).to(device)
    
    rand_sample = dist.Cauchy(torch.tensor([0.0]), torch.tensor([1.0])).sample((1, channel * h * w)).view(1,
                                                                                                              -1).to(device)
    rand_proj = rand_sample / rand_sample.norm(dim=1).view(-1, 1)

    proj_matrix = rand_proj.transpose(0, 1).to(device)

    latent_projections = z.matmul(proj_matrix)  # [N x S]
    prior_projections = prior_z.matmul(proj_matrix)  # [N x S]
    return latent_projections, prior_projections

    # The Wasserstein distance is computed by sorting the two projections
    # across the batches and computing their element-wise l2 distance
def compute_swd_final(latent_projections, prior_projections):
    w_dist = torch.sort(latent_projections.t(), dim=1)[0] - \
             torch.sort(prior_projections.t(), dim=1)[0]
    sw_dist = w_dist.pow(2)
    return sw_dist.mean()
    
def compute_lower(z, latent):
    batch, channel, h, w = z.shape
    z = z.view(batch, -1)
    latent = latent.view(batch, -1)
    
    ##first_term = torch.sum(torch.log(z + 10e-5)) + torch.sum(z * torch.log(z + 10e-5))
    first_term = torch.sum(z * torch.log(z + 10e-5))
    return torch.div(first_term, channel)

def compute_hingle(latent_projections, prior_projections): 
    num = latent_projections.size(1)
    distance_matrix = latent_projections.expand(num, num)
    min_distance = torch.pow((distance_matrix.t() - distance_matrix), 2)
    min_distance = min_distance + torch.eye(num).cuda()
    min_index = min_distance.argmin(dim=1) 
    
    distance_mean_matrix = prior_projections.expand(num, num)
    distance_mean_matrix = torch.pow((distance_mean_matrix.t() - distance_matrix), 2)
    
    hingle = 0
    
    for i in range(len(distance_mean_matrix)):
        hingle += 1 + distance_mean_matrix[i][i] - distance_mean_matrix[i][min_index[i]]
    return hingle
        
    
