from torch import nn
from torch.utils import model_zoo
from torchvision.models.resnet import BasicBlock, model_urls, Bottleneck
import torch
from torch import distributions as dist
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Recon_Head(nn.Module):
    def __init__(self, block = Bottleneck):
        super(Recon_Head, self).__init__()
        
        self.decoder = nn.Sequential(
            UpBlock(512*block.expansion, 512, upsample=True),
            nn.Conv2d(512, 512, 1, 1, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            UpBlock(512, 256, upsample=True),
            nn.Conv2d(256, 256, 1, 1, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            UpBlock(256, 128, upsample=True),
            UpBlock(128, 64, upsample=True),
            UpBlock(64, 32, upsample=True),
            nn.Conv2d(32, 3, 5, 1),
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
        return x
 
                
def resnet18(pretrained=True, **kwargs):
    """Constructs a ResNet-18 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet18']), strict=False)
    return model

def resnet50(pretrained=False, **kwargs):
    """Constructs a ResNet-50 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet50']), strict=False)
        print(">>>>>>>>>>>> Load ImageNet Weight")
    return model

if __name__ == '__main__':
    encoder = resnet50()
    decoder = Recon_Head()
    ip = torch.rand(2, 3, 300, 300)
    print(decoder(encoder(ip)).shape)
    