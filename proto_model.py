import pretrainedmodels
import torch
import torch.nn as nn
import torch.nn.functional as F

class ResNeXt(nn.Module):
    def __init__(self, num_classes=100, pretrained=False):
        super(ResNeXt,self).__init__()
        print('pre', pretrained)
        if pretrained:
            features = pretrainedmodels.resnext101_32x4d(num_classes=1000, pretrained='imagenet')
        else:
            features = pretrainedmodels.resnext101_32x4d(num_classes=num_classes, pretrained=None)
        self.first_layer = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)

        self.features = nn.Sequential(*list(features.children())[0][1:-1])

        self.classifier = nn.Sequential(
            nn.Linear(1024, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        feature_map = self.first_layer(x)

        feature_map = self.features(feature_map)

        feature_vec = mac(feature_map)
        feature_vec = l2n(feature_vec)

        feature_vec = feature_vec.view(feature_vec.size(0), -1)

        output = self.classifier(feature_vec)
        return output, feature_vec

def mac(x):
    # x = F.relu(x, inplace=True)
    return F.max_pool2d(x, (x.size(-2), x.size(-1)))


def l2n(x, eps=1e-6):
    return x / (torch.norm(x, p=2, dim=1, keepdim=True) + eps).expand_as(x)


