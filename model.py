import torch
import torch.nn
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torchvision.models.resnet import resnet50

from patch import Patch


# Reused from https://arxiv.org/abs/2007.00224 's original implementation
class SelfSupervisedModel(nn.Module):
    def __init__(self, feature_dim=128):
        super(SelfSupervisedModel, self).__init__()

        self.f = []
        for name, module in resnet50().named_children():
            if name == 'conv1':
                module = nn.Conv2d(10, 64, kernel_size=3, stride=1, padding=1, bias=False)
            if not isinstance(module, nn.Linear) and not isinstance(module, nn.MaxPool2d):
                self.f.append(module)
        # encoder
        self.f = nn.Sequential(*self.f)
        # projection head
        self.g = nn.Sequential(nn.Linear(2048, 512, bias=False), nn.BatchNorm1d(512),
                               nn.ReLU(inplace=True), nn.Linear(512, feature_dim, bias=True))

    def forward(self, x):
        x = self.f(x)
        feature = torch.flatten(x, start_dim=1)
        out = self.g(feature)
        return F.normalize(feature, dim=-1), F.normalize(out, dim=-1)


class SupervisedModel(nn.Module):
    def __init__(self):
        super(SupervisedModel, self).__init__()

        self.f = resnet50()
        self.f.conv1 = nn.Conv2d(Patch.bands, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.f.fc = nn.Linear(512 * 4, Patch.classes)

    def forward(self, x: Tensor) -> Tensor:
        x = self.f(x)
        return x
