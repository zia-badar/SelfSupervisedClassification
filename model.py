from torch import nn, Tensor
from torchvision.models import resnet50

from patch import Patch


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

        self.f = resnet50()
        self.f.conv1 = nn.Conv2d(Patch.bands, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.f.fc = nn.Linear(512 * 4, Patch.classes)

    def forward(self, x: Tensor) -> Tensor:
        x = self.f(x)
        return None, x
