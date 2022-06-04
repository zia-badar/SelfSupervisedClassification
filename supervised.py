from cmath import nan
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision.models.resnet import resnet50
from torch import nn
from torch import Tensor
from tqdm import tqdm

import utils
from patch import Patch
from serbia import Serbia
import xml.etree.ElementTree as tree


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

        f = []
        for child in resnet50(num_classes = Patch.classes).named_children():
            layer = child[1]
            if child[0] == 'conv1':
                layer = nn.Conv2d(Patch.bands, 64, kernel_size=7, stride=2, padding=3, bias=False)
            f.append(layer)
            if child[0] == 'avgpool':
                f.append(nn.Flatten())

        f.append(nn.Sigmoid())

        self.f = nn.Sequential(*f)

    def forward(self, x: Tensor) -> Tensor:
        x = self.f(x)
        x = torch.sigmoid(x)
        return x


if __name__ == '__main__':
    batch_size = 128
    no_workers = 8
    epochs = 100

    train_dataset = Serbia(Path('serbia_dataset_lmdb'), split='train')
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, num_workers=8, shuffle=True, drop_last=True)

    test_dataset = Serbia(Path('serbia_dataset_lmdb'), split='test')
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, num_workers=8, shuffle=True, drop_last=True)

    model = Model().cuda()

    optim = torch.optim.Adam(model.parameters())
    loss_func = nn.BCEWithLogitsLoss()

    for epoch in range(1, epochs):

        bar = tqdm(train_dataloader, desc=f'training epoch: {epoch}')
        model.train()
        for x, l in bar:
            pred = model(x.cuda())
            loss = loss_func(pred, l.cuda())

            optim.zero_grad()
            loss.backward()
            optim.step()

        test_loss = 0
        bar = tqdm(test_dataloader)
        model.eval()
        for x, l in bar:
            pred = model(x.cuda())
            test_loss += loss_func(pred, l.cuda()).item()
            bar.set_description(f'testing, test loss: {test_loss}')