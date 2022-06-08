import re
from pathlib import Path

import torch
from torch import Tensor
from torch import nn
from torch.utils.data import DataLoader
from torchvision.models import resnet18
from tqdm import tqdm

from patch import Patch
from serbia import Serbia

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

        self.f = resnet18(pretrained=True)
        self.f.conv1 = nn.Conv2d(Patch.bands, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.f.fc = nn.Linear(512 * 1, Patch.classes)

    def forward(self, x: Tensor) -> Tensor:
        x = self.f(x)
        return x

if __name__ == '__main__':
    batch_size = 128
    no_workers = 16
    epochs = 300
    result_folder = 'supervised_learning_results'
    continue_training = True

    train_dataset = Serbia(split='train')
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, num_workers=no_workers, shuffle=True, drop_last=True, pin_memory=True)

    test_dataset = Serbia(split='test')
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, num_workers=no_workers, shuffle=True, drop_last=True, pin_memory=True)

    model = Model()
    model = model.cuda()
    model = nn.DataParallel(model)

    starting_epoch = 1
    if continue_training:
        saved_models = [(len(str(path)), str(path)) for path in Path(result_folder).glob('*')]
        saved_models.sort(reverse=True)
        if len(saved_models) > 0:
            latest_saved_model = saved_models[0][1]
            starting_epoch = (int)(re.sub('.*_', '', latest_saved_model))
            model.load_state_dict(torch.load(latest_saved_model))

    optim = torch.optim.Adam(model.parameters())
    loss_func = nn.BCEWithLogitsLoss()

    for epoch in range(starting_epoch, epochs+1):

        bar = tqdm(train_dataloader, desc=f'training epoch: {epoch}')
        model.train()
        for x, l in bar:
            optim.zero_grad()
            x = x.cuda(non_blocking=True)
            pred = model(x)
            loss = loss_func(pred, l.cuda(non_blocking=True))

            loss.backward()
            optim.step()

        torch.save(model.state_dict(), f'{result_folder}/trained_supervised_model_epoch_{epoch}')

        if epoch % 5 == 0:
            model.eval()

            bar = tqdm(train_dataloader)
            correct = 0
            for i, (x, l) in enumerate(bar):
                pred = model(x.cuda(non_blocking=True))
                pred = torch.sigmoid(pred)
                pred = torch.round(pred)
                correct += torch.sum(torch.eq(pred, l.cuda(non_blocking=True)), dim=0).item()
                bar.set_description(f'train accuracy: {correct/((i+1)*batch_size)}')

            bar = tqdm(test_dataloader)
            correct = 0
            for i, (x, l) in enumerate(bar):
                pred = model(x.cuda(non_blocking=True))
                pred = torch.sigmoid(pred)
                pred = torch.round(pred)
                correct += torch.sum(torch.eq(pred, l.cuda(non_blocking=True)), dim=0).item()
                bar.set_description(f'test accuracy: {correct/((i+1)*batch_size)}')