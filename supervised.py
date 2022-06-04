import re
from pathlib import Path

import torch
from torch import Tensor
from torch import nn
from torch.utils.data import DataLoader
from torchvision.models.resnet import resnet50
from tqdm import tqdm

from patch import Patch
from serbia import Serbia

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

        self.f = nn.Sequential(*f)

    def forward(self, x: Tensor) -> Tensor:
        x = self.f(x)
        return x

if __name__ == '__main__':
    batch_size = 128
    no_workers = 16
    epochs = 300
    result_folder = 'supervised_learning_results'
    continue_training = True

    train_dataset = Serbia(Path('serbia_dataset_lmdb'), split='train')
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, num_workers=no_workers, shuffle=True, drop_last=True)

    test_dataset = Serbia(Path('serbia_dataset_lmdb'), split='test')
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, num_workers=no_workers, shuffle=True, drop_last=True)

    model = Model()
    model = model.cuda()
    model = nn.DataParallel(model)

    starting_epoch = 0
    if continue_training:
        saved_models = sorted(Path(result_folder).glob('*'), reverse=True)
        if len(saved_models) > 0:
            latest_saved_model = saved_models[0]
            starting_epoch = (int)(re.sub('.*_', '', latest_saved_model.stem))
            model.load_state_dict(torch.load(latest_saved_model))

    optim = torch.optim.Adam(model.parameters())
    loss_func = nn.BCEWithLogitsLoss()

    for epoch in range(starting_epoch, epochs+1):

        bar = tqdm(train_dataloader, desc=f'training epoch: {epoch}')
        model.train()
        for x, l in bar:
            pred = model(x.cuda())
            loss = loss_func(pred, l.cuda())

            optim.zero_grad()
            loss.backward()
            optim.step()
            torch.save(model.state_dict(), f'{result_folder}/trained_supervised_model_epoch_{epoch}')

        bar = tqdm(test_dataloader)
        model.eval()
        correct = 0
        for i, (x, l) in enumerate(bar):
            pred = model(x.cuda())
            pred = torch.sigmoid(pred)
            pred = torch.round(pred)
            correct += torch.sum(torch.all(torch.eq(pred, l.cuda()), dim=1)).item()
            bar.set_description(f'testing, test accuracy: {correct/((i+1)*batch_size)}')