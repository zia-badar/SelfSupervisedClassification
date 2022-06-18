import re
from pathlib import Path

import torch
from torch import Tensor
from torch import nn
from torch.utils.data import DataLoader
from torchvision.models import resnet50
from tqdm import tqdm

import utils
from evaluator import Evaluator
from metrics import Accuracy, Loss
from patch import Patch
from serbia import Serbia
from loss import DB_Loss
from model import Model

if __name__ == '__main__':
    batch_size = 256
    no_workers = 16
    epochs = 150
    results_directory = Path('results/supervised')
    models_directory = results_directory / 'models'
    continue_training = True

    train_dataset = Serbia(split='train')
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, num_workers=no_workers, shuffle=True, drop_last=True, pin_memory=True)

    validation_dataset = Serbia(split='validation')
    validation_dataloader = DataLoader(validation_dataset, batch_size=batch_size, num_workers=no_workers, shuffle=True, drop_last=True, pin_memory=True)

    test_dataset = Serbia(split='test')
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, num_workers=no_workers, shuffle=True, drop_last=True, pin_memory=True)

    model = Model()
    model = model.cuda()
    model = nn.DataParallel(model)

    starting_epoch = 1
    if continue_training:
        saved_models = [(len(str(path)), str(path)) for path in models_directory.glob('*')]
        saved_models.sort(reverse=True)
        if len(saved_models) > 0:
            latest_saved_model = saved_models[0][1]
            starting_epoch = (int)(re.sub('.*_', '', latest_saved_model)) + 1
            model.load_state_dict(torch.load(latest_saved_model))

    optim = torch.optim.Adam(model.parameters())
    positive_weights, negative_weights, count = utils.calculate_class_weights(train_dataloader)
    positive_weights = positive_weights.cuda(non_blocking=True)
    negative_weights = negative_weights.cuda(non_blocking=True)

    # loss_func = DB_Loss(train_dataloader)
    loss_func = torch.nn.BCEWithLogitsLoss()

    for epoch in range(starting_epoch, epochs+1):

        bar = tqdm(train_dataloader, desc=f'training epoch: {epoch}')
        model.train()
        for x, l in bar:
            optim.zero_grad()
            x = x.cuda(non_blocking=True)
            l = l.cuda(non_blocking=True)
            pred = model(x)
            loss = loss_func(pred, l)

            loss.backward()
            optim.step()

        torch.save(model.state_dict(), str(models_directory / f'trained_supervised_model_epoch_{epoch}'))

    evaluator = Evaluator(results_directory)
    dataloaders = {'train': train_dataloader, 'validation': validation_dataloader, 'test': test_dataloader}
    metrics = [Accuracy().cuda(), Loss(loss_func).cuda()]
    evaluator.evaluate(dataloaders, metrics, Model)
    evaluator.save()
    # evaluator = evaluator.load(Path('results/supervised/evaluations/evaluation_1'))
    evaluator.plot()
