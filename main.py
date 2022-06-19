from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn.parameter import Parameter
from torch.utils.data import DataLoader
from tqdm import tqdm

import DCL.model
from dcl_loss import DCL_loss
from evaluator import Evaluator
from metrics import Loss, Accuracy
from patch import Patch
from serbia import Serbia

def train(net, data_loader, train_optimizer, dcl_loss):
    net.train()
    total_loss, total_num, train_bar = 0.0, 0, tqdm(data_loader)
    for pos_1, pos_2, target in train_bar:
        loss = dcl_loss(pos_1, pos_2)
        train_optimizer.zero_grad()
        loss.backward()
        train_optimizer.step()

        total_num += batch_size
        total_loss += loss.item() * batch_size

        train_bar.set_description('Train Epoch: [{}/{}] Loss: {:.4f}'.format(epoch, epochs, total_loss / total_num))

    print(f'average loss: {total_loss / total_num}')
    return total_loss / total_num

if __name__ == '__main__':

    batch_size = 32
    no_workers = 16
    epochs = 200
    results_directory = Path('results/unsupervised')
    models_directory = results_directory / 'models'
    continue_training = True

    train_dataset = Serbia(split='train')
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, num_workers=no_workers, shuffle=True, drop_last=True, pin_memory=True)

    test_dataset = Serbia(split='test')
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, num_workers=no_workers, shuffle=True, drop_last=True, pin_memory=True)

    validation_dataset = Serbia(split='validation')
    validation_dataloader = DataLoader(validation_dataset, batch_size=batch_size, num_workers=no_workers, shuffle=True, drop_last=True, pin_memory=True)

    model = DCL.model.Model(128).cuda()
    model = nn.DataParallel(model)

    model.load_state_dict(torch.load(str(results_directory / 'models' / 'model_60')))

    starting_epoch = 1
    if continue_training:
        saved_models = [(len(str(path)), str(path)) for path in models_directory.glob('*')]
        saved_models.sort(reverse=True)
        if len(saved_models) > 0:
            latest_saved_model = saved_models[0][1]
            starting_epoch = (int)(latest_saved_model.rsplit('_', 1)[1]) + 1
            model.load_state_dict(torch.load(latest_saved_model))

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-6)
    loss = DCL_loss(.5, True, 1./Patch.classes, batch_size)

    for epoch in range(starting_epoch, epochs + 1):
        train(model, train_dataloader, optimizer, loss)
        if epoch % 5 == 0:
            torch.save(model.state_dict(), str(models_directory / f'model_{epoch}'))

    evaluator = Evaluator(results_directory)
    dataloaders = {'train': train_dataloader, 'validation': validation_dataloader, 'test': test_dataloader}
    metrics = [Loss(loss).cuda()]
    evaluator.evaluate(dataloaders, metrics, DCL.model.Model)
    evaluator.save()
    # evaluator = evaluator.load(Path(results_directory / 'evaluations' / 'evaluation_1'))
    # evaluator.plot()

    # model.eval()
    # m = torch.nn.Sequential(torch.nn.Linear(2048, 2048), torch.nn.Linear(2048, 19))
    # m = m.cuda()
    # optimizer = torch.optim.Adam(m.parameters())
    # for e in tqdm(range(1, 100)):
    #     for x, aug, l in tqdm(train_dataloader, position=0, desc='models'):
    #         optimizer.zero_grad()
    #         feat, _ = model(x)
    #         feat = feat.cuda()
    #         pred = m(feat)
    #         loss = torch.nn.BCEWithLogitsLoss()(pred, l.cuda())
    #         loss.backward()
    #         optimizer.step()
    #
    #     metric = Accuracy().cuda()
    #     metric.reset()
    #     for x, aug, l in tqdm(test_dataloader, position=1, leave=False, desc='dataloaders'):
    #         feat, _ = model(x)
    #         feat = feat.cuda()
    #         pred = m(feat)
    #         metric(pred, l)
    #     print(f'accuracy: {metric.compute()}')
