from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

import DCL.model
from dcl_loss import DCL_loss, DCL_classifier
from evaluator import Evaluator
from metrics import CustomAccuracy, Accuracy
from patch import Patch
from serbia import Serbia

def train(net, data_loader, train_optimizer, dcl_loss):
    net.train()
    total_loss, total_num, train_bar = 0.0, 0, tqdm(data_loader)
    for pos_1, pos_2, target in train_bar:
        loss = dcl_loss(net, pos_1, pos_2, target)
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
    no_workers = 16       # dont use no. of workeer for some reason, classification accuracy changes, cause is not yet found
    epochs = 60
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

    train_x = []
    train_y = []
    with torch.no_grad():
        for x, _, l in tqdm(train_dataloader):
            f, _ = model(x)
            train_x.append(f)
            train_y.append(l)

    train_x = torch.cat(train_x, dim=0)
    train_y = torch.cat(train_y, dim=0).cuda()


    evaluator = Evaluator(results_directory)
    dataloaders = {'train': train_dataloader, 'validation': validation_dataloader, 'test': test_dataloader}
    metrics = [Accuracy().cuda(), CustomAccuracy().cuda()]
    dcl_classifier = DCL_classifier(None, (train_x, train_y))
    def model_wrapper(m):
        dcl_classifier.dcl_model = m
        return dcl_classifier
    evaluator.evaluate(dataloaders, metrics, DCL.model.Model, model_wrapper, percentage_diff=0.1, max_percentage=0.5)
    evaluator.save()
    evaluator.plot()