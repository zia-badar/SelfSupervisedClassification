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

def train(net, data_loader, train_optimizer, dcl_loss, silent_mode=False):
    net.train()
    total_loss, total_num, train_bar = 0.0, 0, tqdm(data_loader) if not silent_mode else data_loader
    for batch in train_bar:
        loss = dcl_loss(net, batch)
        train_optimizer.zero_grad()
        loss.backward()
        train_optimizer.step()

        total_num += batch[0].shape[0]
        total_loss += loss.item() * batch[0].shape[0]

        if not silent_mode:
            train_bar.set_description('Train Epoch: [{}/{}] Loss: {:.4f}'.format(epoch, epochs, total_loss / total_num))

    if not silent_mode:
        print(f'average loss: {total_loss / total_num}')
    return total_loss / total_num

def get_batch_size(modelCLass, dataset):
    batch_size = 1024

    model = modelCLass(128).cuda()
    model = nn.DataParallel(model)
    optimizer = torch.optim.Adam(model.parameters())
    loss = DCL_loss(.5, True, 1. / Patch.classes, -1)

    def possible(new_batch_size):
        try:
            def dataloader_wrapper(dataloader):
                max_iter = 1
                for i, batch in enumerate(dataloader):
                    if i >= max_iter:
                        break
                    yield batch
            train(model, dataloader_wrapper(DataLoader(dataset, new_batch_size, shuffle=True, pin_memory=True, drop_last=True)), train_optimizer=optimizer, dcl_loss=loss, silent_mode=True)
            return True
        except RuntimeError as ex:
            return False

    min, max, = 1, batch_size
    best_batch_size = min
    while min <= max:
        mid = (int)((min + max)/2)
        if possible(mid):
            best_batch_size = mid
            min = mid + 1
        else:
            max = mid - 1

        torch.cuda.empty_cache()            # avoid cluter of previous iterations on gpu

    return best_batch_size


if __name__ == '__main__':

    no_workers = 16
    epochs = 60
    results_directory = Path('results/unsupervised')
    models_directory = results_directory / 'models'
    continue_training = True
    model_class = DCL.model.Model

    train_dataset = Serbia(split='train')
    batch_size = get_batch_size(model_class, train_dataset)
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
    loss = DCL_loss(.5, True, 1./Patch.classes, -1)

    for epoch in range(starting_epoch, epochs + 1):
       train(model, train_dataloader, optimizer, loss)
       if epoch % 1 == 0:
            torch.save(model.state_dict(), str(models_directory / f'model_{epoch}'))

            train_x = []
            train_y = []
            with torch.no_grad():
                for x, l in tqdm(train_dataloader):
                    x = x[:, 0]
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
            evaluator.evaluate(dataloaders, metrics, DCL.model.Model, model_wrapper, percentage_diff=0.5, max_percentage=0.5)
            print(evaluator)

    # evaluator.save()
    # evaluator.plot()