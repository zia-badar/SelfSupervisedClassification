import copy
import re
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from model import Model
from serbia import Serbia


def get_batch_size(modelCLass, dataset):
    batch_size = 1024

    model = modelCLass()
    model = model.cuda()
    model = nn.DataParallel(model)

    optimizer = torch.optim.Adam(model.parameters())
    loss_func = torch.nn.BCEWithLogitsLoss()

    def possible(new_batch_size):
        try:
            def dataloader_wrapper(dataloader):
                max_iter = 1
                for i, batch in enumerate(dataloader):
                    if i >= max_iter:
                        break
                    yield batch

            dataloader = dataloader_wrapper(DataLoader(dataset, batch_size=new_batch_size, shuffle=True, drop_last=True, pin_memory=True))

            model.train()
            for x, l in dataloader:
                optimizer.zero_grad()
                x = x.cuda(non_blocking=True)[:, 0]
                l = l.cuda(non_blocking=True)
                pred = model(x)
                loss = loss_func(pred, l)

                loss.backward()
                optimizer.step()

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

training_percentages = np.arange(1, 101, 1)

if __name__ == '__main__':
    no_workers = 32
    results_directory = Path('results/supervised')
    models_directory = results_directory / 'models'
    evaluation_directory = results_directory / 'evaluations'
    continue_training = True

    train_dataset = Serbia(split='train')
    batch_size = get_batch_size(Model, train_dataset) - 5

    validation_dataset = Serbia(split='validation')
    validation_dataloader = DataLoader(validation_dataset, batch_size=batch_size, num_workers=no_workers, shuffle=True, drop_last=True, pin_memory=True)

    if continue_training:
        saved_models = [(len(str(path)), str(path)) for path in models_directory.glob('*')]
        saved_models.sort(reverse=True)
        if len(saved_models) > 0:
            latest_saved_model = saved_models[0][1]
            latest = (float)(re.sub('.*_', '', latest_saved_model))
            percentages = training_percentages[training_percentages > latest]

    # loss_func = DB_Loss(train_dataloader)
    loss_func = torch.nn.BCEWithLogitsLoss()

    percent_tqdm = tqdm(percentages, position=1)
    for percent in percent_tqdm:
        percent = np.round(percent, 2)
        percent_tqdm.set_description(f"percentage of data for training: {percent}")
        model = Model()
        model = model.cuda()
        model = nn.DataParallel(model)
        optim = torch.optim.Adam(model.parameters())

        last_loss = sys.float_info.max
        last_model = None
        epoch = 1
        subset_size = (int)(len(train_dataset) * percent / 100)
        train_percent_dataset = random_split(train_dataset, [subset_size, len(train_dataset) - subset_size])[0]
        train_dataloader = DataLoader(train_percent_dataset, batch_size=batch_size, num_workers=no_workers,
                                      shuffle=True, pin_memory=True)
        early_stop = False
        while not early_stop:
            model.train()
            for x, l in tqdm(train_dataloader, desc=f'training epoch: {epoch}', position=2, leave=False):
                optim.zero_grad()
                x = x.cuda(non_blocking=True)[:, 0]
                l = l.cuda(non_blocking=True)
                pred = model(x)
                loss = loss_func(pred, l)

                loss.backward()
                optim.step()

            if epoch % 5 == 0:
                model.eval()
                with torch.no_grad():
                    loss = 0
                    for x, l in validation_dataloader:
                        x = x[:, 0].cuda()
                        l = l.cuda()
                        loss += loss_func(model(x), l)

                early_stop = (last_loss - loss < 0)
                if not early_stop:
                    last_loss = loss
                    last_model = copy.deepcopy(model)

            epoch += 1

        torch.save(last_model.state_dict(), str(models_directory / f'trained_supervised_model_{"{:.2f}".format(percent)}'))
