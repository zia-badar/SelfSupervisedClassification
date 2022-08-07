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
                break

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

def get_percent_dataloader_from_dataset(dataset, percent=100):          # make sure batch_size and no_worker is in scope from where this function is being called
    subset_size = (int)(len(dataset) * percent / 100)
    percent_dataset = random_split(dataset, [subset_size, len(dataset) - subset_size])[0]
    percent_dataloader = DataLoader(percent_dataset, batch_size=batch_size, num_workers=no_workers, shuffle=True, pin_memory=True)

    return percent_dataloader

training_percentages = list(np.arange(1, 21)) + [100]

if __name__ == '__main__':
    no_workers = 40
    results_directory = Path('results/supervised')
    models_directory = results_directory / 'models'
    evaluation_directory = results_directory / 'evaluations'
    continue_training = False

    train_dataset = Serbia(split='train', augementation_type=1, augmentation_count=1)
    batch_size = 256

    percentages = training_percentages
    if continue_training:
        saved_models = [(len(str(path)), str(path)) for path in models_directory.glob('*')]
        saved_models.sort(reverse=True)
        if len(saved_models) > 0:
            latest_saved_model = saved_models[0][1]
            latest = (float)(re.sub('.*_', '', latest_saved_model))
            percentages = percentages[percentages > latest]

    loss_func = torch.nn.BCEWithLogitsLoss()

    best_early_stop_epoch = 27                      # calculated by plotting validation loss over percent and epoch
    percent_tqdm = tqdm(percentages, position=1)
    for percent in percent_tqdm:
        percent = np.round(percent, 2)
        percent_tqdm.set_description(f"percentage of data for training: {percent}")
        model = Model()
        model = model.cuda()
        model = nn.DataParallel(model)
        optim = torch.optim.Adam(model.parameters())

        train_dataloader = get_percent_dataloader_from_dataset(train_dataset, percent)

        for epoch in range(1, best_early_stop_epoch+1):
            model.train()
            for x, l in tqdm(train_dataloader, desc=f'training epoch: {epoch}', position=2, leave=False):
                optim.zero_grad()
                x = x.cuda(non_blocking=True)[:, 0]
                l = l.cuda(non_blocking=True)
                pred = model(x)
                loss = loss_func(pred, l)

                loss.backward()
                optim.step()

        torch.save(model.state_dict(), str(models_directory / f'supervised_{"{:.2f}".format(percent)}'))
