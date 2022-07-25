import re
import sys
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

import utils
from evaluator import Evaluator
from metrics import Loss
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


if __name__ == '__main__':
    no_workers = 16
    start_percent = 5
    end_percent = 100
    diff_percent = 5
    results_directory = Path('results/supervised')
    models_directory = results_directory / 'models'
    continue_training = True

    train_dataset = Serbia(split='train')
    batch_size = get_batch_size(Model, train_dataset) - 5

    validation_dataset = Serbia(split='validation')
    validation_dataloader = DataLoader(validation_dataset, batch_size=batch_size, num_workers=no_workers, shuffle=True, drop_last=True, pin_memory=True)

    test_dataset = Serbia(split='test')
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, num_workers=no_workers, shuffle=True, drop_last=True, pin_memory=True)

    if continue_training:
        saved_models = [(len(str(path)), str(path)) for path in models_directory.glob('*')]
        saved_models.sort(reverse=True)
        if len(saved_models) > 0:
            latest_saved_model = saved_models[0][1]
            start_percent = (int)(re.sub('.*_', '', latest_saved_model)) + diff_percent

    # positive_weights, negative_weights, count = utils.calculate_class_weights(train_dataloader)
    # positive_weights = positive_weights.cuda(non_blocking=True)
    # negative_weights = negative_weights.cuda(non_blocking=True)

    # loss_func = DB_Loss(train_dataloader)
    loss_func = torch.nn.BCEWithLogitsLoss()

    percent_tqdm = tqdm(range(start_percent, end_percent+diff_percent, diff_percent), position=1)
    for percent in percent_tqdm:
        percent_tqdm.set_description(f"percentage of data for training: {percent}")
        model = Model()
        model = model.cuda()
        model = nn.DataParallel(model)
        optim = torch.optim.Adam(model.parameters())

        last_loss = sys.float_info.max
        epoch = 1
        subset_size = (int)(len(train_dataset) * percent / 100)
        train_percent_dataset = random_split(train_dataset, [subset_size, len(train_dataset) - subset_size])[0]
        train_dataloader = DataLoader(train_percent_dataset, batch_size=batch_size, num_workers=no_workers,
                                      shuffle=True, pin_memory=True)
        bar = tqdm(train_dataloader, desc=f'training epoch: {epoch}', position=2, leave=False)
        early_stop = False
        while not early_stop:
            model.train()
            for x, l in bar:
                optim.zero_grad()
                x = x.cuda(non_blocking=True)[:, 0]
                l = l.cuda(non_blocking=True)
                pred = model(x)
                loss = loss_func(pred, l)

                loss.backward()
                optim.step()

            if epoch % 5 == 0:
                with torch.no_grad():
                    loss = 0
                    for x, l in validation_dataloader:
                        x = x[:, 0].cuda()
                        l = l.cuda()
                        loss += loss_func(model(x), l)

                early_stop = (last_loss - loss < 0)
                last_loss = loss

            epoch += 1

        torch.save(model.state_dict(), str(models_directory / f'trained_supervised_model_epoch_{percent}'))

    evaluator = Evaluator(results_directory)
    dataloaders = {'validation': validation_dataloader}
    metrics = [Loss(loss_func).cuda()]
    evaluator.evaluate(dataloaders, metrics, Model)
    evaluator.save()
    # evaluator = evaluator.load(Path('results/supervised/evaluations/evaluation_1'))
    evaluator.plot()
