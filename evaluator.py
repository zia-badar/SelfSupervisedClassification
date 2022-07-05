import pickle as pk
from pathlib import Path
from typing import Type

import matplotlib.pyplot as pyplot
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchmetrics import Metric
from tqdm import tqdm
import numpy as np


class Evaluator():

    def __init__(self, results_directory):
        self.results_directory = results_directory

    def evaluate(self, dataloaders: dict[str, DataLoader], metrics: list[Metric], model_class: Type[nn.Module], model_wrapper, epoch_spacing = 1, percentage_diff=1, max_percentage=1):
        self.dataloader_names = list(dataloaders.keys())
        dataloaders = dataloaders.values()
        self.metric_names = [m.name for m in metrics]

        models = list(map(lambda path: (len(str(path)), str(path)), list((self.results_directory / 'models').glob('*'))))
        models.sort()
        models = list(map(lambda t: t[1], models))
        self.selected_epochs = [i for i in range(epoch_spacing, len(models) + 1, epoch_spacing)]
        models = [models[i - 1] for i in self.selected_epochs]
        self.selected_percentage = [np.round(i, 2) for i in np.arange(percentage_diff, max_percentage, percentage_diff)]
        self.selected_percentage.append(max_percentage)

        if len(self.selected_epochs) > 1 and len(self.selected_percentage) > 1:
            print('error: one of models or percentage should be of size 1')
            exit(-1)

        self.metrics_evaluation = [[[[0 for _ in range(len(models))] for _ in range(len(self.selected_percentage))] for _ in range(len(dataloaders))] for _ in range(len(metrics))]

        with torch.no_grad():
            for i, model_path in enumerate(tqdm(models, position=0, desc='models')):
                model = model_class()
                model = model.cuda()
                model = nn.DataParallel(model)
                model.load_state_dict(torch.load(model_path))
                model.eval()
                model = model_wrapper(model)

                for j, percent in enumerate(tqdm(self.selected_percentage, position=1, leave=False, desc='percentage')):
                    model.train_subset_ratio = percent

                    for k, dataloader in enumerate(tqdm(dataloaders, position=2, leave=False, desc='dataloaders')):
                        for metric in metrics:
                            metric.reset()

                        for x, l in tqdm(dataloader, position=3, leave=False, desc=f'{self.dataloader_names[k]}'):
                            x = x[:, 0]

                            for metric in metrics:
                                metric.update(model(x), l)

                        for metric, metric_evaluation in zip(metrics, self.metrics_evaluation):
                            metric_evaluation[k][j][i] = metric.compute()

    def get_latest_evaluation_path(self):
        evaluations_directory = self.results_directory / 'evaluations'
        evaluations = list(map(lambda path: (len(str(path)), str(path)), list(evaluations_directory.glob('*'))))
        evaluations.sort()
        evaluations = list(map(lambda t: str(t[1]), evaluations))
        return None if len(evaluations) == 0 else  evaluations[-1]

    def save(self):
        latest_evaluation_name = self.get_latest_evaluation_path()
        new_evaluation_name = 'evaluation_' + str(1 if latest_evaluation_name == None else str(int(latest_evaluation_name.rsplit('_', 1)[1]) + 1))

        with open(str( self.results_directory / 'evaluations' / new_evaluation_name), 'wb') as fout:
            pk.dump(self, fout)

    def load(self, n=-1):
        if n != -1:
            evaluation_path = self.results_directory / 'evaluations' / ('evaluation_' + str(n))
        else:
            evaluation_path = Path(self.get_latest_evaluation_path())

        if not evaluation_path.exists():
            print(f'evaluation path: {str(evaluation_path)} does not exists')
            exit(-1)

        with open(str(evaluation_path), 'rb') as fin:
            return pk.load(fin)

    def load(self, path:Path):
        with open(str(path), 'rb') as fin:
            return pk.load(fin)

    def plot(self):
        for metric_name, metric_evaluation in zip(self.metric_names, self.metrics_evaluation):
            pyplot.figure(figsize=(30, 30))
            pyplot.title(f'{metric_name}')
            for dl_name, dl_metric_evaluation in zip(self.dataloader_names, metric_evaluation):
                pyplot.plot(self.selected_epochs if len(self.selected_epochs) > 1 else self.selected_percentage, np.array(dl_metric_evaluation).reshape(-1), label=f'{dl_name}')

            pyplot.xticks(self.selected_epochs if len(self.selected_epochs) > 1 else self.selected_percentage)
            pyplot.legend()
            pyplot.show()

    def __repr__(self):
        for metric_name, metric_evaluation in zip(self.metric_names, self.metrics_evaluation):
            for dl_name, dl_metric_evaluation in zip(self.dataloader_names, metric_evaluation):
                print(f'metric: {metric_name}, dataloader: {dl_name}, value: {dl_metric_evaluation}')