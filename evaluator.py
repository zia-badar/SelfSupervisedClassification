import pickle as pk
from pathlib import Path
from typing import Type

import matplotlib.pyplot as pyplot
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchmetrics import Metric
from tqdm import tqdm


class Evaluator():

    def __init__(self, results_directory):
        self.results_directory = results_directory

    def evaluate(self, dataloaders: dict[str, DataLoader], metrics: list[Metric], model_class: Type[nn.Module], epoch_spacing = 100):
        self.dataloader_names = list(dataloaders.keys())
        dataloaders = dataloaders.values()
        self.metric_names = [m.name for m in metrics]

        models = list(map(lambda path: (len(str(path)), str(path)), list((self.results_directory / 'models').glob('*'))))
        models.sort()
        models = list(map(lambda t: t[1], models))
        self.selected_epochs = [i for i in range(epoch_spacing, len(models) + 1, epoch_spacing)]
        self.selected_epochs.insert(0, 1)
        models = [models[i - 1] for i in self.selected_epochs]

        self.metrics_evaluation = [[[0 for _ in range(len(models))] for _ in range(len(dataloaders))] for _ in range(len(metrics))]

        for i, model_path in enumerate(tqdm(models, position=0, desc='models')):
            model = model_class()
            model = model.cuda()
            model = nn.DataParallel(model)
            model.load_state_dict(torch.load(model_path))
            model.eval()

            for j, dataloader in enumerate(tqdm(dataloaders, position=1, leave=False, desc='dataloaders')):
                for metric in metrics:
                    metric.reset()

                for x, l in dataloader:
                    x = x.cuda(non_blocking=True)
                    l = l.cuda(non_blocking=True)

                    for metric in metrics:
                        metric(model(x), l)

                for metric, metric_evaluation in zip(metrics, self.metrics_evaluation):
                    metric_evaluation[j][i] = metric.compute()

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

    def plot(self):
        for metric_name, metric_evaluation in zip(self.metric_names, self.metrics_evaluation):
            pyplot.title(f'{metric_name}')
            for dl_name, dl_metric_evaluation in zip(self.dataloader_names, metric_evaluation):
                pyplot.plot(self.selected_epochs, dl_metric_evaluation, label=f'{dl_name}')

            pyplot.xticks(self.selected_epochs)
            pyplot.legend()
            pyplot.show()