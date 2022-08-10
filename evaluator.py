import collections
import pickle as pk
from pathlib import Path
from typing import Type

import matplotlib.pyplot as pyplot
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchmetrics import Metric
from tqdm import tqdm

from classifier import ContrastiveWeightedKNN
from metrics import Loss
from patch import Patch


class Evaluator():

    def __init__(self, evaluations_directory, models_directory):
        self.evaluations_directory = evaluations_directory
        self.models_directory = models_directory

    def evaluate(self, dataloaders: dict[str, DataLoader], metrics: list[Metric], model_class: Type[nn.Module],
                 model_wrapper=None, percentages=[None]):
        self.dataloader_names = list(dataloaders.keys())
        dataloaders = dataloaders.values()
        self.metric_names = [m.name for m in metrics]

        models = list(map(lambda path: (len(str(path)), str(path)), list((self.models_directory).glob('*'))))
        models.sort()
        models = list(map(lambda t: t[1], models))
        if len(models) > 1:  # sorting models
            dict = {}
            for m in models:
                dict[(float)(m.rsplit('_', 1)[1])] = m
            models = list(collections.OrderedDict(sorted(dict.items())).values())

        self.x = [(float)(model_name.rsplit('_', 1)[1]) for model_name in models] if len(models) > 1 else percentages

        if len(models) > 1 and len(percentages) > 1:
            print('error: one of models or percentage should be of size 1')
            exit(-1)

        self.metrics_evaluation = [
            [[[0 for _ in range(len(models))] for _ in range(len(percentages))] for _ in range(len(dataloaders))] for _
            in range(len(metrics))]

        with torch.no_grad():
            for i, model_path in enumerate(tqdm(models, position=0, desc='models')):
                model = model_class()
                model = model.cuda()
                model = nn.DataParallel(model)
                model.load_state_dict(torch.load(model_path))
                model.eval()
                model = model_wrapper(model) if model_wrapper != None else model

                for j, percent in enumerate(tqdm(percentages, position=1, leave=False, desc='percentage')):
                    if isinstance(model, ContrastiveWeightedKNN):
                        model.train_subset_ratio = percent / 100.

                    for k, dataloader in enumerate(tqdm(dataloaders, position=2, leave=False, desc='dataloaders')):
                        for metric in metrics:
                            metric.reset()

                        for x, l in tqdm(dataloader, position=3, leave=False, desc=f'{self.dataloader_names[k]}'):
                            x = x[:, 0]
                            x = x.cuda()
                            l = l.cuda()

                            for metric in metrics:
                                if isinstance(metric, Loss):
                                    metric.update(model, (x, l))
                                else:
                                    metric.update(model(x), l)

                        for metric, metric_evaluation in zip(metrics, self.metrics_evaluation):
                            metric_evaluation[k][j][i] = metric.compute()

    def get_latest_evaluation_path(self):
        evaluations = list(map(lambda path: (len(str(path)), str(path)), list(self.evaluations_directory.glob('*'))))
        evaluations.sort()
        evaluations = list(map(lambda t: str(t[1]), evaluations))
        return None if len(evaluations) == 0 else evaluations[-1]

    def save(self, name=None):
        if name == None:
            latest_evaluation_name = self.get_latest_evaluation_path()
            new_evaluation_name = 'evaluation_' + str(
                1 if latest_evaluation_name == None else str(int(latest_evaluation_name.rsplit('_', 1)[1]) + 1))
        else:
            new_evaluation_name = name

        with open(str(self.evaluations_directory / new_evaluation_name), 'wb') as fout:
            pk.dump(self, fout)

    def load(self, n=-1):
        if n != -1:
            evaluation_path = self.evaluations_directory / ('evaluation_' + str(n))
        else:
            evaluation_path = Path(self.get_latest_evaluation_path())

        if not evaluation_path.exists():
            print(f'evaluation path: {str(evaluation_path)} does not exists')
            exit(-1)

        with open(str(evaluation_path), 'rb') as fin:
            return pk.load(fin)

    def load(path: Path):
        with open(str(path), 'rb') as fin:
            return pk.load(fin)

    def flat_list(listoflist):
        return [item for list in listoflist for item in list]

    def plot(evaluator_names, evaluators,
             critical_evaluator_name=None):  # this function is only intended for plotting required plots, not for any kind of reuse, because plot requirements always changes and useless to write plot generalize code
        metric_names = evaluators[0].metric_names

        class_percentage = 100 * torch.tensor([15790., 100394., 1197., 73411., 1037., 53534., 86569., 6182.,
                                               35349., 11620., 67260., 39114., 91930., 8438., 6663., 50981.,
                                               15862., 77593., 38783.]) / 269695.

        for i, metric_name in enumerate(metric_names):
            if metric_name.endswith('per class'):
                pyplot.rcParams.update({'font.size': 20})
                fig, axis = pyplot.subplots(5, 4, figsize=(30, 20), dpi=100)
            else:
                pyplot.rcParams.update({'font.size': 30})
                fig, axes = pyplot.subplots(1, 2, figsize=(40, 20), gridspec_kw={'width_ratios': [9, 1]}, sharey=True,
                                            dpi=100)

            for evaluator_name, evaluator in zip(evaluator_names, evaluators):
                for dl_name, dl_metric_evaluation in zip(evaluator.dataloader_names, evaluator.metrics_evaluation[i]):
                    if metric_name.endswith('per class'):
                        eval = np.array(torch.stack(Evaluator.flat_list(dl_metric_evaluation)).cpu())
                        classnames = list(Patch._19_label_to_index.keys())
                        for c in range(Patch.classes):
                            current_axis = axis[int(c / 4), c % 4]
                            current_axis.plot(evaluator.x[:-1], eval[:, c][:-1], label=f'{evaluator_name}, {dl_name}',
                                              alpha=0.7)
                            current_axis.set_title(
                                (classnames[c] if len(classnames[c]) < 15 else (classnames[c][
                                                                                :15]) + "...") + f' {"{:.1f}".format(class_percentage[c].item())}% samples',
                                color=(
                                    '#cc3300' if evaluator_name == critical_evaluator_name and eval[:,
                                                                                               c].sum() < 0.1 else '#339900'))
                            current_axis.grid(True, alpha=0.3)
                    else:
                        axes[0].plot(evaluator.x, np.array(torch.tensor(dl_metric_evaluation)).reshape(-1),
                                     label=f'{evaluator_name}, {dl_name}', alpha=0.7)
                        axes[1].plot(evaluator.x, np.array(torch.tensor(dl_metric_evaluation)).reshape(-1),
                                     label=f'{evaluator_name}, {dl_name}', alpha=0.7)
                        axes[0].set_xlim(0, evaluator.x[-2])
                        axes[1].set_xlim(evaluator.x[-2], evaluator.x[-1])
                        axes[0].set_ylim(0, 1)
                        axes[1].set_ylim(0, 1)
                        axes[0].set_xticks(np.arange(1, evaluator.x[-2] + 1))
                        axes[1].set_xticks(evaluator.x[-2:])
                        axes[0].set_yticks(np.arange(0, 1.1, 0.1))
                        axes[0].spines['right'].set_visible(False)
                        axes[1].spines['left'].set_visible(False)
                        axes[1].yaxis.tick_right()
                        axes[0].grid(True, alpha=0.3)
                        axes[1].grid(True, alpha=0.3)
                        d = .03
                        kwargs = dict(transform=axes[0].transAxes, color='k', clip_on=False)
                        axes[0].plot((1 - d / 9, 1 + d / 9), (1 - d, 1 + d), **kwargs)
                        axes[0].plot((1 - d / 9, 1 + d / 9), (-d, +d), **kwargs)

                        kwargs.update(transform=axes[1].transAxes)
                        axes[1].plot((-d, +d), (1 - d, 1 + d), **kwargs)
                        axes[1].plot((-d, +d), (- d, + d), **kwargs)

            if metric_name.endswith('per class'):
                pyplot.setp(axis, xticks=np.arange(0, evaluator.x[-2] + 1, 5), yticks=np.arange(0.0, 1.1, 0.5),
                            ylim=[-0.1, 1.2])
                fig.suptitle(metric_name)
                fig.supxlabel('percentage of labelled data used')
                fig.supylabel(metric_name, x=0.01)
                lines, labels = axis[0, 0].get_legend_handles_labels()
                fig.legend(lines, labels, loc=(0.8, 0.06))
                axis[-1, -1].axis('off')
            else:
                fig.suptitle(metric_name)
                fig.supxlabel('percentage of labelled data used')
                fig.supylabel(metric_name, x=0.01)
                lines, labels = axes[0].get_legend_handles_labels()
                fig.legend(lines, labels, loc=(0.06, 0.8), framealpha=0.7)

            fig.tight_layout()
            pyplot.show()

    def __repr__(self):
        for metric_name, metric_evaluation in zip(self.metric_names, self.metrics_evaluation):
            for dl_name, dl_metric_evaluation in zip(self.dataloader_names, metric_evaluation):
                print(f'metric: {metric_name}, dataloader: {dl_name}, value: {dl_metric_evaluation}')

        return ""
