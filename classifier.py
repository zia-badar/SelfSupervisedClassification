from typing import Type

import torch

from loss import temperature
from patch import Patch


class ContrastiveWeightedKNN():
    def __init__(self, contrastive_model: Type[torch.nn.Module], train, train_subset_ratio=.5, k=10):
        self.contrastive_model = contrastive_model
        self.train_x, self.train_y = train
        self.train_subset_ratio = train_subset_ratio
        self.k = k

    def shuffle(self):
        self.train_x = self.train_x[torch.randperm(self.train_x.shape[0])]
        self.train_y = self.train_y[torch.randperm(self.train_y.shape[0])]

    def w_knn(self, test_x):
        train_x = self.train_x[:(int)(self.train_subset_ratio * len(self.train_x))]
        train_y = self.train_y[:(int)(self.train_subset_ratio * len(self.train_y))]

        _c = Patch.classes
        _cn = 2
        weights = torch.mm(test_x, train_x.t())  # |test| x |train|
        kweights, k_train_indices = torch.topk(weights, self.k, dim=-1)  # |test| x k
        kweights = (kweights / temperature).exp()
        klabels = train_y[k_train_indices, :]  # |test| x k x c
        klabels = torch.nn.functional.one_hot(klabels.long(), num_classes=_cn)  # |test| x k x c x _cn
        voting = kweights.unsqueeze(-1).unsqueeze(-1) * klabels
        voting = torch.sum(voting, dim=1)
        voting = torch.argsort(voting, dim=-1, descending=True)

        return voting[:, :, 0]

    def __call__(self, x):
        x = x.cuda(non_blocking=True)
        features, _ = self.contrastive_model(x)
        prediction = self.w_knn(features)
        return prediction
