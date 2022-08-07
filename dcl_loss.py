from typing import Type

import torch

from patch import Patch

temperature = .5

class DCL_loss():
    def __init__(self, tau_plus):
        self.tau_plus = tau_plus

    def __call__(self, model, batch):
        x, _ = batch
        batch, aug = x.shape[:2]
        x = x.view((batch*aug,) + x.shape[2:])
        _, x = model(x)    # (batch*aug) x dim
        f = torch.exp((x @ x.t())/temperature).view(batch, aug, batch, aug)      # batch x aug x batch x aug
        N = (batch-1) * aug
        f_x_u_filter = torch.ones((batch, aug, batch, aug), dtype=torch.bool).cuda()
        for i in range(batch):
            f_x_u_filter[i, :, i, :] = 0
        f_x_u = torch.masked_select(f, f_x_u_filter).view(batch, aug, batch-1, aug)
        M = aug - 1
        f_x_v_filter = torch.logical_not(f_x_u_filter)
        for i in range(aug):
            f_x_v_filter[:, i, :, i] = 0
        f_x_v = torch.masked_select(f, f_x_v_filter).view(batch, aug, aug-1)
        f_x_v_p = f_x_v

        m = N*torch.exp(torch.tensor([-1/temperature])).cuda()
        Ng = torch.clip((1/(1-self.tau_plus)) * (f_x_u.view(batch, aug, -1).sum(-1) - ((N*self.tau_plus)/M)*f_x_v.sum(-1) ), min=m)[:, :, None]    # batch x aug x 1

        loss = torch.mean(-torch.log(f_x_v_p/(f_x_v_p + Ng)))

        return loss

class DCL_classifier():
    def __init__(self, dcl_model: Type[torch.nn.Module], train, train_subset_ratio=.5, k=10):
        self.dcl_model = dcl_model
        self.train_x, self.train_y = train
        self.train_subset_ratio = train_subset_ratio
        self.k = k

    def w_knn(self, test):
        train_x = self.train_x[:(int)(self.train_subset_ratio * len(self.train_x))]
        train_y = self.train_y[:(int)(self.train_subset_ratio * len(self.train_y))]
        test_x, test_y = test

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
        dcl_features, _ = self.dcl_model(x)
        prediction = self.w_knn((dcl_features, None))
        return prediction