import torch

from patch import Patch

temperature = .5


# Debaised Contrastive Loss from https://arxiv.org/abs/2007.00224
class DebaisedContrastiveLoss():
    def __init__(self, tau_plus):
        self.tau_plus = tau_plus

    def __call__(self, model, batch):
        x, _ = batch
        batch, aug = x.shape[:2]
        x = x.view((batch * aug,) + x.shape[2:])
        _, x = model(x)  # (batch*aug) x dim
        f = torch.exp((x @ x.t()) / temperature).view(batch, aug, batch, aug)  # batch x aug x batch x aug
        N = (batch - 1) * aug
        f_x_u_filter = torch.ones((batch, aug, batch, aug), dtype=torch.bool).cuda()
        for i in range(batch):
            f_x_u_filter[i, :, i, :] = 0
        f_x_u = torch.masked_select(f, f_x_u_filter).view(batch, aug, batch - 1, aug)
        M = aug - 1
        f_x_v_filter = torch.logical_not(f_x_u_filter)
        for i in range(aug):
            f_x_v_filter[:, i, :, i] = 0
        f_x_v = torch.masked_select(f, f_x_v_filter).view(batch, aug, aug - 1)
        f_x_v_p = f_x_v

        m = N * torch.exp(torch.tensor([-1 / temperature])).cuda()
        Ng = torch.clip((1 / (1 - self.tau_plus)) * (
                    f_x_u.view(batch, aug, -1).sum(-1) - ((N * self.tau_plus) / M) * f_x_v.sum(-1)), min=m)[:, :,
             None]  # batch x aug x 1

        loss = torch.mean(-torch.log(f_x_v_p / (f_x_v_p + Ng)))

        return loss


# https://arxiv.org/abs/2007.09654 's implementation, did not provide much advantage and not used
class DistributedBalanceLoss():
    def __init__(self, dataloader):
        self.C = Patch.classes
        self.n_i = torch.zeros(self.C).cuda(non_blocking=True)
        self.n = torch.zeros(1).cuda(non_blocking=True)
        for _, l in dataloader:
            l = l.cuda(non_blocking=True)
            self.n_i += torch.sum(l, dim=0)
            self.n += l.shape[0]

        self.kappa = 0.05
        self._lambda = 5

        self.v_i = self.kappa * torch.log(1 / (self.n_i / self.n) - 1)
        self.v_i = torch.nan_to_num(self.v_i, posinf=0)

    def p_i(self, x):
        N_i = self.n_i.expand(x.shape[0], self.C)
        return (1 / Patch.classes) * torch.sum(torch.where(x == 1, 1 / N_i, torch.zeros_like(N_i)), dim=1)

    def r_i_k(self, x):
        r = ((1 / self.C) * (1 / self.n_i)) / self.p_i(x)[:, None]
        r = torch.nan_to_num(r, posinf=0)  # some class have zero samples
        alpha = 0.1
        beta = 10.0
        gamma = 0.2
        return alpha + 1 / (1 + torch.exp(-beta * (r - gamma)))

    def loss(self, x, y):
        x = x - self.v_i
        t = torch.nn.ReLU()(x)
        return torch.mean((y * (t - x + ((x - t).exp() + (-t).exp()).log()) + (1 / self._lambda) * (1 - y) * (
                self._lambda * t + (
                    (-self._lambda * t).exp() + (self._lambda * x - self._lambda * t).exp()).log())) * self.r_i_k(y))

    def __call__(self, x, y):
        return self.loss(x, y)
