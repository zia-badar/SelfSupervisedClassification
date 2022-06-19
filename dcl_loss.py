import numpy as np
import torch



class DCL_loss():
    def __init__(self, temperature, debiased, tau_plus, batch_size):
        self.temperature = temperature
        self.debiased = debiased
        self.tau_plus = tau_plus
        self.batch_size = batch_size

    def get_negative_mask(batch_size):
        negative_mask = torch.ones((batch_size, 2 * batch_size), dtype=bool)
        for i in range(batch_size):
            negative_mask[i, i] = 0
            negative_mask[i, i + batch_size] = 0

        negative_mask = torch.cat((negative_mask, negative_mask), 0)
        return negative_mask

    def __call__(self, net, pos_1, pos_2, l):
        pos_1, pos_2 = pos_1.cuda(non_blocking=True), pos_2.cuda(non_blocking=True)
        feature_1, out_1 = net(pos_1)
        feature_2, out_2 = net(pos_2)

        # neg score
        out = torch.cat([out_1, out_2], dim=0)
        neg = torch.exp(torch.mm(out, out.t().contiguous()) / self.temperature)
        mask = DCL_loss.get_negative_mask(self.batch_size).cuda()
        neg = neg.masked_select(mask).view(2 * self.batch_size, -1)

        # pos score
        pos = torch.exp(torch.sum(out_1 * out_2, dim=-1) / self.temperature)
        pos = torch.cat([pos, pos], dim=0)

        # estimator g()
        if self.debiased:
            N = self.batch_size * 2 - 2
            Ng = (-self.tau_plus * N * pos + neg.sum(dim = -1)) / (1 - self.tau_plus)
            # constrain (optional)
            Ng = torch.clamp(Ng, min = N * np.e**(-1 / self.temperature))
        else:
            Ng = neg.sum(dim=-1)

        # contrastive loss
        loss = (- torch.log(pos / (pos + Ng) )).mean()

        return loss