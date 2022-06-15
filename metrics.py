import torch
from torchmetrics import Metric

class Accuracy(Metric):         # same as sklearn.metrics.accuracy_score for multi label
    name = 'Accuracy'
    full_state_update = False

    def __init__(self):
        super().__init__()
        self.add_state("correct", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        preds = preds.cuda(non_blocking = True).sigmoid().round()
        target = target.cuda(non_blocking = True)
        self.correct += torch.sum(torch.all(torch.eq(preds, target), dim=1))
        self.total += target.shape[0]

    def compute(self):
        return (self.correct.float() / self.total).item()

class Loss(Metric):
    name = 'Loss'
    full_state_update = False

    def __init__(self, loss_func):
        super().__init__()
        self.loss_func = loss_func
        self.add_state("loss", default=torch.tensor(0, dtype=torch.float32), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        self.loss += self.loss_func(preds, target)

    def compute(self):
        return self.loss.item()