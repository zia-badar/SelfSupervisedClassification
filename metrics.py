import sklearn.metrics
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

class CustomAccuracy(Metric):         # as recommended by lars, TP/(TP+FP+FN)
    name = 'TP/(TP+FP+FN) Metric'
    full_state_update = False

    def __init__(self):
        super().__init__()
        self.add_state("sum", default=torch.tensor(0, dtype=torch.float32), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        preds = preds.cuda(non_blocking=True).sigmoid().round()
        target = target.cuda(non_blocking=True)
        tp = torch.sum(preds * target, dim=1)
        fp = torch.sum((preds == True) * (target == False), dim=1)
        fn = torch.sum((preds == False) * (target == True), dim=1)
        m = tp / (tp+fp+fn)
        self.sum += m.sum()
        self.total += target.shape[0]

    def compute(self):
        return (self.sum.float() / self.total).item()


class Loss(Metric):
    name = 'Loss'
    full_state_update = False

    def __init__(self, loss_func):
        super().__init__()
        self.loss_func = loss_func
        self.add_state("loss", default=torch.tensor(0, dtype=torch.float32), dist_reduce_fx="sum")

    def update(self, net, batch):
        self.loss += self.loss_func(net(batch[0]), batch[1])

    def compute(self):
        return self.loss.item()