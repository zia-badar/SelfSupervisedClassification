import torch
from torchmetrics import Metric

from patch import Patch


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
        return self.sum / self.total

class Precision(Metric):
    full_state_update = False

    def __init__(self, type='micro'):
        super().__init__()
        self.add_state("numerator", default=torch.zeros(Patch.classes, dtype=torch.float32), dist_reduce_fx="sum")
        self.add_state("denominator", default=torch.zeros(Patch.classes, dtype=torch.float32), dist_reduce_fx="sum")
        self.type = type
        self.name = 'Precision(TP/(TP+FP)) ' + self.type

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        preds = preds.cuda(non_blocking=True).sigmoid().round()
        target = target.cuda(non_blocking=True)

        tp = torch.sum(preds * target, dim=0)
        fp = torch.sum((preds == True) * (target == False), dim=0)

        self.numerator += tp
        self.denominator += tp+fp

    def compute(self):
        result = 0
        if self.type == 'micro':
            result = torch.nan_to_num(self.numerator.sum() / self.denominator.sum(), nan=0)
        elif self.type == 'macro':
            result = (torch.nan_to_num(self.numerator / self.denominator, nan=0).sum() / self.numerator.shape[0])
        elif self.type == 'per class':
            result = torch.nan_to_num(self.numerator / self.denominator, nan=0)
        return result

class Recall(Metric):
    full_state_update = False

    def __init__(self, type='micro'):
        super().__init__()
        self.add_state("numerator", default=torch.zeros(Patch.classes, dtype=torch.float32), dist_reduce_fx="sum")
        self.add_state("denominator", default=torch.zeros(Patch.classes, dtype=torch.float32), dist_reduce_fx="sum")
        self.type = type
        self.name = 'Recall(TP/(TP+FN)) ' + self.type

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        preds = preds.cuda(non_blocking=True).sigmoid().round()
        target = target.cuda(non_blocking=True)

        tp = torch.sum(preds * target, dim=0)
        fn = torch.sum((preds == False) * (target == True), dim=0)

        self.numerator += tp
        self.denominator += tp+fn

    def compute(self):
        result = 0
        if self.type == 'micro':
            result = torch.nan_to_num(self.numerator.sum() / self.denominator.sum(), nan=0)
        elif self.type == 'macro':
            result = (torch.nan_to_num(self.numerator / self.denominator, nan=0).sum() / self.numerator.shape[0])
        elif self.type == 'per class':
            result = torch.nan_to_num(self.numerator / self.denominator, nan=0)
        return result

class F1_Score(Metric):
    full_state_update = False

    def __init__(self, type='micro'):
        super().__init__()
        self.type = type
        self.name = 'F1_Score(2*precision*recall/(precision+recall)) ' + self.type
        self.precision = Precision(self.type).cuda()
        self.recall = Recall(self.type).cuda()

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        self.precision.update(preds, target)
        self.recall.update(preds, target)

    def compute(self):
        precision = self.precision.compute()
        recall = self.recall.compute()
        return torch.nan_to_num(2 * (precision * recall) / (precision + recall), nan=0)

    def reset(self):
        self.precision = Precision(self.type).cuda()
        self.recall = Recall(self.type).cuda()

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