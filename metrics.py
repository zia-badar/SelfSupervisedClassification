import torch
from torchmetrics import Metric

from patch import Patch

class MetricAggregator(Metric):             # make sure this metric is reset when all metrics dependent on it are done using it
    name = 'aggregator metric'
    full_state_update = False

    def __init__(self):
        super().__init__()
        self.add_state('tp', default=torch.zeros(Patch.classes, dtype=torch.int))
        self.add_state('fp', default=torch.zeros(Patch.classes, dtype=torch.int))
        self.add_state('tn', default=torch.zeros(Patch.classes, dtype=torch.int))
        self.add_state('fn', default=torch.zeros(Patch.classes, dtype=torch.int))

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        preds = preds.cuda(non_blocking=True).sigmoid().round()
        target = target.cuda(non_blocking=True)

        preds = preds.type(torch.int)
        target = target.type(torch.int)

        tp = torch.sum(preds * target, dim=0)
        fp = torch.sum((preds == True) * (target == False), dim=0)
        tn = torch.sum((preds == False) * (target == False), dim=0)
        fn = torch.sum((preds == False) * (target == True), dim=0)

        self.tp += tp
        self.fp += fp
        self.tn += tn
        self.fn += fn

    def compute(self):
        return self.tp, self.fp, self.tn, self.fn


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

    def __init__(self, metricAggregator: MetricAggregator, type='micro'):
        super().__init__()
        self.metricAggregator = metricAggregator
        self.type = type
        self.name = 'Precision(TP/(TP+FP)) ' + self.type

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        pass    # compute metric aggreator only


    def compute(self):
        tp, fp, tn, fn = self.metricAggregator.compute()
        numerator = tp
        denominator = tp+fp
        result = 0
        if self.type == 'micro':
            result = torch.nan_to_num(numerator.sum() / denominator.sum(), nan=0)
        elif self.type == 'macro':
            result = (torch.nan_to_num(numerator / denominator, nan=0).sum() / numerator.shape[0])
        elif self.type == 'per class':
            result = torch.nan_to_num(numerator / denominator, nan=0)
        return result

class Recall(Metric):
    full_state_update = False

    def __init__(self, metricAggregator: MetricAggregator, type='micro'):
        super().__init__()
        self.metricAggregator = metricAggregator
        self.type = type
        self.name = 'Recall(TP/(TP+FN)) ' + self.type

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        pass    # compute metric aggreator only

    def compute(self):
        tp, fp, tn, fn = self.metricAggregator.compute()
        numerator = tp
        denominator = tp+fp
        result = 0
        if self.type == 'micro':
            result = torch.nan_to_num(numerator.sum() / denominator.sum(), nan=0)
        elif self.type == 'macro':
            result = (torch.nan_to_num(numerator / denominator, nan=0).sum() / numerator.shape[0])
        elif self.type == 'per class':
            result = torch.nan_to_num(numerator / denominator, nan=0)
        return result

class F1_Score(Metric):
    full_state_update = False

    def __init__(self, metricAggregator: MetricAggregator, type='micro'):
        super().__init__()
        self.metricAggregator = metricAggregator
        self.type = type
        self.name = 'F1_Score(2*precision*recall/(precision+recall)) ' + self.type
        self.precision = Precision(self.metricAggregator, self.type).cuda()
        self.recall = Recall(self.metricAggregator, self.type).cuda()

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        pass    # compute metric aggreator only

    def compute(self):
        precision = self.precision.compute()
        recall = self.recall.compute()
        return torch.nan_to_num(2 * (precision * recall) / (precision + recall), nan=0)

    def reset(self):
        self.precision = Precision(self.metricAggregator, self.type).cuda()
        self.recall = Recall(self.metricAggregator, self.type).cuda()

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