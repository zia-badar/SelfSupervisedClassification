from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

import supervised
from classifier import ContrastiveWeightedKNN
from dataset import BigEarthDataset
from evaluator import Evaluator
from metrics import CustomAccuracy, Precision, Recall, F1_Score, MetricAggregator
from model import SupervisedModel, SelfSupervisedModel


def get_batch_size(modelCLass, dataset):
    batch_size = 1024

    model = modelCLass().cuda()
    model = nn.DataParallel(model)

    def possible(new_batch_size):
        try:
            def dataloader_wrapper(dataloader):
                max_iter = 1
                for i, batch in enumerate(dataloader):
                    if i >= max_iter:
                        break
                    yield batch

            for x, l in dataloader_wrapper(
                    DataLoader(dataset, new_batch_size, shuffle=True, pin_memory=True, drop_last=True)):
                with torch.no_grad():
                    model.eval()
                    model(x[:, 0].cuda())
                break
            return True
        except RuntimeError as ex:
            return False

    min, max, = 1, batch_size
    best_batch_size = min
    while min <= max:
        mid = (int)((min + max) / 2)
        if possible(mid):
            best_batch_size = mid
            min = mid + 1
        else:
            max = mid - 1

        torch.cuda.empty_cache()  # avoid cluter of previous iterations on gpu

    return best_batch_size


if __name__ == '__main__':

    no_workers = 40

    metricAggregator = MetricAggregator().cuda()
    metrics = [
        metricAggregator,
        CustomAccuracy().cuda(),
        Precision(metricAggregator, 'micro').cuda(),
        Precision(metricAggregator, 'macro').cuda(),
        Precision(metricAggregator, 'per class').cuda(),
        Recall(metricAggregator, 'micro').cuda(),
        Recall(metricAggregator, 'macro').cuda(),
        Recall(metricAggregator, 'per class').cuda(),
        F1_Score(metricAggregator, 'micro').cuda(),
        F1_Score(metricAggregator, 'macro').cuda(),
        F1_Score(metricAggregator, 'per class').cuda()
    ]

    # supervised evaluation

    train_dataset = BigEarthDataset(split='train', augementation_type=1, augmentation_count=1)
    batch_size = get_batch_size(SupervisedModel, train_dataset)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, num_workers=no_workers, shuffle=True,
                                  drop_last=True, pin_memory=True)

    test_dataset = BigEarthDataset(split='test', augementation_type=1, augmentation_count=1)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, num_workers=no_workers, shuffle=True,
                                 drop_last=True, pin_memory=True)

    supervised_result_directory = Path('results/supervised')
    supervised_evaluation_directory = supervised_result_directory / 'evaluations'
    supervised_model_directory = supervised_result_directory / 'models'
    supervised_evaluation_name = 'supervised'

    evaluator = Evaluator(supervised_evaluation_directory, supervised_model_directory)
    dataloaders = {'test': test_dataloader}
    evaluator.evaluate(dataloaders, metrics, SupervisedModel)
    evaluator.save(supervised_evaluation_name)

    # self supervised evaluation

    train_dataset = BigEarthDataset(split='train', augementation_type=2, augmentation_count=1)
    batch_size = get_batch_size(SelfSupervisedModel, train_dataset)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, num_workers=no_workers, shuffle=True,
                                  drop_last=True, pin_memory=True)

    test_dataset = BigEarthDataset(split='test', augementation_type=2, augmentation_count=1)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, num_workers=no_workers, shuffle=True,
                                 drop_last=True, pin_memory=True)

    self_supervised_result_directory = Path('results/self_supervised')
    self_supervised_evaluation_directory = self_supervised_result_directory / 'evaluations'
    self_supervised_model_directory = self_supervised_result_directory / 'models'
    self_supervised_evaluation_name = 'self-supervised'
    self_supervised_random_evaluation_name = 'self-supervised_random'

    evaluator = Evaluator(self_supervised_evaluation_directory, self_supervised_model_directory)
    dataloaders = {'test': test_dataloader}


    def model_wrapper(m):
        train_x = []
        train_y = []
        with torch.no_grad():
            for x, l in tqdm(train_dataloader):
                x = x[:, 0]
                f, _ = m(x)
                train_x.append(f)
                train_y.append(l)

        train_x = torch.cat(train_x, dim=0)
        train_y = torch.cat(train_y, dim=0).cuda()

        classifier = ContrastiveWeightedKNN(m, (train_x, train_y))
        return classifier


    evaluator.evaluate(dataloaders, metrics, SelfSupervisedModel, model_wrapper,
                       percentages=supervised.training_percentages)
    evaluator.save(self_supervised_evaluation_name)

    # plotting
    Evaluator.plot(
        ['supervised', 'contrastive repr + wknn', 'random repr + wknn'],
        [Evaluator.load(supervised_evaluation_directory / supervised_evaluation_name),
         Evaluator.load(self_supervised_evaluation_directory / self_supervised_evaluation_name),
         Evaluator.load(self_supervised_evaluation_directory / self_supervised_random_evaluation_name)])
