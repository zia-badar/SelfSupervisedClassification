from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

import DCL.model as dcl_model
import supervised
from dcl_loss import DCL_classifier
from evaluator import Evaluator
from metrics import CustomAccuracy, Precision, Recall, F1_Score
from model import Model
from serbia import Serbia

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
            for x, l in dataloader_wrapper(DataLoader(dataset, new_batch_size, shuffle=True, pin_memory=True, drop_last=True)):
                model(x[:, 0].cuda())
                break
            return True
        except RuntimeError as ex:
            return False

    min, max, = 1, batch_size
    best_batch_size = min
    while min <= max:
        mid = (int)((min + max)/2)
        if possible(mid):
            best_batch_size = mid
            min = mid + 1
        else:
            max = mid - 1

        torch.cuda.empty_cache()            # avoid cluter of previous iterations on gpu

    return best_batch_size


if __name__ == '__main__':

    no_workers = 32

    train_dataset = Serbia(split='train')
    batch_size = get_batch_size(dcl_model.Model, train_dataset)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True, pin_memory=True)

    test_dataset = Serbia(split='test')
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, drop_last=True, pin_memory=True)

    metrics = [
               CustomAccuracy().cuda(),
               Precision('micro').cuda(),
               Precision('macro').cuda(),
               Precision('per class').cuda(),
               Recall('micro').cuda(),
               Recall('macro').cuda(),
               Recall('per class').cuda(),
               F1_Score('micro').cuda(),
               F1_Score('macro').cuda(),
               F1_Score('per class').cuda()
               ]

    supervised_result_directory = Path('results/supervised')
    supervised_evaluation_directory = supervised_result_directory / 'evaluations'
    supervised_model_directory = supervised_result_directory / 'models'
    supervised_evaluation_name = 'supervised'

    evaluator = Evaluator(supervised_evaluation_directory, supervised_model_directory)
    dataloaders = {'test': test_dataloader}
    evaluator.evaluate(dataloaders, metrics, Model)
    evaluator.save(supervised_evaluation_name)

    # not showing improvement than bce
    # evaluator = Evaluator(evaluation_directory, results_directory / 'models_db_loss')
    # dataloaders = {'test': test_dataloader}
    # metrics = [PerClassAccuracy().cuda(), CustomAccuracy().cuda()]
    # evaluator.evaluate(dataloaders, metrics, Model, model_multiplier=5)
    # evaluator.save('db_loss')

    self_supervised_result_directory = Path('results/self_supervised')
    self_supervised_evaluation_directory = self_supervised_result_directory / 'evaluations'
    self_supervised_model_directory = self_supervised_result_directory / 'models'
    self_supervised_evaluation_name = 'self-supervised'

    model = dcl_model.Model().cuda()
    model = nn.DataParallel(model)

    saved_models = [(len(str(path)), str(path)) for path in self_supervised_model_directory.glob('*')]
    saved_models.sort(reverse=True)
    if len(saved_models) > 0:
        latest_saved_model = saved_models[0][1]
        model.load_state_dict(torch.load(latest_saved_model))

    train_x = []
    train_y = []
    with torch.no_grad():
        for x, l in tqdm(train_dataloader):
            x = x[:, 0]
            f, _ = model(x)
            train_x.append(f)
            train_y.append(l)

    train_x = torch.cat(train_x, dim=0)
    train_y = torch.cat(train_y, dim=0).cuda()

    evaluator = Evaluator(self_supervised_evaluation_directory, self_supervised_model_directory)
    dataloaders = {'test': test_dataloader}
    dcl_classifier = DCL_classifier(None, (train_x, train_y))

    def model_wrapper(m):
        dcl_classifier.dcl_model = m
        return dcl_classifier

    evaluator.evaluate(dataloaders, metrics, dcl_model.Model, model_wrapper, percentages=supervised.training_percentages)
    evaluator.save(self_supervised_evaluation_name)


    Evaluator.plot([supervised_evaluation_name, self_supervised_evaluation_name], [Evaluator.load(supervised_evaluation_directory / supervised_evaluation_name), Evaluator.load(self_supervised_evaluation_directory / self_supervised_evaluation_name)], critical_evaluator_name = self_supervised_evaluation_name)
