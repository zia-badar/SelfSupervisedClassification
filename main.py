from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

import DCL.model
from serbia import Serbia

def get_negative_mask(batch_size):
    negative_mask = torch.ones((batch_size, 2 * batch_size), dtype=bool)
    for i in range(batch_size):
        negative_mask[i, i] = 0
        negative_mask[i, i + batch_size] = 0

    negative_mask = torch.cat((negative_mask, negative_mask), 0)
    return negative_mask


def train(net, data_loader, test_dataloader, train_optimizer, temperature, debiased, tau_plus):
    net.train()
    total_loss, total_num, train_bar = 0.0, 0, tqdm(data_loader)
    for pos_1, pos_2, target in train_bar:
        pos_1, pos_2 = pos_1.cuda(non_blocking=True), pos_2.cuda(non_blocking=True)
        feature_1, out_1 = net(pos_1)
        feature_2, out_2 = net(pos_2)

        # neg score
        out = torch.cat([out_1, out_2], dim=0)
        neg = torch.exp(torch.mm(out, out.t().contiguous()) / temperature)
        mask = get_negative_mask(batch_size).cuda()
        neg = neg.masked_select(mask).view(2 * batch_size, -1)

        # pos score
        pos = torch.exp(torch.sum(out_1 * out_2, dim=-1) / temperature)
        pos = torch.cat([pos, pos], dim=0)

        # estimator g()
        if debiased:
            N = batch_size * 2 - 2
            Ng = (-tau_plus * N * pos + neg.sum(dim = -1)) / (1 - tau_plus)
            # constrain (optional)
            Ng = torch.clamp(Ng, min = N * np.e**(-1 / temperature))
        else:
            Ng = neg.sum(dim=-1)

        # contrastive loss
        loss = (- torch.log(pos / (pos + Ng) )).mean()

        train_optimizer.zero_grad()
        loss.backward()
        train_optimizer.step()

        total_num += batch_size
        total_loss += loss.item() * batch_size

        train_bar.set_description('Train Epoch: [{}/{}] Loss: {:.4f}'.format(epoch, epochs, total_loss / total_num))

    if epochs % 5 == 0:
    	net.eval()
    	total_loss, total_num, train_bar = 0.0, 0, tqdm(test_dataloader)
    	for pos_1, pos_2, target in train_bar:
    	    pos_1, pos_2 = pos_1.cuda(non_blocking=True), pos_2.cuda(non_blocking=True)
    	    feature_1, out_1 = net(pos_1)
    	    feature_2, out_2 = net(pos_2)

    	    # neg score
    	    out = torch.cat([out_1, out_2], dim=0)
    	    neg = torch.exp(torch.mm(out, out.t().contiguous()) / temperature)
    	    mask = get_negative_mask(batch_size).cuda()
    	    neg = neg.masked_select(mask).view(2 * batch_size, -1)

    	    # pos score
    	    pos = torch.exp(torch.sum(out_1 * out_2, dim=-1) / temperature)
    	    pos = torch.cat([pos, pos], dim=0)

    	    # estimator g()
    	    if debiased:
    	        N = batch_size * 2 - 2
    	        Ng = (-tau_plus * N * pos + neg.sum(dim = -1)) / (1 - tau_plus)
    	        # constrain (optional)
    	        Ng = torch.clamp(Ng, min = N * np.e**(-1 / temperature))
    	    else:
    	        Ng = neg.sum(dim=-1)

    	    # contrastive loss
    	    loss = (- torch.log(pos / (pos + Ng) )).mean()

    	    train_optimizer.zero_grad()
    	    loss.backward()
    	    train_optimizer.step()

    	    total_num += batch_size
    	    total_loss += loss.item() * batch_size

    	    train_bar.set_description('Train Epoch: [{}/{}] Loss: {:.4f}'.format(epoch, epochs, total_loss / total_num))

    	print(f'test_loss: {total_loss / total_num}')

    return total_loss / total_num

if __name__ == '__main__':

    batch_size = 32
    no_workers = 16
    epochs = 200
    results_directory = Path('results/unsupervised')
    models_directory = results_directory / 'models'
    continue_training = True

    train_dataset = Serbia(split='train')
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, num_workers=no_workers, shuffle=True, drop_last=True, pin_memory=True)

    test_dataset = Serbia(split='test')
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, num_workers=no_workers, shuffle=True, drop_last=True, pin_memory=True)

    model = DCL.model.Model(128).cuda()
    model = nn.DataParallel(model)

    starting_epoch = 1
    if continue_training:
        saved_models = [(len(str(path)), str(path)) for path in models_directory.glob('*')]
        saved_models.sort(reverse=True)
        if len(saved_models) > 0:
            latest_saved_model = saved_models[0][1]
            starting_epoch = (int)(latest_saved_model.rsplit('_', 1)[1]) + 1
            model.load_state_dict(torch.load(latest_saved_model))

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-6)

    for epoch in range(starting_epoch, epochs + 1):
        train(model, train_dataloader, test_dataloader, optimizer, .5, True, .1)
        if epoch % 5 == 0:
            torch.save(model.state_dict(), str(models_directory / f'model_{epoch}'))
