from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from DCL.model import Model
from serbia import Serbia
from utils import generate_serbia_from_bigearth


def get_negative_mask(batch_size):
    negative_mask = torch.ones((batch_size, 2 * batch_size), dtype=bool)
    for i in range(batch_size):
        negative_mask[i, i] = 0
        negative_mask[i, i + batch_size] = 0

    negative_mask = torch.cat((negative_mask, negative_mask), 0)
    return negative_mask


def train(net, data_loader, train_optimizer, temperature, debiased, tau_plus):
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

    return total_loss / total_num

if __name__ == '__main__':
	generate_serbia_from_bigearth(Path('/beegfs/scratch/rsim_data/BigEarthNet-v1.0/'), Path('/home/users/z/zia_badar/extended_ben_gdf.parquet'))
    # generate_lmdb_from_dataset(Path('serbia_dataset'), Path('serbia_dataset_lmdb'))

	batch_size = 8
	epochs = 100
	
	train_dl = DataLoader(Serbia(Path('serbia_dataset_lmdb'), split='train'), batch_size=batch_size, num_workers=16, shuffle=True, drop_last=True)
	
	model = Model(128).cuda()
	model = nn.DataParallel(model)
	optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-6)
	
	for epoch in range(1, epochs + 1):
			train(model, train_dl, optimizer, .5, True, .1)
			torch.save(model.state_dict(), f'results/model_{epoch}.pth')
