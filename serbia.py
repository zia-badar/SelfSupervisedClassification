import pickle as pk
from pathlib import Path

import lmdb
import numpy as np
import torch
import torchvision
from torch.utils.data import Dataset
from torchvision.transforms.functional import hflip

from patch import Patch
from utils import LMDB_MAP_SIZE


class Serbia(Dataset):

    size = 128, 128
    resize = torchvision.transforms.Resize(size, interpolation=torchvision.transforms.InterpolationMode.BICUBIC)

    def __init__(self, lmdb_directory:Path, split:str='train'):
        self.split = split
        self.lmdb_directory = lmdb_directory

        env = lmdb.open(str(self.lmdb_directory), map_size=LMDB_MAP_SIZE)
        txn = env.begin()

        self.data_keys = []
        for _, v in txn.cursor():
            patch = pk.loads(v)
            if (patch.split == self.split):
                self.data_keys.append(patch.name)

        env.close()

    def __getitem__(self, item):
        env = lmdb.open(str(self.lmdb_directory), map_size=LMDB_MAP_SIZE)
        txn = env.begin()

        patch = pk.loads(txn.get(pk.dumps(self.data_keys[item])))

        processed = torch.zeros((len(Patch.band_to_index),) + Serbia.size, dtype=torch.float32)
        for i, bdata in enumerate(patch.data):
            bdata = torch.from_numpy(bdata.astype(np.int32))
            bdata = (bdata - torch.min(bdata)) / (torch.max(bdata) - torch.min(bdata))
            bdata = Serbia.resize(bdata.unsqueeze(0)).squeeze(0)
            processed[i] = bdata

        env.close()

        labels = torch.zeros(len(Patch.label_to_index))
        labels[patch.labels] = 1

        return processed, hflip(processed), labels

    def __len__(self):
        return len(self.data_keys)