import torch
from torch.utils.data import Dataset
import lmdb
from pathlib import Path
import pickle as pk
import numpy as np
from torch.nn.functional import interpolate
import torchvision.transforms

from utils import LMDB_MAP_SIZE

class Serbia(Dataset):

    def __init__(self, lmdb_directory:Path, split:str="train"):
        self.split = split

        env = lmdb.open(str(lmdb_directory), map_size=LMDB_MAP_SIZE)
        txn = env.begin()

        self.data = []
        for _, v in txn.cursor():
            patch = pk.loads(v)
            if (patch.split == self.split):
                self.data.append(pk.loads(v))

        env.close()

    def __getitem__(self, item):
        self.data[item].process_to_tensor()
        return self.data[item].data

    def __len__(self):
        return len(self.data)
