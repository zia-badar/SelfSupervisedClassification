import pickle as pk
from multiprocessing import Pool
from pathlib import Path

import lmdb
import torch
import torchvision
from matplotlib import pyplot as plt
from torch.utils.data import Dataset
from torchvision.transforms.functional import hflip, vflip
from tqdm import tqdm

from patch import Patch
from utils import LMDB_MAP_SIZE


def func_3(args):
    key, split = args
    str_key = pk.loads(memoryview(key))
    split_ = str_key.rsplit('_', 1)[1]
    return key if split_ == split else None

class Serbia(Dataset):

    size = 120, 120
    resize = torchvision.transforms.Resize(size, interpolation=torchvision.transforms.InterpolationMode.BICUBIC)
    augmentations = 3

    mean = torch.tensor([0.0067, 0.0095, 0.0092, 0.0147, 0.0274, 0.0317, 0.0339, 0.0346, 0.0242, 0.0153])
    std = torch.tensor([0.0094, 0.0095, 0.0109, 0.0116, 0.0170, 0.0196, 0.0210, 0.0208, 0.0165, 0.0125])
    normalize = torchvision.transforms.Normalize(mean, std)


    def __init__(self, lmdb_directory:Path = Path('../bigearth_subset_lmdb'), split:str='train'):
        self.split = split
        self.data_keys = []

        self.env = lmdb.open(str(lmdb_directory), map_size=LMDB_MAP_SIZE, readonly=True, lock=False)

        with self.env.begin(buffers=True) as txn:
            with Pool(processes=20) as pool:
                def gen_wrapper():
                    for k, _ in txn.cursor():
                        yield k.tobytes(), split

                for patch_name in pool.imap_unordered(func_3, tqdm(gen_wrapper(), total=txn.stat()['entries']), chunksize=1024):
                    if patch_name != None:
                        self.data_keys.append(patch_name)

    def __getitem__(self, item):
        with self.env.begin(buffers=True) as txn:
            patch = pk.loads(txn.get(memoryview(self.data_keys[item])))

        processed = torch.zeros((Patch.bands,) + Serbia.size, dtype=torch.float32)
        for i, bdata in enumerate(patch.data):
            bdata = bdata / (1 << 16)
            bdata = Serbia.resize(bdata.unsqueeze(0)).squeeze(0)
            processed[i] = bdata

        processed = Serbia.normalize(processed)

        x = torch.empty((Serbia.augmentations,) + processed.shape)
        indexes = torch.randperm(8)[:Serbia.augmentations]

        hflipped = None
        for i, ind in enumerate(indexes):
            img = None
            if ind / 4 == 1:
                img = hflipped = hflip(processed) if hflipped == None else hflipped
            else:
                img = processed

            x[i] = torchvision.transforms.functional.rotate(img, ind.item() * 90)

        labels = torch.zeros(Patch.classes)
        labels[patch.labels] = 1

        return x, labels

    def __len__(self):
        return len(self.data_keys)