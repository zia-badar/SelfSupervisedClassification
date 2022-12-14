import pickle
import pickle as pk
from multiprocessing import Pool
from os.path import exists
from pathlib import Path

import lmdb
import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms.functional import hflip
from tqdm import tqdm

from patch import Patch
from utils import LMDB_MAP_SIZE

size = 128, 128


def func_3(args):
    key, split = args
    str_key = pk.loads(memoryview(key))
    split_ = str_key.rsplit('_', 1)[1]
    return key if split_ == split else None


class BigEarthDataset(Dataset):
    resize = torchvision.transforms.Resize(size, interpolation=torchvision.transforms.InterpolationMode.BICUBIC)
    default_augmentation_count = 4

    mean = torch.tensor([0.0067, 0.0095, 0.0092, 0.0147, 0.0274, 0.0317, 0.0339, 0.0346, 0.0242, 0.0153])
    std = torch.tensor([0.0094, 0.0095, 0.0109, 0.0116, 0.0170, 0.0196, 0.0210, 0.0208, 0.0165, 0.0125])
    normalize = torchvision.transforms.Normalize(mean, std)

    def __init__(self, lmdb_directory: Path = Path('../bigearth_subset_lmdb'), split: str = 'train', augmentation=True,
                 augementation_type=1, augmentation_count=default_augmentation_count):
        self.split = split
        self.augmentation = augmentation
        self.augmentation_type = augementation_type
        self.augmentation_count = augmentation_count
        self.data_keys = []

        self.env = lmdb.open(str(lmdb_directory), map_size=LMDB_MAP_SIZE, readonly=True, lock=False)
        keys_filename = lmdb_directory / (split + '_keys')

        if exists(keys_filename):  # saves time and take very little extra storage, specially for hpc
            with open(keys_filename, 'rb') as f:
                self.data_keys = pickle.load(f)
        else:
            with self.env.begin(buffers=True) as txn:
                with Pool(processes=20) as pool:
                    def gen_wrapper():
                        for k, _ in txn.cursor():
                            yield k.tobytes(), split

                    for patch_name in pool.imap_unordered(func_3, tqdm(gen_wrapper(), total=txn.stat()['entries']),
                                                          chunksize=1024):
                        if patch_name != None:
                            self.data_keys.append(patch_name)

            with open(keys_filename, 'wb') as f:
                pickle.dump(self.data_keys, f, pickle.HIGHEST_PROTOCOL)

    def __getitem__(self, item):
        with self.env.begin(buffers=True) as txn:
            patch = pk.loads(txn.get(memoryview(self.data_keys[item])))

        processed = torch.zeros((Patch.bands,) + size, dtype=torch.float32)
        for i, bdata in enumerate(patch.data):
            bdata = bdata / (1 << 16)
            bdata = BigEarthDataset.resize(bdata.unsqueeze(0)).squeeze(0)
            processed[i] = bdata

        processed = BigEarthDataset.normalize(processed)

        x = torch.empty((self.augmentation_count,) + processed.shape)
        if self.augmentation:
            if self.augmentation_type == 1:
                indexes = torch.randperm(8)[:self.augmentation_count]

                hflipped = None
                for i, ind in enumerate(indexes):
                    img = None
                    if ind / 4 == 1:
                        img = hflipped = hflip(processed) if hflipped == None else hflipped
                    else:
                        img = processed

                    x[i] = torchvision.transforms.functional.rotate(img, ind.item() * 90)
            elif self.augmentation_type == 2:
                percentage_to_mask = 0.2
                no_splits = torch.tensor([8, 8])
                image_size = torch.tensor([size[0], size[1]])
                split_size = (image_size / no_splits).int()
                for i in range(self.augmentation_count):
                    a = processed.unfold(1, split_size[0], split_size[0]).unfold(2, split_size[1],
                                                                                 split_size[1]).reshape(
                        [-1, no_splits.prod(), split_size[0], split_size[1]])[:, torch.randperm(no_splits.prod())]
                    a[:, torch.randperm(no_splits.prod())[:(int)(percentage_to_mask * no_splits.prod())]] = 0
                    a = a.unfold(1, no_splits[0], no_splits[0]).permute([0, 4, 2, 1, 3]).reshape(-1, image_size[0],
                                                                                                 image_size[1])
                    x[i] = a

        labels = torch.zeros(Patch.classes)
        labels[patch.labels] = 1

        return x if self.augmentation else processed.unsqueeze(0), labels

    def __len__(self):
        return len(self.data_keys)


def compute_mean_and_std_from_dataset(dataloader: DataLoader):
    sum = torch.zeros(Patch.bands).cuda()
    n = 0
    for bx, bl in tqdm(dataloader):
        bx = bx.cuda()
        sum += torch.sum(bx, dim=(0, 2, 3))
        n += bx.shape[0]

    mean = sum / (n * size[0] * size[1])

    sum = torch.zeros(Patch.bands).cuda()
    n = 0
    for bx, bl in tqdm(dataloader):
        bx = bx.cuda()
        sum += torch.sum(torch.pow(torch.sub(bx, mean[None, :, None, None]), 2.), dim=(0, 2, 3))
        n += bx.shape[0]

    std = torch.sqrt(sum / (n * size[0] * size[1]))

    return mean, std
