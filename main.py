from pathlib import Path

from tqdm import tqdm

from serbia import Serbia
from utils import generate_lmdb_from_dataset
import torch

if __name__ == '__main__':
    # generate_lmdb_from_dataset(Path('serbia_dataset'), Path('serbia_dataset_lmdb'))

    dataset = Serbia(Path('serbia_dataset_lmdb'), 'test')
    for i in tqdm(range(dataset.__len__())):
        a = dataset.__getitem__(i)