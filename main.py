from pathlib import Path

from serbia import Serbia
from utils import generate_lmdb_from_dataset
import torch

if __name__ == '__main__':
    generate_lmdb_from_dataset(Path('serbia_dataset'), Path('serbia_dataset_lmdb'))
    #
    dataset = Serbia(Path("serbia_dataset_lmdb"))
    print(dataset.__getitem__(0).shape)