from pathlib import Path

from tqdm import tqdm

from serbia import Serbia
from utils import generate_lmdb_from_dataset
from torch.utils.data import DataLoader

if __name__ == '__main__':
    # generate_lmdb_from_dataset(Path('serbia_dataset'), Path('serbia_dataset_lmdb'))

    train_dl = DataLoader(Serbia(Path('serbia_dataset_lmdb'), 'train'), batch_size=256, num_workers=16)
    test_dl = DataLoader(Serbia(Path('serbia_dataset_lmdb'), 'test'), batch_size=32, num_workers=8)

    for d in tqdm(train_dl):
        a = d
