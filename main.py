from utils import generate_serbia_from_bigearth, generate_lmdb_from_dataset, load_patches_from_lmdb
from pathlib import Path

if __name__ == '__main__':
    generate_lmdb_from_dataset(Path('serbia_dataset'), Path('serbia_dataset_lmdb'))
    patches = load_patches_from_lmdb(Path('serbia_dataset_lmdb'))