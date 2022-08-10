import os
import pickle as pk
import shutil
from multiprocessing import Pool
from pathlib import Path

import geopandas as gp
import lmdb
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from patch import Patch

LMDB_MAP_SIZE = 1024 * 1024 * 1024 * 128  # 128 gb


def func_1(args):
    row, from_, to = args
    patch_name = row['name']
    split_folder = (' ' if row['original_split'] == None else row['original_split']) + '/'
    shutil.copytree(str(from_ / patch_name), str(to / split_folder / patch_name))


def generate_subset_from_bigearth(big_earth_dataset_path=Path('/BigEarthNet-v1.0/'),
                                  big_eath_meta_parquet_path=Path('/extended_ben_gdf.parquet'),
                                  serbia_dataset_path=Path('../bigearth_subset/')):
    gdf = gp.read_parquet(str(big_eath_meta_parquet_path))
    gdf = gdf[gdf.snow != True]
    gdf = gdf[gdf.cloud_or_shadow != True]

    with Pool(processes=20) as pool:
        def gen_wrapper():
            for _, row in gdf.iterrows():
                yield row, big_earth_dataset_path, serbia_dataset_path

        for _ in pool.imap_unordered(func_1, tqdm(gen_wrapper(), total=len(gdf)), chunksize=1024):
            pass

    print(
        f'[train, validation, test] = [{len(gdf[gdf.original_split == "train"])}, {len(gdf[gdf.original_split == "validation"])}, {len(gdf[gdf.original_split == "test"])}]')
    print('dataset generated in folder: ' + str(serbia_dataset_path))


def func_2(args):
    patch_directory, split = args
    return pk.dumps(patch_directory.name + '_' + split), pk.dumps(Patch(patch_directory, split))


def generate_lmdb_from_dataset(dataset_directory=Path('../bigearth_subset/'),
                               lmdb_directory=Path('../bigearth_subset_lmdb/')):
    env = lmdb.open(str(lmdb_directory), map_size=LMDB_MAP_SIZE)
    txn = env.begin(write=True)

    for dataset_split_directory in dataset_directory.iterdir():
        split = dataset_split_directory.name
        if (split != ' '):
            with Pool(processes=20) as pool:
                def gen_wrapper():
                    for directory in dataset_split_directory.iterdir():
                        yield directory, split

                for key, value in pool.imap_unordered(func_2, tqdm(gen_wrapper(),
                                                                   total=len(os.listdir(dataset_split_directory))),
                                                      chunksize=1024):
                    txn.put(key, value)

    txn.commit()
    env.close()


# required for distributed balance loss, which is not use
def calculate_class_weights(dataloader: DataLoader):
    positive = torch.zeros(Patch.classes)
    negative = torch.zeros(Patch.classes)
    count = torch.zeros(Patch.classes)

    for _, l in dataloader:
        positive += torch.sum(l == 1, dim=0)
        negative += torch.sum(l == 0, dim=0)
        count += torch.sum(l, dim=0)

    n = positive[0] + negative[0]
    positive = n / (2 * positive)
    negative = n / (2 * negative)

    positive = torch.nan_to_num(positive, posinf=0)
    negative = torch.nan_to_num(negative, posinf=0)

    return positive, negative, count
