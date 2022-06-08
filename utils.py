import os
import shutil
import geopandas as gp
from pathlib import Path
import lmdb
import torch
from tqdm import tqdm
import pickle as pk
from multiprocessing import Pool
from torch.utils.data import DataLoader

from patch import Patch

LMDB_MAP_SIZE = 1024*1024*1024*128           # 128 gb

def func_1(args):
    row, from_, to = args
    patch_name = row['name']
    split_folder = (' ' if row['original_split'] == None else row['original_split']) + '/'
    shutil.copytree(str(from_ / patch_name), str(to / split_folder / patch_name))

def generate_subset_from_bigearth(big_earth_dataset_path = Path('/BigEarthNet-v1.0/'), big_eath_meta_parquet_path = Path('/extended_ben_gdf.parquet'), serbia_dataset_path = Path('../bigearth_subset/')):
    
    gdf = gp.read_parquet(str(big_eath_meta_parquet_path))
    gdf = gdf[gdf.snow != True]
    gdf = gdf[gdf.cloud_or_shadow != True]

    with Pool(processes=20) as pool:
        def gen_wrapper():
            for _, row in gdf.iterrows():
                yield row, big_earth_dataset_path, serbia_dataset_path

        for _ in pool.imap_unordered(func_1, tqdm(gen_wrapper(), total=len(gdf)), chunksize=1024):
            pass

    print(f'[train, validation, test] = [{len(gdf[gdf.original_split == "train"])}, {len(gdf[gdf.original_split == "validation"])}, {len(gdf[gdf.original_split == "test"])}]')
    print('dataset generated in folder: ' + str(serbia_dataset_path))

def func_2(args):
    patch_directory, split = args
    return pk.dumps(patch_directory.name + '_' + split), pk.dumps(Patch(patch_directory, split))

def generate_lmdb_from_dataset(dataset_directory = Path('../bigearth_subset/'), lmdb_directory = Path('../bigearth_subset_lmdb/')):

    env = lmdb.open(str(lmdb_directory), map_size=LMDB_MAP_SIZE)
    txn = env.begin(write=True)

    for dataset_split_directory in dataset_directory.iterdir():
        split = dataset_split_directory.name
        if (split != ' '):
            with Pool(processes=20) as pool:
                def gen_wrapper():
                    for directory in dataset_split_directory.iterdir():
                        yield directory, split

                for key, value in pool.imap_unordered(func_2, tqdm(gen_wrapper(), total=len(os.listdir(dataset_split_directory))), chunksize=1024):
                    txn.put(key, value)

    txn.commit()
    env.close()

def compute_mean_and_std_from_dataset(dataloader: DataLoader):
    sum = torch.zeros(Patch.bands).cuda()
    n = 0
    for bx, bl in tqdm(dataloader):
        bx = bx.cuda()
        sum += torch.sum(bx, dim=(0, 2, 3))
        n += bx.shape[0]

    mean = sum / (n * Serbia.size[0] * Serbia.size[1])

    sum = torch.zeros(Patch.bands).cuda()
    n = 0
    for bx, bl in tqdm(dataloader):
        bx = bx.cuda()
        sum += torch.sum(torch.pow(torch.sub(bx, mean[None, :, None, None]), 2.), dim=(0, 2, 3))
        n += bx.shape[0]

    std = torch.sqrt(sum / (n * Serbia.size[0] * Serbia.size[1]))

    return mean, std