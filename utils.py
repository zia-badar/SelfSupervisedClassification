import os
import shutil
import geopandas as gp
from pathlib import Path
import lmdb
from tqdm import tqdm
import pickle as pk

from patch import Patch

def generate_serbia_from_bigearth():
    big_eath_meta_parquet_path = '/home/zia/Downloads/extended_ben_gdf.parquet'
    big_earth_dataset_path = '/home/zia/Downloads/BigEarthNet-v1.0/'
    serbia_dataset_path = 'serbia_dataset/'

    gdf = gp.read_parquet(big_eath_meta_parquet_path)
    gdf = gdf[gdf.snow != True]
    gdf = gdf[gdf.cloud_or_shadow != True]
    gdf = gdf[gdf.country == 'Serbia']

    for _, row in gdf.iterrows():
        patch_name = row['name']
        split_folder = (' ' if row['original_split'] == None else row['original_split']) + '/'
        shutil.copytree(big_earth_dataset_path + patch_name, serbia_dataset_path + split_folder + patch_name)

    train = gdf[gdf.original_split == 'train'].size
    validation = gdf[gdf.original_split == 'validation'].size
    test = gdf[gdf.original_split == 'test'].size
    print(f'[train, validation, test] = [{train}, {validation}, {test}]')
    print('dataset generated in folder: ' + serbia_dataset_path)

_32_GBs = 1024*1024*1024*32

def generate_lmdb_from_dataset(dataset_directory:Path, lmdb_directory:Path):

    env = lmdb.open(str(lmdb_directory), map_size=_32_GBs)    # 32 gb
    txn = env.begin(write=True)

    for dataset_split_directory in dataset_directory.iterdir():
        split = dataset_split_directory.name
        if (split != ' '):
            total_patches = len(os.listdir(dataset_split_directory))
            for patch_dir in tqdm(dataset_split_directory.iterdir(), total = total_patches):
                txn.put(pk.dumps(patch_dir.name), pk.dumps(Patch(patch_dir, split)))

    txn.commit()
    env.close()

def load_patches_from_lmdb(lmdb_directory:Path):
    env = lmdb.open(str(lmdb_directory), map_size=_32_GBs)
    txn = env.begin()

    patches = []
    for _, v in txn.cursor():
        patches.append(pk.loads(v))

    env.close()

    return patches