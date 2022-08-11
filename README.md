# Multi label classification on remote images using self-supervised contrastive representation

![](https://i.imgur.com/SroH9ro.png)

## Requirements:
The code has been tested with the following requirements.
- python 3.10.4
- torch 1.11.0+cu113
- python-lmdb 1.3.0
- pillow 9.1.1  

## Usage:
There are 3 important file in project, supervised.py, self-supervised.py, evaluation.py. There are 3 ways to use the project

- Train from scratch and evaluate
- Use pretrained model and evaluate
- Use pre-evaluated evaluations to plot

### Tasks
#### Data preperation
- Download BigEarthNet-S2 from https://bigearth.net/ and extract
- Download extended_ben_gdf.parquet from https://docs.kai-tub.tech/ben-docs/20_raw-data.html#pre-converted-metadata
- run the function generate_subset_from_bigearth from utils and give it the location of dataset and parquet file, It will generate a filtered dataset in one directory above the project folder.
- run the function generate_lmdb_from_dataset from utils, it will generate the lmdb file

#### Evaluation
- run evaluation.py to evaluate the train models.[make sure self-supervised models folder contain only one model because evaluation of self-supervised model is done on percentage of data not on different models unlike supervised, to also evaluate random model you would need to evaluate one model at a time]

#### Download pretrained
- download pretrained models and evaluations from https://drive.google.com/file/d/1DXRlLeJrkjWD6DdhTdKAgEGwWL7UvbuM/view?usp=sharing [no need to delete evaluations they will get overwritten]

### Train from scratch and evaluate
- run data preperation task
- run supervised.py to train supervised models.[change training_percentage in this file to train for different values]
- run self_supervised.py to train self_supervised.py. [change epochs in this file to train for different epochs, although each epoch get saved so you also terminate the program at some desired epoch]
- run evaluation task

### Use pretrained models and evaluate
- run data preperation task first
- run download pretrained task
- run evaluation task

### Use pre-evaluated evaluations and only plot
- run download pretrained task
- run evaluation task and comment evaluation code