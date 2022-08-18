# Multi label classification on remote images using self-supervised contrastive representation

## Project:
This project is to check if self-supervised contrastive learned classification can perform better than supervised learning when less labelled data is used. Contrastive learning can be done by labelled data or unlabelled data but contrastive leanring on unlabelled data makes the model baised because of which I have used [unlabelled debaised contrastive learning](https://arxiv.org/pdf/2007.00224.pdf). 

## Debaised Contrastive loss:
<img src="https://i.imgur.com/lUzvnw2.png" width="600" />

Above debaised contrastive loss is used to learn the representation of the data, with the no. of augmentation M = 4.

## Augmentation technique used:
<img src="https://i.imgur.com/JSoTlxo.png" width="800" />

Augmentation technique used is taken as an idea from this [paper](https://arxiv.org/pdf/1603.09246.pdf) , they are also using it for learning data representation but in a different way, although image shows 4x4 splits model is trained with 8x8 for better performance.

# Results
<img src="https://i.imgur.com/B5GA9Q7.png" width="600" />
<img src="https://i.imgur.com/fDImWV9.png" width="600" />

Although the self-supervise gets very close to supervised learning, still it is behind supervised learning, which I think might get improved by using a bigger batch size for contrastive learning because I was limited by 2 A100 gpus or using a more complex augmentation technique.

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
- Download BigEarthNet-S2 from [link](https://bigearth.net/) and extract
- Download extended_ben_gdf.parquet from [link](https://docs.kai-tub.tech/ben-docs/20_raw-data.html#pre-converted-metadata)
- run the function generate_subset_from_bigearth from utils and give it the location of dataset and parquet file, It will generate a filtered dataset in one directory above the project folder.
- run the function generate_lmdb_from_dataset from utils, it will generate the lmdb file

#### Evaluation
- run evaluation.py to evaluate the train models.[make sure self-supervised models folder contain only one model because evaluation of self-supervised model is done on percentage of data not on different models unlike supervised, to also evaluate random model you would need to evaluate one model at a time]

#### Download pretrained
- download pretrained models and evaluations from [link](https://drive.google.com/file/d/1Pt-_N_CwU_fb_BREFQa21Xr5JCCB4Zx3/view?usp=sharing)

### Train from scratch and evaluate
- run data preperation task
- run supervised.py to train supervised models.[change training_percentage in this file to train for different values]
- run self_supervised.py to train self_supervised.py.
- run evaluation task

### Use pretrained models and evaluate
- run data preperation task first
- run download pretrained task
- run evaluation task

### Use pre-evaluated evaluations and only plot
- run download pretrained task
- run evaluation task and comment evaluation code