from pathlib import Path
from PIL import Image
from torchvision import transforms
import json
import re
import torch
import numpy as np
import matplotlib.pyplot as plt
import torchvision


class Patch:

    band_to_index = {'B01':0, 'B02':1, 'B03':2, 'B04':3, 'B05':4, 'B06':5, 'B07':6, 'B08':7, 'B8A':8, 'B09':9, 'B11':10, 'B12':11}

    def __init__(self, patch_dir: Path, split):
        self.name = patch_dir.name
        self.split = split
        self.data = [None] * len(Patch.band_to_index)

        for band_path in patch_dir.iterdir():
            if (band_path.suffix == '.tif'):
                image = Image.open(band_path)
                band = re.sub('.*_', '', band_path.stem)
                self.data[Patch.band_to_index[band]] = np.array(image, dtype=np.uint16)

        meta_json = json.load(open(str(patch_dir.glob('*.json').__next__())))
        self.labels = meta_json['labels']

    def __repr__(self):
        plt.imshow(self.data[1:4].permute(1, 2, 0))
        plt.show()
        return ""