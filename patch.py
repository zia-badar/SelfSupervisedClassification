from pathlib import Path
from PIL import Image
import numpy as np
import json

class Patch:

    def __init__(self, patch_dir:Path, split):
        self.name = patch_dir.name
        self.split = split
        self.map = {}

        for bands in patch_dir.iterdir():
            if (bands.suffix == '.tif'):
                self.map[bands.stem] = np.array(Image.open(bands))

        meta_json = json.load(open(str(patch_dir.glob("*.json").__next__())))
        self.labels = meta_json['labels']