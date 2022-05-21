import collections
from pathlib import Path
from PIL import Image
import numpy as np

class Patch:

    def __init__(self, patch_dir:Path, split):
        self.map = {}
        self.split = split
        for bands in patch_dir.iterdir():
            if (bands.suffix == '.tif'):
                self.map[bands.name] = np.array(Image.open(bands))