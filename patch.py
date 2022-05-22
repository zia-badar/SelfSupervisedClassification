import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


class Patch:
    band_to_index = {'B01': 0, 'B02': 1, 'B03': 2, 'B04': 3, 'B05': 4, 'B06': 5, 'B07': 6, 'B08': 7, 'B8A': 8, 'B09': 9, 'B11': 10, 'B12': 11}
    label_to_index = {'Agro-forestry areas': 0, 'Airports': 1, 'Annual crops associated with permanent crops': 2, 'Bare rock': 3, 'Beaches, dunes, sands': 4, 'Broad-leaved forest': 5, 'Burnt areas': 6, 'Coastal lagoons': 7, 'Complex cultivation patterns': 8, 'Coniferous forest': 9, 'Construction sites': 10, 'Continuous urban fabric': 11, 'Discontinuous urban fabric': 12, 'Dump sites': 13, 'Estuaries': 14, 'Fruit trees and berry plantations': 15, 'Green urban areas': 16, 'Industrial or commercial units': 17, 'Inland marshes': 18, 'Intertidal flats': 19, 'Land principally occupied by agriculture, with significant areas of natural vegetation': 20, 'Mineral extraction sites': 21, 'Mixed forest': 22, 'Moors and heathland': 23, 'Natural grassland': 24, 'Non-irrigated arable land': 25, 'Olive groves': 26, 'Pastures': 27, 'Peatbogs': 28, 'Permanently irrigated land': 29, 'Port areas': 30, 'Rice fields': 31, 'Road and rail networks and associated land': 32, 'Salines': 33, 'Salt marshes': 34, 'Sclerophyllous vegetation': 35, 'Sea and ocean': 36, 'Sparsely vegetated areas': 37, 'Sport and leisure facilities': 38, 'Transitional woodland/shrub': 39, 'Vineyards': 40, 'Water bodies': 41, 'Water courses': 42}

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
        self.labels = []

        for l in meta_json['labels']:
            self.labels.append(Patch.label_to_index[l])

    def __repr__(self):
        plt.imshow(self.data[1:4].permute(1, 2, 0))
        plt.show()
        return ""
