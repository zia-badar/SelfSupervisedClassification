import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
from PIL import Image
from torchvision.transforms import ToTensor


class Patch:
    band_to_index = {'B02' : 0, 'B03' : 1, 'B04' : 2, 'B05' : 3, 'B06' : 4, 'B07' : 5, 'B08' : 6, 'B8A' : 7, 'B11' : 8, 'B12' : 9}
    band_to_exclude = {'B01', 'B09', 'B10'}
    _43_to_19 = {'Continuous urban fabric': 'Urban fabric', 'Discontinuous urban fabric': 'Urban fabric', 'Industrial or commercial units': 'Industrial or commercial units', 'Road and rail networks and associated land': 'REMOVED', 'Port areas': 'REMOVED', 'Airports': 'REMOVED', 'Mineral extraction sites': 'REMOVED', 'Dump sites': 'REMOVED', 'Construction sites': 'REMOVED', 'Green urban areas': 'REMOVED', 'Sport and leisure facilities': 'REMOVED', 'Non-irrigated arable land': 'Arable land', 'Permanently irrigated land': 'Arable land', 'Rice fields': 'Arable land', 'Vineyards': 'Permanent crops', 'Fruit trees and berry plantations': 'Permanent crops', 'Olive groves': 'Permanent crops', 'Pastures': 'Pastures', 'Annual crops associated with permanent crops': 'Permanent crops', 'Complex cultivation patterns': 'Complex cultivation patterns', 'Land principally occupied by agriculture, with significant areas of natural vegetation': 'Land principally occupied by agriculture, with significant areas of natural vegetation', 'Agro-forestry areas': 'Agro-forestry areas', 'Broad-leaved forest': 'Broad-leaved forest', 'Coniferous forest': 'Coniferous forest', 'Mixed forest': 'Mixed forest', 'Natural grassland': 'Natural grassland and sparsely vegetated areas', 'Moors and heathland': 'Moors, heathland and sclerophyllous vegetation', 'Sclerophyllous vegetation': 'Moors, heathland and sclerophyllous vegetation', 'Transitional woodland/shrub': 'Transitional woodland, shrub', 'Beaches, dunes, sands': 'Beaches, dunes, sands', 'Bare rock': 'REMOVED', 'Sparsely vegetated areas': 'Natural grassland and sparsely vegetated areas', 'Burnt areas': 'REMOVED', 'Inland marshes': 'Inland wetlands', 'Peatbogs': 'Inland wetlands', 'Salt marshes': 'Coastal wetlands', 'Salines': 'Coastal wetlands', 'Intertidal flats': 'REMOVED', 'Water courses': 'Inland waters', 'Water bodies': 'Inland waters', 'Coastal lagoons': 'Marine waters', 'Estuaries': 'Marine waters', 'Sea and ocean': 'Marine waters'}
    _19_label_to_index = {'Agro-forestry areas': 0, 'Arable land': 1, 'Beaches, dunes, sands': 2, 'Broad-leaved forest': 3, 'Coastal wetlands': 4, 'Complex cultivation patterns': 5, 'Coniferous forest': 6, 'Industrial or commercial units': 7, 'Inland waters': 8, 'Inland wetlands': 9, 'Land principally occupied by agriculture, with significant areas of natural vegetation': 10, 'Marine waters': 11, 'Mixed forest': 12, 'Moors, heathland and sclerophyllous vegetation': 13, 'Natural grassland and sparsely vegetated areas': 14, 'Pastures': 15, 'Permanent crops': 16, 'Transitional woodland, shrub': 17, 'Urban fabric': 18}

    classes = len(_19_label_to_index)
    bands = len(band_to_index)

    def __init__(self, patch_dir: Path, split):
        self.name = patch_dir.name
        self.split = split
        self.data = [None] * Patch.bands

        for band_path in patch_dir.iterdir():
            if (band_path.suffix == '.tif'):
                band = re.sub('.*_', '', band_path.stem)
                if (band not in Patch.band_to_exclude):
                    self.data[Patch.band_to_index[band]] = ToTensor()(Image.open(band_path))

        meta_json = json.load(open(str(patch_dir.glob('*.json').__next__())))
        self.labels = []

        for l in meta_json['labels']:
            if Patch._43_to_19[l] != 'REMOVED':
                self.labels.append(Patch._19_label_to_index[Patch._43_to_19[l]])

    def __repr__(self):
        plt.imshow(self.data[1:4].permute(1, 2, 0))
        plt.show()
        return ""
