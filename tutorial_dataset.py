import cv2
import json
import random
import numpy as np
import os

from torch.utils.data import Dataset
from PIL import Image
import albumentations


class MyDataset(Dataset):
    def __init__(self):
        self.data = []
        json_path = '/media/Storage3/rhw/dataset_prompts_5class.json' 
        with open(json_path, 'r', encoding='utf-8') as f:
            self.prompts_dict = json.load(f)
            self.filenames = list(self.prompts_dict.keys())

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        file_name = self.filenames[idx]
        prompt = self.prompts_dict[file_name]

        source_filename = os.path.join('/media/Storage3/rhw/data/data_origan/consep/aa/train_masks_dtm', file_name)
        target_filename = os.path.join('/media/Storage3/rhw/consep_train/images', file_name)

        source = cv2.imread(source_filename)
        source = cv2.cvtColor(source, cv2.COLOR_BGR2RGB)
        


        target = cv2.imread(target_filename)
        target = cv2.cvtColor(target, cv2.COLOR_BGR2RGB)


        source = source.astype(np.float32) / 255.0
        target = target.astype(np.float32) / 127.5 - 1.0

        return dict(jpg=target, txt=prompt, hint=source)

    def transform(self, size=256):
        transforms = albumentations.Compose(
                        [
                            albumentations.Resize(height=size, width=size)
                        ]
                    )
        return transforms
