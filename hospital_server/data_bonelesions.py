import os
import numpy as np
import torch
from torch.utils.data import Dataset


class KahlerDataset(Dataset):

    def __init__(self, img_dir, mask_dir):

        self.image_dir = img_dir
        self.mask_dir = mask_dir
        self.image_files = sorted(os.listdir(self.image_dir))
        self.mask_files = sorted(os.listdir(self.mask_dir))

    def check_files(self):

        print(self.image_files, self.mask_files)

    def __len__(self):

        return len(self.image_files)

    def __getitem__(self, idx):
        
        img_path = os.path.join(self.image_dir, self.image_files[idx])
        mask_path = os.path.join(self.mask_dir, self.mask_files[idx])

        #file=os.path.basename(img_path)

        image = np.load(img_path)
        mask = np.load(mask_path)

        image = np.expand_dims(image, axis=0)
        mask = np.expand_dims(mask, axis=0)

        image = torch.from_numpy(image)
        mask = torch.from_numpy(mask)

        return image, mask#, file