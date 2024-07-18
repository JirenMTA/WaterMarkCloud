import cv2
import numpy as np
from torch.utils.data import Dataset
import pandas as pd
import utils
import os

current_dir = os.path.dirname(os.path.abspath(__file__))

class Dataset38Cloud(Dataset):
    def __init__(self, set_type, max_num):
        assert set_type in {'train', 'test'}

        if set_type == 'train':
            self.DATASET_PATH = os.path.join(current_dir, "./dataset/38-Cloud_training")
            patch_names = pd.read_csv(
                f"{self.DATASET_PATH}/training_patches_38-Cloud.csv")
        else:
            self.DATASET_PATH = os.path.join(current_dir,"./dataset/38-Cloud_test")
            patch_names = pd.read_csv(
                f"{self.DATASET_PATH}/test_patches_38-Cloud.csv")
        self.set_type = set_type
        self.max_num = max_num
        self.patch_names = patch_names.name[:max_num]
    def __getitem__(self, item):
        red = utils.read_img(f'{self.DATASET_PATH}/{self.set_type}_red/red_{self.patch_names[item]}.TIF')
        blue = utils.read_img(f'{self.DATASET_PATH}/{self.set_type}_blue/blue_{self.patch_names[item]}.TIF')
        green = utils.read_img(f'{self.DATASET_PATH}/{self.set_type}_green/green_{self.patch_names[item]}.TIF')
        nir = utils.read_img(f'{self.DATASET_PATH}/{self.set_type}_nir/nir_{self.patch_names[item]}.TIF')

        mask = utils.read_img(f'{self.DATASET_PATH}/{self.set_type}_gt/gt_{self.patch_names[item]}.TIF')

        return np.transpose(cv2.merge((red, green, blue, nir)), (2, 0, 1))/(2.**16-1), np.expand_dims(mask, axis=0)

    def __len__(self):
        return self.max_num