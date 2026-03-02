import os
import gc
import random
import torch
import tempfile
import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset


class TrainDataset_ARFI(Dataset):
    def __init__(self, img_path_ir, img_path_vis, transform=None, seed=42):
        self.img_path_ir = img_path_ir
        self.img_path_vis = img_path_vis
        self.transform = transform
        self.crop_size = (128, 128)
        self.seed = seed
        self.img_names = [img_name for img_name in os.listdir(img_path_ir) if img_name.endswith(('.jpg', 'png', 'jpeg'))]

    def __len__(self):
        return len(self.img_names)

    def names(self):
        return self.img_names

    def set_seed(self):
        random.seed(self.seed)
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed(self.seed)
        torch.cuda.manual_seed_all(self.seed)
        import numpy as np
        np.random.seed(self.seed)


    def __getitem__(self, index):
        self.set_seed()

        img_name = self.img_names[index]
        img_path_ir = os.path.join(self.img_path_ir, img_name)
        img_path_vis = os.path.join(self.img_path_vis, img_name)
        image_ir = Image.open(img_path_ir)
        image_vis = Image.open(img_path_vis)

        width, height = image_ir.size
        crop_x = random.randint(0, width - self.crop_size[0])
        crop_y = random.randint(0, height - self.crop_size[1])
        crop_box = (crop_x, crop_y, crop_x + self.crop_size[0], crop_y + self.crop_size[1])

        image_ir = image_ir.crop(crop_box)
        image_vis = image_vis.crop(crop_box)

        image_ycbcr = image_vis.convert('YCbCr')
        y, cb, cr = image_ycbcr.split()

        if self.transform is not None:
            image_ir = self.transform(image_ir)
            y_tensor = self.transform(y)
            cb = self.transform(cb)
            cr = self.transform(cr)
        image_ir_1 = image_ir[:1,:,:]

        return image_ir_1, y_tensor, cb, cr


class TestDataset(Dataset):
    def __init__(self, img_path_vis, transform=None, max_token_len=128):
        self.img_path_vis = img_path_vis
        self.transform = transform
        self.img_names = [img_name for img_name in os.listdir(img_path_vis) if img_name.endswith(('.jpg', 'png', 'jpeg'))]

    def __len__(self):
        return len(self.img_names)

    def names(self):
        return self.img_names

    def __getitem__(self, index):
        img_name = self.img_names[index]
        img_path_vis = os.path.join(self.img_path_vis, img_name)
        image_vis = Image.open(img_path_vis)
        image_vis = image_vis.convert('YCbCr')
        y, cb, cr = image_vis.split()

        if self.transform is not None:
            y_tensor = self.transform(y)
            cb = self.transform(cb)
            cr = self.transform(cr)

        return y_tensor, cb, cr