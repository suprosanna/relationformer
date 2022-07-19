import scipy
import os
import sys
import numpy as np
import random
import pickle
import json
import scipy.ndimage
import imageio
import math
import torch
import pyvista
from torch.utils.data import Dataset
from scipy.sparse import csr_matrix
import torchvision.transforms.functional as tvf

# train_transform = Compose(
#     [
#         Flip,
#         Rotate90,
#         ToTensor,
#     ]
# )
train_transform = []
# val_transform = Compose(
#     [
#         ToTensor,
#     ]
# )
val_transform = []

class Sat2GraphDataLoader(Dataset):
    """[summary]

    Args:
        Dataset ([type]): [description]
    """
    def __init__(self, data, transform):
        """[summary]

        Args:
            data ([type]): [description]
            transform ([type]): [description]
        """
        self.data = data
        self.transform = transform

        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]
    
    def __len__(self):
        """[summary]

        Returns:
            [type]: [description]
        """
        return len(self.data)
    
    def __getitem__(self, idx):
        """[summary]

        Args:
            idx ([type]): [description]

        Returns:
            [type]: [description]
        """
        data = self.data[idx]
        image_data = imageio.imread(data['img'])
        image_data = torch.tensor(image_data, dtype=torch.float).permute(2,0,1)
        image_data = image_data/255.0
        vtk_data = pyvista.read(data['vtp'])
        seg_data = imageio.imread(data['seg'])
        seg_data = seg_data/np.max(seg_data)
        seg_data = torch.tensor(seg_data, dtype=torch.int).unsqueeze(0)

        image_data = tvf.normalize(torch.tensor(image_data, dtype=torch.float), mean=self.mean, std=self.std)


        # correction of shift in the data
        # shift = [np.shape(image_data)[0]/2 -1.8, np.shape(image_data)[1]/2 + 8.3, 4.0]
        # coordinates = np.float32(np.asarray(vtk_data.points))
        # lines = np.asarray(vtk_data.lines.reshape(-1, 3))

        coordinates = torch.tensor(np.float32(np.asarray(vtk_data.points)), dtype=torch.float)
        lines = torch.tensor(np.asarray(vtk_data.lines.reshape(-1, 3)), dtype=torch.int64)

        return image_data, seg_data-0.5, coordinates[:,:2], lines[:,1:]


def build_road_network_data(config, mode='train', split=0.95):
    """[summary]

    Args:
        data_dir (str, optional): [description]. Defaults to ''.
        mode (str, optional): [description]. Defaults to 'train'.
        split (float, optional): [description]. Defaults to 0.8.

    Returns:
        [type]: [description]
    """    
    img_folder = os.path.join(config.DATA.DATA_PATH, 'raw')
    seg_folder = os.path.join(config.DATA.DATA_PATH, 'seg')
    vtk_folder = os.path.join(config.DATA.DATA_PATH, 'vtp')
    img_files = []
    vtk_files = []
    seg_files = []

    for file_ in os.listdir(img_folder):
        file_ = file_[:-8]
        img_files.append(os.path.join(img_folder, file_+'data.png'))
        vtk_files.append(os.path.join(vtk_folder, file_+'graph.vtp'))
        seg_files.append(os.path.join(seg_folder, file_+'seg.png'))

    data_dicts = [
        {"img": img_file, "vtp": vtk_file, "seg": seg_file} for img_file, vtk_file, seg_file in zip(img_files, vtk_files, seg_files)
        ]
    if mode=='train':
        ds = Sat2GraphDataLoader(
            data=data_dicts,
            transform=train_transform,
        )
        return ds
    elif mode=='test':
        img_folder = os.path.join(config.DATA.TEST_DATA_PATH, 'raw')
        seg_folder = os.path.join(config.DATA.TEST_DATA_PATH, 'seg')
        vtk_folder = os.path.join(config.DATA.TEST_DATA_PATH, 'vtp')
        img_files = []
        vtk_files = []
        seg_files = []

        for file_ in os.listdir(img_folder):
            file_ = file_[:-8]
            img_files.append(os.path.join(img_folder, file_+'data.png'))
            vtk_files.append(os.path.join(vtk_folder, file_+'graph.vtp'))
            seg_files.append(os.path.join(seg_folder, file_+'seg.png'))

        data_dicts = [
            {"img": img_file, "vtp": vtk_file, "seg": seg_file} for img_file, vtk_file, seg_file in zip(img_files, vtk_files, seg_files)
            ]
        ds = Sat2GraphDataLoader(
            data=data_dicts,
            transform=val_transform,
        )
        return ds
    elif mode=='split':
        img_folder = os.path.join(config.DATA.DATA_PATH, 'raw')
        seg_folder = os.path.join(config.DATA.DATA_PATH, 'seg')
        vtk_folder = os.path.join(config.DATA.DATA_PATH, 'vtp')
        img_files = []
        vtk_files = []
        seg_files = []

        for file_ in os.listdir(img_folder):
            file_ = file_[:-8]
            img_files.append(os.path.join(img_folder, file_+'data.png'))
            vtk_files.append(os.path.join(vtk_folder, file_+'graph.vtp'))
            seg_files.append(os.path.join(seg_folder, file_+'seg.png'))

        data_dicts = [
            {"img": img_file, "vtp": vtk_file, "seg": seg_file} for img_file, vtk_file, seg_file in zip(img_files, vtk_files, seg_files)
            ]
        random.seed(config.DATA.SEED)
        random.shuffle(data_dicts)
        train_split = int(split*len(data_dicts))
        train_files, val_files = data_dicts[:train_split], data_dicts[train_split:]
        train_ds = Sat2GraphDataLoader(
            data=train_files,
            transform=train_transform,
        )
        val_ds = Sat2GraphDataLoader(
            data=val_files,
            transform=val_transform,
        )
        return train_ds, val_ds