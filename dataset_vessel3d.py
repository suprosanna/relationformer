"""
@author: suprosanna
"""
from numpy.core.arrayprint import dtype_is_implied
import torch
import math
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from monai.data import Dataset
import torchvision.transforms.functional as tvf
from torchvision import transforms
from medpy.io import load
import pyvista
import numpy as np
import random
import os
from scipy import ndimage
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from skimage.morphology import skeletonize_3d
from utils import Bresenham3D
import itertools

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

class vessel_loader(Dataset):
    """[summary]

    Args:
        Dataset ([type]): [description]
    """
    def __init__(self, data, transform, num_patch, patch_size=(64,64,64), pad=(5,5,5), rand_patch=True):
        """[summary]

        Args:
            data ([type]): [description]
            transform ([type]): [description]
        """
        self.data = data
        self.transform = transform
        self.num_patch = num_patch
        self.patch_size = patch_size
        self.pad = pad
        self.rand_patch=rand_patch
    
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
        image_data, _ = load(data['nifti'])
        image_data = torch.tensor(image_data, dtype=torch.float).unsqueeze(0).unsqueeze(0)
        vmax = image_data.max()*0.001
        image_data = image_data/vmax
        vtk_data = pyvista.read(data['vtp'])
        seg_data, _ = load(data['seg'])
        seg_data = torch.tensor(seg_data, dtype=torch.int).unsqueeze(0).unsqueeze(0)



        # correction of shift in the data
        # shift = [np.shape(image_data)[0]/2 -1.8, np.shape(image_data)[1]/2 + 8.3, 4.0]
        # coordinates = np.float32(np.asarray(vtk_data.points/3.0+shift))
        # lines = np.asarray(vtk_data.lines.reshape(vtk_data.n_cells, 3))

        coordinates = torch.tensor(np.float32(np.asarray(vtk_data.points)), dtype=torch.float)
        lines = torch.tensor(np.asarray(vtk_data.lines.reshape(-1, 3)), dtype=torch.int64)

        return [image_data-0.5], [seg_data], [coordinates], [lines[:,1:]]


def build_vessel_data(config, mode='train', split=0.95):
    """[summary]

    Args:
        data_dir (str, optional): [description]. Defaults to ''.
        mode (str, optional): [description]. Defaults to 'train'.
        split (float, optional): [description]. Defaults to 0.8.

    Returns:
        [type]: [description]
    """    
    nifti_folder = os.path.join(config.DATA.DATA_PATH, 'raw')
    seg_folder = os.path.join(config.DATA.DATA_PATH, 'seg')
    vtk_folder = os.path.join(config.DATA.DATA_PATH, 'vtp')
    nifti_files = []
    vtk_files = []
    seg_files = []

    for i,file_ in enumerate(os.listdir(nifti_folder)):
        file_ = file_[:-7]
        nifti_files.append(os.path.join(nifti_folder, file_+'.nii.gz'))
        vtk_files.append(os.path.join(vtk_folder, file_[:-4]+'graph.vtp'))
        seg_files.append(os.path.join(seg_folder, file_[:-4]+'seg.nii.gz'))
        # if i>45000:
        #     break

    data_dicts = [
        {"nifti": nifti_file, "vtp": vtk_file, "seg": seg_file} for nifti_file, vtk_file, seg_file in zip(nifti_files, vtk_files, seg_files)
        ]
    if mode=='train':
        ds = vessel_loader(
            data=data_dicts,
            transform=train_transform,
            num_patch=config.DATA.NUM_PATCH,
            patch_size=config.DATA.IMG_SIZE,
            rand_patch=True,
        )
        return ds
    elif mode=='val':
        ds = vessel_loader(
            data=data_dicts,
            transform=val_transform,
            num_patch=config.DATA.NUM_PATCH,
            patch_size=config.DATA.IMG_SIZE,
            rand_patch=False,
        )
        return ds
    elif mode=='split':
        random.seed(config.DATA.SEED)
        random.shuffle(data_dicts)
        train_split = int(split*len(data_dicts))
        train_files, val_files = data_dicts[:train_split], data_dicts[train_split:]
        train_ds = vessel_loader(
            data=train_files,
            transform=train_transform,
            num_patch=config.DATA.NUM_PATCH,
            patch_size=config.DATA.IMG_SIZE,
        )
        val_ds = vessel_loader(
            data=val_files,
            transform=val_transform,
            num_patch=config.DATA.NUM_PATCH,
            patch_size=config.DATA.IMG_SIZE,
        )
        return train_ds, val_ds