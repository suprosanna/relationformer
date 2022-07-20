"""
@author: suprosanna
"""
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as tvf
from torchvision import transforms, datasets
from medpy.io import load
import pyvista
import time
import pickle
import numpy as np
import random
import os
from skimage.morphology import skeletonize_3d

SEED = 120
# train_transform = Compose(
#     [
#         LoadImaged(keys=ALL_KEYS),
#         AddChanneld(keys=ALL_KEYS),
#         RandCropByPosNegLabeld(  # crop with center in label>0 with proba pos / (neg + pos)
#             keys=ALL_KEYS,
#             label_key="label",
#             spatial_size=PATCH_SIZE,
#             pos=1,
#             neg=1,
#             num_samples=1,
#             image_key=None,  # for no restriction with image thresholding
#             image_threshold=0,
#         ),
#         RandFlipd(ALL_KEYS, spatial_axis=[0, 1, 2], prob=0.5),
#         CastToTyped(keys=ALL_KEYS, dtype=(np.float32,) * len(IMAGE_KEYS) + (np.uint8,)),
#         ToTensord(keys=ALL_KEYS),
#     ]
# )
train_transform = []
# val_transform = Compose(
#     [
#         LoadImaged(keys=ALL_KEYS),
#         AddChanneld(keys=ALL_KEYS),
#         CastToTyped(keys=ALL_KEYS, dtype=(np.float32,)*len(IMAGE_KEYS)+(np.uint8,)),
#         ToTensord(keys=ALL_KEYS),
#     ]
# )
val_transform = []

class vessel_loader(Dataset):
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
        vtk_data = pyvista.read(data['vtk'])

        # correction of shift in the data
        shift = [np.shape(image_data)[0]/2 -1.8, np.shape(image_data)[1]/2 + 8.3, 4.0]
        coordinates = np.asarray(vtk_data.points/3.0+shift)
        lines = np.asarray(vtk_data.lines.reshape(vtk_data.n_cells, 3))

        # if self.transform:
        patch_list, patch_coord_list, patch_edge_list = patch_extract(image_data, coordinates, lines, num_patch=2)

        return patch_list, patch_coord_list, patch_edge_list


def get_indices_sparse(data):
    """[summary]

    Args:
        data ([type]): [description]

    Returns:
        [type]: [description]
    """
    M = compute_M(data)
    return [np.unravel_index(row.data, data.shape) for row in M[1:]]


def compute_M(data):
    """[summary]

    Args:
        data ([type]): [description]

    Returns:
        [type]: [description]
    """
    cols = np.arange(data.size)
    return csr_matrix((cols, (data.ravel(), cols)), shape=(data.max() + 1, data.size))

def solve_line(p1, p2, val, dim):
    if dim==0:
        x = val
        y = p1[1]+(val-p1[0])*(p2[1]-p1[1])/(p2[0]-p1[0])
        z = p1[2]+(val-p1[0])*(p2[2]-p1[2])/(p2[0]-p1[0])
    elif dim==1:
        x = p1[0]+(val-p1[1])*(p2[0]-p1[0])/(p2[1]-p1[1])
        y = val
        z = p1[2]+(val-p1[1])*(p2[2]-p1[2])/(p2[1]-p1[1])
    elif dim==2:
        x = p1[0]+(val-p1[2])*(p2[0]-p1[0])/(p2[2]-p1[2])
        y = p1[1]+(val-p1[2])*(p2[1]-p1[1])/(p2[2]-p1[2])
        z = val
    return [x,y,z]

def find_intersect(p1, p2, check, surface_val):
    inds = [i for i, x in enumerate(check) if x]
    for ind in inds:
        val = surface_val[ind]#(ind_+1)//3
        dim = (ind)%3
        p3 = solve_line(p1, p2, val, dim)
        if p3>=surface_val[:3] and p3<=surface_val[3:]:
            return np.expand_dims(np.array(p3), 0)


def patch_extract(image, coordinates, lines, patch_size=(64,64,64), num_patch=2):
    """[summary]

    Args:
        image ([type]): [description]
        coordinates ([type]): [description]
        lines ([type]): [description]
        patch_size (tuple, optional): [description]. Defaults to (64,64,64).
        num_patch (int, optional): [description]. Defaults to 2.

    Returns:
        [type]: [description]
    """
    # TODO: edge on the boundary of patch not included, edge which passes through the volume not included
    x_bounds =  np.shape(image)[0]/2 +2.0
    y_bounds =  np.shape(image)[1]/2 + 2.0
    z_bounds =  -4.0

    p_h, p_w, p_d = patch_size

    ## preprocess to remove unnecessary part and find patch based on the skeleton in the center
    mask = np.zeros(image.shape)
    mask[10:-10-p_h,35:-10-p_w,:-20-p_d]=1
    skel = skeletonize_3d(image)*(mask>0.0)
    ind = np.where(skel == 1)
    ind = [(i,j,k) for i, j, k in zip(ind[0], ind[1], ind[2])]

    random.shuffle(ind)

    patch_list = []
    patch_coord_list = []
    patch_edge_list = []

    for i in range(num_patch):
        start = ind[i]
        end = tuple(np.array(start) + np.array(patch_size)-1)
        patch = image[start[0]:start[0]+p_h, start[1]:start[1]+p_w, start[2]:start[2]+p_d]
        patch_list.append(patch)

        # collect all the nodes
        patch_coord_ind = np.where((np.prod(coordinates>=start, 1)*np.prod(coordinates<=end, 1))>0.0)
        patch_coordinates = coordinates[patch_coord_ind[0], :]

        # collect all the edges inside the volume
        patch_line_ind_inside = [tuple(l) for l in lines[:,1:] if l[0] in patch_coord_ind[0] and l[1] in patch_coord_ind[0]]

        # collect all the edges going outside from the volume
        patch_line_ind_outside = [tuple(l) for l in lines[:,1:] if l[0] in patch_coord_ind[0] and l[1] not in patch_coord_ind[0]] +\
            [tuple(np.flip(l)) for l in lines[:,1:] if l[0] not in patch_coord_ind[0] and l[1] in patch_coord_ind[0]]

        # find new edges at the boundary of the volume
        temp = np.array(patch_line_ind_inside).flatten()
        new_ind = list(range(patch_coord_ind[0].shape[0]))
        arr = np.empty(temp.max() + 1, dtype=patch_coord_ind[0].dtype)
        arr[patch_coord_ind[0]]=new_ind
        patch_line_ind_inside = list(arr[temp].reshape(-1,2))

        surface_val = list(start+end)
        new_ind = patch_coordinates.shape[0]
        for line in patch_line_ind_outside:
            p1 = coordinates[line[0], :]
            p2 = coordinates[line[1], :]
            check = list(p2<start)+list(p2>end)
            p3 = find_intersect(p1, p2, check, surface_val)
            patch_coordinates = np.concatenate((patch_coordinates, p3), axis=0)
            patch_line_ind_inside.append(np.array([np.where(patch_coord_ind[0] == line[0])[0][0], new_ind]))
            new_ind = new_ind+1
        patch_coordinates = patch_coordinates-start
        patch_coord_list.append(patch_coordinates)
        patch_edge_list.append(np.array(patch_line_ind_inside))

    return patch_list, patch_coord_list, patch_edge_list


def prune_patch(patch_list, patch_coord_list, patch_edge_list):  # TODO:
    """[summary]

    Args:
        patch_list ([type]): [description]
        patch_coord_list ([type]): [description]
        patch_edge_list ([type]): [description]

    Returns:
        [type]: [description]
    """


    return patch_list, patch_coord_list, patch_edge_list


def augment_patch(patch_list, patch_coord_list, patch_edge_list):  # TODO:
    """[summary]

    Args:
        patch_list ([type]): [description]
        patch_coord_list ([type]): [description]
        patch_edge_list ([type]): [description]

    Returns:
        [type]: [description]
    """

    return patch_list, patch_coord_list, patch_edge_list


def save_patch(patch, patch_coord, patch_edge):
    """[summary]

    Args:
        patch ([type]): [description]
        patch_coord ([type]): [description]
        patch_edge ([type]): [description]
    """
    vertices, faces, _, _ = marching_cubes_lewiner(patch)
    vertices = vertices
    faces = np.concatenate((np.int32(3*np.ones((faces.shape[0],1))), faces), 1)
    patch_edge = np.concatenate((np.int32(2*np.ones((patch_edge.shape[0],1))), patch_edge), 1)
    
    mesh = pyvista.PolyData(vertices)
    mesh.faces = faces.flatten()
    print(mesh, mesh.points.shape, mesh.faces.shape)
    mesh.save('./segmentation.stl')
    
    mesh = pyvista.PolyData(patch_coord)
    mesh.lines = patch_edge.flatten()
    print(mesh, mesh.points.shape, mesh.lines.shape)
    mesh.save('./graph.vtp')


def build_vessel_data(data_dir='', mode='train', split=0.8):
    """[summary]

    Args:
        data_dir (str, optional): [description]. Defaults to ''.
        mode (str, optional): [description]. Defaults to 'train'.
        split (float, optional): [description]. Defaults to 0.8.

    Returns:
        [type]: [description]
    """    
    nifti_folder = os.path.join(data_dir, 'seg')
    vtk_folder = os.path.join(data_dir, 'vtk')
    nifti_files = []
    vtk_files = []

    for file_ in os.listdir(nifti_folder):
        file_ = file_[:-7]
        nifti_files.append(os.path.join(nifti_folder, file_+'.nii.gz'))
        vtk_files.append(os.path.join(vtk_folder, file_+'.vtk'))

    data_dicts = [
        {"nifti": nifti_file, "vtk": vtk_file} for nifti_file, vtk_file in zip(nifti_files, vtk_files)
        ]
    if mode=='train':
        ds = vessel_loader(
            data=data_dicts,
            transform=train_transform
        )
        return ds
    elif mode=='val':
        ds = vessel_loader(
            data=data_dicts,
            transform=val_transform
        )
        return ds
    elif mode=='split':
        random.seed(SEED)
        random.shuffle(data_dicts)
        train_split = int(split*len(data_dicts))
        train_files, val_files = data_dicts[:train_split], data_dicts[train_split:]
        train_ds = vessel_loader(
            data=train_files,
            transform=train_transform
        )
        val_ds = vessel_loader(
            data=val_files,
            transform=val_transform
        )
        return train_ds, val_ds