
import torch
import torch.nn.functional as F
import numpy as np
import logging
from mmcv.utils import get_logger
import pyvista
from skimage.measure import marching_cubes_lewiner
from scipy.ndimage.morphology import grey_dilation
from scipy import ndimage
from itertools import product
import pdb

def image_graph_collate(batch):
    images = torch.cat([item_ for item in batch for item_ in item[0]], 0).contiguous()
    segs = torch.cat([item_ for item in batch for item_ in item[1]], 0).contiguous()
    points = [item_ for item in batch for item_ in item[2]]
    edges = [item_ for item in batch for item_ in item[3]]
    return [images, segs, points, edges]


def get_root_logger(log_file=None, log_level=logging.INFO):
    """Use ``get_logger`` method in mmcv to get the root logger.
    The logger will be initialized if it has not been initialized. By default a
    StreamHandler will be added. If ``log_file`` is specified, a FileHandler
    will also be added. The name of the root logger is the top-level package
    name, e.g., "mmaction".
    Args:
        log_file (str | None): The log filename. If specified, a FileHandler
            will be added to the root logger.
        log_level (int): The root logger level. Note that only the process of
            rank 0 is affected, while other processes will set the level to
            "Error" and be silent most of the time.
    Returns:
        :obj:`logging.Logger`: The root logger.
    """
    return get_logger(__name__.split('.')[0], log_file, log_level)

def save_input(path, idx, patch, patch_coord, patch_edge):
    """[summary]

    Args:
        patch ([type]): [description]
        patch_coord ([type]): [description]
        patch_edge ([type]): [description]
    """
    
    # vertices, faces, _, _ = marching_cubes_lewiner(patch)
    # vertices = vertices/np.array(patch.shape)
    # faces = np.concatenate((np.int32(3*np.ones((faces.shape[0],1))), faces), 1)
    
    # mesh = pyvista.PolyData(vertices)
    # mesh.faces = faces.flatten()
    # mesh.save(path+'_sample_'+str(idx).zfill(3)+'_segmentation.stl')
    
    patch_edge = np.concatenate((np.int32(2*np.ones((patch_edge.shape[0],1))), patch_edge), 1)
    mesh = pyvista.PolyData(patch_coord)
    # print(patch_edge.shape)
    mesh.lines = patch_edge.flatten()
    mesh.save(path+'_sample_'+str(idx).zfill(3)+'_graph.vtp')
    
    
def save_output(path, idx, patch_coord, patch_edge):
    """[summary]

    Args:
        patch ([type]): [description]
        patch_coord ([type]): [description]
        patch_edge ([type]): [description]
    """
    print('Num nodes:', patch_coord.shape[0], 'Num edges:', patch_edge.shape[0])
    patch_edge = np.concatenate((np.int32(2*np.ones((patch_edge.shape[0],1))), patch_edge), 1)
    mesh = pyvista.PolyData(patch_coord)
    if patch_edge.shape[0]>0:
        mesh.lines = patch_edge.flatten()
    mesh.save(path+'_sample_'+str(idx).zfill(3)+'_graph.vtp')


def patchify_voxel(volume, patch_size, pad):
    p_h, p_w, p_d = patch_size
    pad_h, pad_w, pad_d = pad

    p_h = p_h -2*pad_h
    p_w = p_w -2*pad_w
    p_d = p_d -2*pad_d
    
    
    v_h, v_w, v_d = volume.shape

    # Calculate the number of patch in ach axis
    n_w = np.ceil(1.0*(v_w-p_w)/p_w+1)
    n_h = np.ceil(1.0*(v_h-p_h)/p_h+1)
    n_d = np.ceil(1.0*(v_d-p_d)/p_d+1)

    n_w = int(n_w)
    n_h = int(n_h)
    n_d = int(n_d)

    pad_1 = (n_w - 1) * p_w + p_w - v_w
    pad_2 = (n_h - 1) * p_h + p_h - v_h
    pad_3 = (n_d - 1) * p_d + p_d - v_d

    volume = np.pad(volume, ((0, pad_1), (0, pad_2), (0, pad_3)), mode='reflect')
    
    h, w, d= volume.shape
    x_ = np.int32(np.linspace(0, h-p_h, n_w))
    y_ = np.int32(np.linspace(0, w-p_w, n_h))
    z_ = np.int32(np.linspace(0, d-p_d, n_d))
    
    ind = np.meshgrid(x_, y_, z_, indexing='ij')
    
    patch_list = []
    start_ind = []
    seq_ind = []
    for i, start in enumerate(list(np.array(ind).reshape(3,-1).T)):
        patch = np.pad(volume[start[0]:start[0]+p_h, start[1]:start[1]+p_w, start[2]:start[2]+p_d], ((pad_h,pad_h),(pad_w,pad_w),(pad_d,pad_d)))
        patch_list.append(patch)
        start_ind.append(start)
        seq_ind.append([i//(y_.shape[0]*z_.shape[0]), (i%(y_.shape[0]*z_.shape[0]))//z_.shape[0], (i%(y_.shape[0]*z_.shape[0]))%z_.shape[0]])
        
    return patch_list, start_ind, seq_ind, volume.shape


def unpatchify_graph(patch_graphs, start_ind, seq_ind, pad, imsize=[128,128,128]):
    """

    :param patches:
    :param step:
    :param imsize:
    :param scale_factor:
    :return:
    """
    patch_coords, patch_edges = patch_graphs['pred_nodes'], patch_graphs['pred_rels']
    occu_matrix = np.empty((8,)+imsize)  # 8 channel occu matrix
    pred_coords = []
    pred_rels = []
    num_nodes = 0
    struct = ndimage.generate_binary_structure(3, 2)
    for i, (patch_coord, patch_edge) in enumerate(zip(patch_coords, patch_edges)):
        patch_node_label = np.zeros(imsize)
        abs_patch_coord = np.array(start_ind[i]-pad) + patch_coord*64
        pred_coords.extend(abs_patch_coord)
        abs_patch_coord = np.int64(abs_patch_coord)
        ch_idx = np.sum(2**(np.array(range(3))[::-1])*(np.array(seq_ind[i])%2))
        # print(start_ind[i], seq_ind[i], np.array(seq_ind[i])%2, ch_idx)

        # local patch occupancy
        patch_node_label[abs_patch_coord[:,0],abs_patch_coord[:,1],abs_patch_coord[:,2]] = np.array(list(range(num_nodes,num_nodes+patch_coord.shape[0])))+1

        # dialate each node regions in isotropic way
        
        # occu_matrix[ch_idx, start_ind[i][0]-pad[0]:start_ind[i][0]-pad[0]+64, start_ind[i][1]-pad[1]:start_ind[i][1]-pad[1]+64, start_ind[i][2]-pad[2]:start_ind[i][2]-pad[2]+64] = 1
        for _ in range(8):
            inst_label = grey_dilation(patch_node_label, footprint=struct) #size=(3,3,3)) # structure=struct)
            inst_label[patch_node_label>0] = patch_node_label[patch_node_label>0]
            patch_node_label = inst_label
            occu_matrix[ch_idx, patch_node_label>0] = patch_node_label[patch_node_label>0]

        # occu_matrix[patch_node_label>0.0] = patch_node_label[patch_node_label>0.0]
    
        pred_rels.extend(patch_edge+num_nodes)
        num_nodes = num_nodes+patch_coord.shape[0]
    
    pred_graph = {'pred_nodes':pred_coords,'pred_rels':pred_rels}
    return occu_matrix, pred_graph

    
def Bresenham3D(p1, p2): 
    """
    Function to compute direct connection in voxel space
    """
    x1, y1, z1 = p1
    x2, y2, z2 = p2
    ListOfPoints = [] 
    ListOfPoints.append((x1, y1, z1)) 
    dx = abs(x2 - x1) 
    dy = abs(y2 - y1) 
    dz = abs(z2 - z1) 
    if (x2 > x1): 
        xs = 1
    else: 
        xs = -1
    if (y2 > y1): 
        ys = 1
    else: 
        ys = -1
    if (z2 > z1): 
        zs = 1
    else: 
        zs = -1
  
    # Driving axis is X-axis" 
    if (dx >= dy and dx >= dz):         
        p1 = 2 * dy - dx 
        p2 = 2 * dz - dx 
        while (x1 != x2): 
            x1 += xs 
            if (p1 >= 0): 
                y1 += ys 
                p1 -= 2 * dx 
            if (p2 >= 0): 
                z1 += zs 
                p2 -= 2 * dx 
            p1 += 2 * dy 
            p2 += 2 * dz 
            ListOfPoints.append((x1, y1, z1)) 
  
    # Driving axis is Y-axis" 
    elif (dy >= dx and dy >= dz):        
        p1 = 2 * dx - dy 
        p2 = 2 * dz - dy 
        while (y1 != y2): 
            y1 += ys 
            if (p1 >= 0): 
                x1 += xs 
                p1 -= 2 * dy 
            if (p2 >= 0): 
                z1 += zs 
                p2 -= 2 * dy 
            p1 += 2 * dx 
            p2 += 2 * dz 
            ListOfPoints.append((x1, y1, z1)) 
  
    # Driving axis is Z-axis" 
    else:         
        p1 = 2 * dy - dz 
        p2 = 2 * dx - dz 
        while (z1 != z2): 
            z1 += zs 
            if (p1 >= 0): 
                y1 += ys 
                p1 -= 2 * dz 
            if (p2 >= 0): 
                x1 += xs 
                p2 -= 2 * dz 
            p1 += 2 * dy 
            p2 += 2 * dx 
            ListOfPoints.append((x1, y1, z1)) 
    return ListOfPoints