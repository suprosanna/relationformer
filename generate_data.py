import math
from tqdm import tqdm
from medpy.io import load, save
import pyvista
import numpy as np
import os
from scipy.sparse import csr_matrix

patch_size = [64,64,64]
pad = [5,5,5]

def angle(v1, v2):
    unit_vector_1 = v1 / np.linalg.norm(v1)
    unit_vector_2 = v2 / np.linalg.norm(v2)
    dot_product = np.dot(unit_vector_1, unit_vector_2)
    return np.arccos(np.clip(dot_product, a_min = -1, a_max=1))


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
        if (p3>=surface_val[:3]).all() and (p3<=surface_val[3:]).all():
            return np.expand_dims(np.array(p3), 0)
        else:
            continue

def save_input(save_path, idx, patch, patch_seg, patch_coord, patch_edge):
    """[summary]

    Args:
        patch ([type]): [description]
        patch_coord ([type]): [description]
        patch_edge ([type]): [description]
    """
    save(patch, save_path+'raw/sample_'+str(idx).zfill(6)+'_data.nii.gz')
    save(patch_seg, save_path+'seg/sample_'+str(idx).zfill(6)+'_seg.nii.gz')
    
    patch_edge = np.concatenate((np.int32(2*np.ones((patch_edge.shape[0],1))), patch_edge), 1)
    mesh = pyvista.PolyData(patch_coord)
    # print(patch_edge.shape)
    mesh.lines = patch_edge.flatten()
    mesh.save(save_path+'vtp/sample_'+str(idx).zfill(6)+'_graph.vtp')


def patch_extract(save_path, image, seg,  mesh, device=None):
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
    global image_id
    # TODO: edge on the boundary of patch not included, edge which passes through the volume not included
    p_h, p_w, p_d = patch_size
    pad_h, pad_w, pad_d = pad

    p_h = p_h -2*pad_h
    p_w = p_w -2*pad_w
    p_d = p_d -2*pad_d
    
    h, w, d= image.shape
    x_ = np.int32(np.linspace(20, h-20-p_h, 10))
    y_ = np.int32(np.linspace(20, w-20-p_w, 10))
    z_ = np.int32(np.linspace(20, d-20-p_d, 20))
    
    ind = np.meshgrid(x_, y_, z_, indexing='ij')
    # Center Crop based on foreground

    for i, start in enumerate(list(np.array(ind).reshape(3,-1).T)):
        # print(image.shape, seg.shape)
        end = start + np.array(patch_size)-1 -2*np.array(pad)
        patch = np.pad(image[start[0]:start[0]+p_h, start[1]:start[1]+p_w, start[2]:start[2]+p_d], ((pad_h,pad_h),(pad_w,pad_w),(pad_d,pad_d)))
        patch_list = [patch]

        patch_seg = np.pad(seg[start[0]:start[0]+p_h, start[1]:start[1]+p_w, start[2]:start[2]+p_d], ((pad_h,pad_h),(pad_w,pad_w),(pad_d,pad_d)))
        seg_list = [patch_seg]

        # collect all the nodes
        bounds = [start[0], end[0], start[1], end[1], start[2], end[2]]
        clipped_mesh = mesh.clip_box(bounds, invert=False)
        patch_coordinates = np.float32(np.asarray(clipped_mesh.points))
        patch_edge = clipped_mesh.cells[np.sum(clipped_mesh.celltypes==1)*2:].reshape(-1,3)
        
        patch_coord_ind = np.where((np.prod(patch_coordinates>=start, 1)*np.prod(patch_coordinates<=end, 1))>0.0)
        patch_coordinates = patch_coordinates[patch_coord_ind[0], :]  # all coordinates inside the patch
        patch_edge = [tuple(l) for l in patch_edge[:,1:] if l[0] in patch_coord_ind[0] and l[1] in patch_coord_ind[0]]
        
        temp = np.array(patch_edge).flatten()  # flatten all the indices of the edges which completely lie inside patch
        temp = [np.where(patch_coord_ind[0] == ind) for ind in temp]  # remap the edge indices according to the new order
        patch_edge = list(np.array(temp).reshape(-1,2))  # reshape the edge list into previous format

        # concatenate final variables
        patch_coordinates = (patch_coordinates-start+np.array(pad))/np.array(patch_size)
        patch_coord_list = [patch_coordinates]#.to(device))
        patch_edge_list = [np.array(patch_edge)]#.to(device))

        mod_patch_coord_list, mod_patch_edge_list = prune_patch(patch_coord_list, patch_edge_list)

        # save data
        for patch, patch_seg, patch_coord, patch_edge in zip(patch_list, seg_list, mod_patch_coord_list, mod_patch_edge_list):
            if patch_seg.sum()>10:
                save_input(save_path, image_id, patch, patch_seg, patch_coord, patch_edge)
                image_id = image_id+1
                # print('Image No', image_id)

def prune_patch(patch_coord_list, patch_edge_list):
    """[summary]

    Args:
        patch_list ([type]): [description]
        patch_coord_list ([type]): [description]
        patch_edge_list ([type]): [description]

    Returns:
        [type]: [description]
    """
    mod_patch_coord_list = []
    mod_patch_edge_list = []

    for coord, edge in zip(patch_coord_list, patch_edge_list):
        dist_adj = np.zeros((coord.shape[0], coord.shape[0]))
        dist_adj[edge[:,0], edge[:,1]] = np.sum((coord[edge[:,0],:]-coord[edge[:,1],:])**2, 1)
        dist_adj[edge[:,1], edge[:,0]] = np.sum((coord[edge[:,0],:]-coord[edge[:,1],:])**2, 1)

        # straighten the graph by removing redundant nodes
        start = True
        node_mask = np.ones(coord.shape[0], dtype=np.bool)
        while start:
            degree = (dist_adj>0).sum(1)
            deg_2 = list(np.where(degree==2)[0])
            # print('Most likely running an infinite loop', deg_2)
            if len(deg_2)==0:
                start = False
            for n, idx in enumerate(deg_2):
                deg_2_neighbor = np.where(dist_adj[idx,:]>0)[0]

                p1 = coord[idx,:]
                p2 = coord[deg_2_neighbor[0],:]
                p3 = coord[deg_2_neighbor[1],:]
                l1 = p2-p1
                l2 = p3-p1
                node_angle = angle(l1,l2)*180 / math.pi
                if node_angle>160:
                    node_mask[idx]=False
                    dist_adj[deg_2_neighbor[0], deg_2_neighbor[1]] = np.sum((p2-p3)**2)
                    dist_adj[deg_2_neighbor[1], deg_2_neighbor[0]] = np.sum((p2-p3)**2)

                    dist_adj[idx, deg_2_neighbor[0]] = 0.0
                    dist_adj[deg_2_neighbor[0], idx] = 0.0
                    dist_adj[idx, deg_2_neighbor[1]] = 0.0
                    dist_adj[deg_2_neighbor[1], idx] = 0.0
                    break
                elif n==len(deg_2)-1:
                    start = False

        new_coord = coord[node_mask,:]
        new_dist_adj = dist_adj[np.ix_(node_mask, node_mask)]
        new_edge = np.array(np.where(np.triu(new_dist_adj)>0)).T

        mod_patch_coord_list.append(new_coord)
        mod_patch_edge_list.append(new_edge)
        # mod_patch_list.append(new_patch)

    return mod_patch_coord_list, mod_patch_edge_list


if __name__ == "__main__":
    DATA_PATH = './data/vessel_data/'

    img_folder = os.path.join(DATA_PATH, 'raw')
    seg_folder = os.path.join(DATA_PATH, 'seg')
    vtk_folder = os.path.join(DATA_PATH, 'vtk')

    raw_files = []
    seg_files = []
    vtk_files = []

    for file_ in os.listdir(seg_folder):
        file_ = file_[:-7]
        raw_files.append(os.path.join(img_folder, file_+'.nii.gz'))
        seg_files.append(os.path.join(seg_folder, file_+'.nii.gz'))
        vtk_files.append(os.path.join(vtk_folder, file_+'.vtk'))

    image_id = 1
    train_path = './data/vessel_data/train_data/'
    if not os.path.isdir(train_path):
        os.makedirs(train_path)
        os.makedirs(train_path+'/seg')
        os.makedirs(train_path+'/vtp')
        os.makedirs(train_path+'/raw')
    else:
        raise Exception("Train folder is non-empty")
    print('Preparing Train Data')
    for idx, seg_file in tqdm(enumerate(seg_files[:40])):
        image_data, _ = load(raw_files[idx])
        image_data = np.int32(image_data)
        seg_data, _ = load(seg_files[idx])
        seg_data = np.int8(seg_data)
        vtk_data = pyvista.read(vtk_files[idx])

        # correction of shift in the data
        shift = [np.shape(image_data)[0]/2 -1.8, np.shape(image_data)[1]/2 + 8.3, 4.0]
        coordinates = np.float32(np.asarray(vtk_data.points/3.0+shift))
        # lines = np.asarray(vtk_data.lines.reshape(vtk_data.n_cells, 3))
        
        vtk_data.points = coordinates

        # if self.transform:
        patch_extract(train_path, image_data, seg_data, vtk_data)

    image_id = 1
    test_path = './data/vessel_data/test_data/'
    if not os.path.isdir(test_path):
        os.makedirs(test_path)
        os.makedirs(test_path+'/seg')
        os.makedirs(test_path+'/vtp')
        os.makedirs(test_path+'/raw')
    else:
        raise Exception("Test folder is non-empty")

    print('Preparing Test Data')
    for idx, seg_file in tqdm(enumerate(seg_files[40:50])):
        image_data, _ = load(raw_files[idx])
        image_data = np.int32(image_data)
        seg_data, _ = load(seg_files[idx])
        seg_data = np.int8(seg_data)
        vtk_data = pyvista.read(vtk_files[idx])

        # correction of shift in the data
        shift = [np.shape(image_data)[0]/2 -1.8, np.shape(image_data)[1]/2 + 8.3, 4.0]
        coordinates = np.float32(np.asarray(vtk_data.points/3.0+shift))
        # lines = np.asarray(vtk_data.lines.reshape(vtk_data.n_cells, 3))
        
        vtk_data.points = coordinates

        # if self.transform:
        patch_extract(test_path, image_data, seg_data, vtk_data)