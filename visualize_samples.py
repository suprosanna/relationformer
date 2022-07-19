import os
import pyvista
import open3d as o3d
import numpy as np
import open3d.visualization.gui as gui
from boxes import box_ops
import torch
import pdb
sample_path = '/home/home/supro/projects/relationformer/runs/baseline_synth_data_seresnet_def_detr_full_dataset_10/val_samples' # TODO: make it input

epoch_num = 50 # TODO: make it slider
num_sample = 8
num_iteration = 8
ref_line_sets = []
pred_line_sets = []
for i in range(0,0+num_sample):
    for j in range(0,0+num_iteration):
        ref_vtp = os.path.join(sample_path, "ref_epoch_"+str(epoch_num).zfill(3)+"_iteration_"+str(j+1).zfill(5)+'_sample_'+str(i).zfill(3)+'_graph.vtp')
        ref_mesh = pyvista.read(ref_vtp)
        ref_points = ref_mesh.points+np.array([i*1.5,j*1.5,0])
        ref_lines = np.asarray(ref_mesh.lines.reshape(-1, 3))[:,1:]
        ref_color = [[1, 0, 0] for i in range(len(ref_lines))]
        ref_line_set = o3d.geometry.LineSet(
                                        points=o3d.utility.Vector3dVector(ref_points),
                                        lines=o3d.utility.Vector2iVector(ref_lines),
                                    )
        ref_line_set.colors = o3d.utility.Vector3dVector(ref_color)
        ref_line_sets.append(ref_line_set)
        
        
        pred_vtp = os.path.join(sample_path, "pred_epoch_"+str(epoch_num).zfill(3)+"_iteration_"+str(j+1).zfill(5)+'_sample_'+str(i).zfill(3)+'_graph.vtp')
        pred_mesh = pyvista.read(pred_vtp)
        pred_points = pred_mesh.points+np.array([i*1.5,j*1.5,0])
        pred_lines = np.asarray(pred_mesh.lines.reshape(-1, 3))[:,1:]
        
        pred_lines1 = pred_lines
        pred_points1 = pred_points
        
        # try:
        #     A = np.zeros((pred_points.shape[0], pred_points.shape[0]))
        #     A[pred_lines[:,0],pred_lines[:,1]] = 1.0
        #     A[pred_lines[:,1],pred_lines[:,0]] = 1.0
            
        #     init_boxes = box_ops.box_cxcyczwhd_to_xyxyzz(torch.cat([torch.tensor(pred_points), 0.2*torch.ones(pred_points.shape)], dim=-1))
        #     init_score = torch.ones(init_boxes.shape[0])
        #     keep = boxes.nms(init_boxes.cpu(), init_score, 0.5)
        #     pred_points = pred_points[keep]
        #     A = np.triu(A[keep, :][:, keep])
        #     pred_lines = np.array(np.where(A)).T
        # except:
        #     pred_lines = pred_lines1
        #     pred_points = pred_points1
        
        pred_color = [[0, 0, 1] for i in range(len(pred_lines))]
        pred_line_set = o3d.geometry.LineSet(
                                        points=o3d.utility.Vector3dVector(pred_points),
                                        lines=o3d.utility.Vector2iVector(pred_lines),
                                    )
        pred_line_set.colors = o3d.utility.Vector3dVector(pred_color)
        pred_line_sets.append(pred_line_set)
        
o3d.visualization.draw_geometries(ref_line_sets+pred_line_sets)