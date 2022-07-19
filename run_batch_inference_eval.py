import os
import time
import torch
import torch.nn.functional as F
import yaml
import sys
import pdb
from medpy.io import load
import pyvista
import json
from argparse import ArgumentParser
import numpy as np
from models import build_model
from inference import relation_infer
from tqdm import tqdm
import csv
from functools import partial
from metrics.smd import compute_meanSMD, SinkhornDistance
from metrics.boxap import box_ap, iou_filter, get_unique_iou_thresholds, get_indices_of_iou_for_each_metric
from metrics.box_ops_np import box_iou_np
from metrics.coco import COCOMetric
from skimage.morphology import skeletonize_3d
import sknw
parser = ArgumentParser()
#TODO the same confg is used for all the models at the moment
parser.add_argument('--config',
                    default=None,
                    help='config file (.yml) containing the hyper-parameters for training. '
                         'If None, use the nnU-Net config.')
parser.add_argument('--model',
                    help='Paths to the checkpoints to use for inference separated by a space.')
parser.add_argument('--device', default='cuda',
                        help='device to use for training')
parser.add_argument('--eval', action='store_true', help='Apply evaluation of metrics')

class obj:
    def __init__(self, dict1):
        self.__dict__.update(dict1)
        
def dict2obj(dict1):
    return json.loads(json.dumps(dict1), object_hook=obj)


def main(args):
    """
    Run inference for all the testing data
    """
    # Load the config files
    with open(args.config) as f:
        print('\n*** Config file')
        print(args.config)
        config = yaml.load(f, Loader=yaml.FullLoader)
        print(config['log']['exp_name'])
    config = dict2obj(config)
    device = torch.device("cuda") if args.device=='cuda' else torch.device("cpu")

    nifti_folder = os.path.join(config.DATA.TEST_DATA_PATH, 'raw')
    seg_folder = os.path.join(config.DATA.TEST_DATA_PATH, 'seg')
    vtk_folder = os.path.join(config.DATA.TEST_DATA_PATH, 'vtp')
    nifti_files = []
    vtk_files = []
    seg_files = []

    for file_ in os.listdir(nifti_folder):
        file_ = file_[:-7]
        nifti_files.append(os.path.join(nifti_folder, file_+'.nii.gz'))
        seg_files.append(os.path.join(seg_folder, file_[:-4]+'seg.nii.gz'))
        if args.eval:
            vtk_files.append(os.path.join(vtk_folder, file_[:-4]+'graph.vtp'))

    net = build_model(config).to(device)
    
    # print('Loading model from:', args.model)
    checkpoint = torch.load(args.model, map_location='cpu')
    net.load_state_dict(checkpoint['net'])
    net.eval()  # Put the CNN in evaluation mode
    
    t_start = time.time()
    sinkhorn_distance = SinkhornDistance(eps=1e-7, max_iter=100)
    
    metrics = tuple([COCOMetric(classes=['Node'], per_class=False, verbose=False)])
    iou_thresholds = get_unique_iou_thresholds(metrics)
    iou_mapping = get_indices_of_iou_for_each_metric(iou_thresholds, metrics)
    box_evaluator = box_ap(box_iou_np, iou_thresholds, max_detections=40)
    
    mean_smd = []
    node_ap_result = []
    edge_ap_result = []
    for idx, _ in tqdm(enumerate(nifti_files)):

        t0 = time.time()
        image_data, _ = load(nifti_files[idx])

        image_data = torch.tensor(image_data, dtype=torch.float).to(device).unsqueeze(0).unsqueeze(0)
        vmax = image_data.max()*0.001
        image_data = image_data/vmax-0.5
        # image_data = F.pad(image_data, (49,49, 49, 49, 0, 0)) -0.5
        h, out = net(image_data)
        out = relation_infer(h.detach(), out, net, config.MODEL.DECODER.OBJ_TOKEN, config.MODEL.DECODER.RLN_TOKEN)
        
        if args.eval:
            pred_nodes = torch.tensor(out['pred_nodes'][0], dtype=torch.float)
            pred_edges = torch.tensor(out['pred_rels'][0], dtype=torch.int64)
            vtk_data = pyvista.read(vtk_files[idx])
            nodes = torch.tensor(np.float32(np.asarray(vtk_data.points)), dtype=torch.float)
            edges = torch.tensor(np.asarray(vtk_data.lines.reshape(-1, 3)), dtype=torch.int64)[:,1:]
            boxes = [torch.cat([nodes, 0.2*torch.ones(nodes.shape, device=nodes.device)], dim=-1).numpy()]
            boxes_class = [np.zeros(boxes[0].shape[0])]
            edge_boxes = torch.stack([nodes[edges[:,0]], nodes[edges[:,1]]], dim=2)
            edge_boxes = torch.cat([torch.min(edge_boxes, dim=2)[0]-0.1, torch.max(edge_boxes, dim=2)[0]+0.1], dim=-1).numpy()
            edge_boxes = [edge_boxes[:,[0,1,3,4,2,5]]]
            if pred_edges.shape[0]>0:
                pred_edge_boxes = torch.stack([pred_nodes[pred_edges[:,0]], pred_nodes[pred_edges[:,1]]], dim=2)
                pred_edge_boxes = torch.cat([torch.min(pred_edge_boxes, dim=2)[0]-0.1, torch.max(pred_edge_boxes, dim=2)[0]+0.1], dim=-1).numpy()
                pred_edge_boxes = [pred_edge_boxes[:,[0,1,3,4,2,5]]]
                edge_boxes_class = [np.zeros(edges.shape[0])]
            else:
                pred_edge_boxes = []
                edge_boxes_class = []
            # boxes_scores = [np.ones(boxes[0].shape[0])]
            
            # mean AP
            node_ap_result.extend(box_evaluator(out["pred_boxes"], out["pred_boxes_class"], out["pred_boxes_score"], boxes, boxes_class))

            # mean AP
            edge_ap_result.extend(box_evaluator(pred_edge_boxes, out["pred_rels_class"], out["pred_rels_score"], edge_boxes, edge_boxes_class, convert_box=False))
            
            # mean SMD            
            A = torch.zeros((nodes.shape[0], nodes.shape[0]))
            pred_A = torch.zeros((pred_nodes.shape[0], pred_nodes.shape[0]))

            A[edges[:,0],edges[:,1]] = 1
            A[edges[:,1],edges[:,0]] = 1
            A = torch.tril(A)

            if nodes.shape[0]>1 and pred_nodes.shape[0]>1 and pred_edges.size != 0:
                # print(pred_edges)
                pred_A[pred_edges[:,0], pred_edges[:,1]] = 1.0
                pred_A[pred_edges[:,1], pred_edges[:,0]] = 1.0
                pred_A = torch.tril(pred_A)

                mean_smd.append(compute_meanSMD(A, nodes, pred_A, pred_nodes, sinkhorn_distance, n_points=100).numpy())
 
    # Accumulate SMD score
    print("Mean SMD:", torch.tensor(mean_smd).mean())
    
    # accumulate AP score
    node_metric_scores = {}
    edge_metric_scores = {}
    for metric_idx, metric in enumerate(metrics):
        _filter = partial(iou_filter, iou_idx=iou_mapping[metric_idx])
        iou_filtered_results = list(map(_filter, node_ap_result))
        score, curve = metric(iou_filtered_results)
        if score is not None:
            node_metric_scores.update(score)

        iou_filtered_results = list(map(_filter, edge_ap_result))
        score, curve = metric(iou_filtered_results)
        if score is not None:
            edge_metric_scores.update(score)
            
    for key in node_metric_scores.keys():
        print(key, node_metric_scores[key])

    for key in edge_metric_scores.keys():
        print(key, edge_metric_scores[key])
    

if __name__ == '__main__':
    args = parser.parse_args()
    main(args)