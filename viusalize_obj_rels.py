import os
os.environ['CUDA_VISIBLE_DEVICES'] = "1"
from argparse import ArgumentParser
import yaml
import sys
import json
from tqdm import tqdm
import logging
import torch
import torch.nn as nn
import pickle
import numpy as np
from tensorboardX import SummaryWriter
from monai.data import DataLoader, DistributedSampler
from dataset_scene import build_scene_graph_data
from util.sg_recall import BasicSceneGraphEvaluator
from collections import defaultdict
from models import build_model
from util.box_ops_np import box_iou_np as bbox_overlaps
from losses import SetCriterion
from datasets.sparse_targets import FrequencyBias
from inference import graph_infer
# %matplotlib widget
sys.path.append("..")
import json
from torch.utils.tensorboard import SummaryWriter
from scipy import ndimage
from scipy.sparse import csr_matrix
import box_ops_2D as box_ops
import matplotlib.pyplot as plt
from matplotlib import text, patheffects
import matplotlib.patches as patches
from utils import image_graph_collate_scene_graph
from functools import reduce
from PIL import Image, ImageDraw, ImageFont
os.environ['HDF5_USE_FILE_LOCKING']="FALSE"
#load dict for visual genome
dict = json.load(open('data/stanford_filtered/VG-SGG-dicts.json'))
class_dict = dict['idx_to_label']
pred_dict = dict['idx_to_predicate']

font = ImageFont.load_default() #ImageFont.truetype('/usr/share/fonts/truetype/freefont/FreeMonoBold.ttf', 32)
mean = np.array([0.485, 0.456, 0.406])
std = np.array([0.229, 0.224, 0.225])

def draw_box(draw, boxx, cls_ind, text_str):
    box = tuple([float(b) for b in boxx])
    if '-GT' in text_str:
        color = (255, 128, 0, 255)
    else:
        color = (0, 128, 0, 255)

    # color = tuple([int(x) for x in cmap(cls_ind)])

    # draw the fucking box
    draw.line([(box[0], box[1]), (box[2], box[1])], fill=color, width=8)
    draw.line([(box[2], box[1]), (box[2], box[3])], fill=color, width=8)
    draw.line([(box[2], box[3]), (box[0], box[3])], fill=color, width=8)
    draw.line([(box[0], box[3]), (box[0], box[1])], fill=color, width=8)

    # draw.rectangle(box, outline=color)
    w, h = draw.textsize(text_str, font=font)

    x1text = box[0]
    y1text = max(box[1] - h, 0)
    x2text = min(x1text + w, draw.im.size[0])
    y2text = y1text + h
    print("drawing {}x{} rectangle at {:.1f} {:.1f} {:.1f} {:.1f}".format(
        h, w, x1text, y1text, x2text, y2text))

    draw.rectangle((x1text, y1text, x2text, y2text), fill=color)
    draw.text((x1text, y1text), text_str, fill='black', font=font)
    return draw


class obj:
    def __init__(self, dict1):
        self.__dict__.update(dict1)


def dict2obj(dict1):
    return json.loads(json.dumps(dict1), object_hook=obj)

#load the config file aND CHECKPOINT PATH
config_file = "output/sample_run/debug_visulagenome_with_60_obj_75bg_auxloss-sftmx_10/config.yaml"
ckpt_path = 'output/sample_run/debug_visulagenome_with_60_obj_75bg_auxloss-sftmx_10/models/checkpoint_epoch=21.pt'
dir_name = os.path.dirname(config_file)

with open(config_file) as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

config = dict2obj(config)

device = "cuda"
train_ds, val_ds = build_scene_graph_data(config, mode='split',
                                              debug=False)

train_loader = DataLoader(train_ds,
                            batch_size=config.DATA.BATCH_SIZE,
                            num_workers=config.DATA.NUM_WORKERS,
                            collate_fn=image_graph_collate_scene_graph,
                            pin_memory=True,
                            sampler= None,)
val_loader = DataLoader(val_ds,
                            batch_size=config.DATA.BATCH_SIZE,
                            num_workers=config.DATA.NUM_WORKERS,
                            collate_fn=image_graph_collate_scene_graph,
                            pin_memory=True,
                            sampler= None,)
#load model
model = build_model(config)
model = model.to(device)
relation_embed = model.relation_embed.to(device)
if config.MODEL.DECODER.FREQ_BIAS: # use freq bias
    freq_baseline = FrequencyBias(config.DATA.FREQ_BIAS, train_ds)
    freq_baseline = freq_baseline.to(device) if config.MODEL.DECODER.FREQ_BIAS else None


#load the model
checkpoint = torch.load(ckpt_path, map_location='cpu')
missing_keys, unexpected_keys = model.load_state_dict(checkpoint['net'], strict=False)
unexpected_keys = [k for k in unexpected_keys if not (k.endswith('total_params') or k.endswith('total_ops'))]
if len(missing_keys) > 0:
    print('Missing Keys: {}'.format(missing_keys))
if len(unexpected_keys) > 0:
    print('Unexpected Keys: {}'.format(unexpected_keys))
print('MODEL loaded sucessfully')


def val_epoch():
    model.eval()
    evaluator = BasicSceneGraphEvaluator.all_modes()
    for b,(images, gt_datas) in enumerate(tqdm(val_loader)):
        val_batch(b,images, gt_datas, evaluator)

    evaluator['sgdet'].print_stats()


def val_batch(batch_num, images, gt_datas, evaluator, thrs=(20, 50, 100)):
    images = [image.to(device, non_blocking=False) for image in images]

    model.eval()
    h, out = model(images)  # todo output logit and edge are same value
    out = graph_infer(h.detach(), out, relation_embed, freq=freq_baseline, emb=False)

    pred_edges = [{'node_pair': pred_rels, 'edge_score': edge_score} for pred_rels, edge_score in
                  zip(out['all_node_pairs'], out['all_relation'])]
    pred_classes = [{'labels': pred_class + 1, 'scores': pred_score, 'boxes': torch.tensor(pred_box)} for
                    pred_class, pred_score, pred_box in
                    zip(out['pred_boxes_class'], out['pred_boxes_score'], out['pred_boxes'])]
    for i, (image, gt_data, pred_class, pred_edge) in enumerate(zip(images,gt_datas, pred_classes, pred_edges)):
        # prepare scene graph evaluation
        pred_to_gt, pred_5ples, rel_scores, _  = evaluator['sgdet'].evaluate_scene_graph_entry(gt_data, [pred_class, pred_edge])
        gt_boxes =  box_ops.box_cxcywh_to_xyxy(gt_data['boxes'])
        gt_classes = gt_data['labels'].data.numpy()
        gt_rels = gt_data['edges'].data.numpy()
        pred_boxes = box_ops.box_cxcywh_to_xyxy(pred_class['boxes']) #box_ops.box_cxcywh_to_xyxy(torch.tensor(out['pred_boxes'][0]).to(device, non_blocking=False))
        obj_score = pred_class['scores']
        pred_objs = pred_class['labels'].astype(int)
        image_id =  gt_data['image_id']

        # pred_objs = pred_objs[obj_score > 0.6].data.cpu().numpy()
        # pred_boxes = pred_boxes[obj_score > 0.6].data.cpu().numpy()
        #
        # pred_boxes = pred_boxes[pred_objs > 0, :]
        # pred_objs = pred_objs[pred_objs > 0]

        # SET RECALL THRESHOLD HERE
        pred_to_gt = pred_to_gt[:20]
        pred_5ples = pred_5ples[:20].astype(int)

        # Get a list of objects that match, and GT objects that dont
        objs_match = (bbox_overlaps(pred_boxes, gt_boxes) >= 0.5) & (
                pred_objs[:, None] == gt_classes[None]
        )
        objs_matched = objs_match.any(1)

        has_seen = defaultdict(int)
        has_seen_gt = defaultdict(int)
        pred_ind2name = {}
        gt_ind2name = {}
        edges = {}
        missededges = {}
        badedges = {}


        def query_pred(pred_ind):
            if pred_ind not in pred_ind2name:
                has_seen[pred_objs[pred_ind]] += 1
                pred_ind2name[pred_ind] = '{}-{}'.format(class_dict[str(int(pred_objs[pred_ind]))],
                                                         has_seen[pred_objs[pred_ind]])
            return pred_ind2name[pred_ind]

        def query_gt(gt_ind):
            gt_cls = gt_classes[gt_ind]
            if gt_ind not in gt_ind2name:
                has_seen_gt[gt_cls] += 1
                gt_ind2name[gt_ind] = '{}-GT{}'.format(class_dict[str(gt_cls)], has_seen_gt[str(gt_cls)])
            return gt_ind2name[gt_ind]

        matching_pred5ples = pred_5ples[np.array([len(x) > 0 for x in pred_to_gt])]
        for fiveple in matching_pred5ples:
            head_name = query_pred(fiveple[0])
            tail_name = query_pred(fiveple[1])

            edges[(head_name, tail_name)] = pred_dict[str(fiveple[4])]

        gt_5ples = np.column_stack((gt_rels[:, :2],
                                    gt_classes[gt_rels[:, 0]],
                                    gt_classes[gt_rels[:, 1]],
                                    gt_rels[:, 2],
                                    ))
        has_match = reduce(np.union1d, pred_to_gt)
        for gt in gt_5ples[np.setdiff1d(np.arange(gt_5ples.shape[0]), has_match)]:
            # Head and tail
            namez = []
            for i in range(2):
                matching_obj = np.where(objs_match[:, gt[i]])[0]
                if matching_obj.size > 0:
                    name = query_pred(matching_obj[0])
                else:
                    name = query_gt(gt[i])
                namez.append(name)
            missededges[tuple(namez)] = pred_dict[str(gt[4])]

        for fiveple in pred_5ples[np.setdiff1d(np.arange(pred_5ples.shape[0]), matching_pred5ples)]:

            if fiveple[0] in pred_ind2name:
                if fiveple[1] in pred_ind2name:
                    badedges[(pred_ind2name[fiveple[0]], pred_ind2name[fiveple[1]])] = pred_dict[str(int(fiveple[4]))]
        #cover image and boxes
        image = image.permute(1, 2, 0).data.cpu().numpy()
        image = np.clip(image * std + mean, 0, 1)
        theimg = Image.open("data/VG_100K/"+str(image_id)+".jpg").resize((image.shape[1],image.shape[0]), Image.ANTIALIAS)#  Image.fromarray(255*image,'RGB')#
        theimg2 = theimg.copy()
        draw1 = ImageDraw.Draw(theimg)
        draw2 = ImageDraw.Draw(theimg2)
        W = theimg.width
        H = theimg.height
        pred_boxes = pred_boxes * np.repeat(np.expand_dims([W, H, W, H], axis=0), repeats=len(pred_boxes), axis=0)
        gt_boxes = gt_boxes * np.repeat(np.expand_dims([W, H, W, H], axis=0), repeats=len(gt_boxes), axis=0)

        # Fix the names
        for gt_ind in gt_ind2name.keys():
            draw1 = draw_box(draw1, gt_boxes[gt_ind],
                             cls_ind=gt_classes[gt_ind],
                             text_str=gt_ind2name[gt_ind])
        for pred_ind in pred_ind2name.keys():
            draw2 = draw_box(draw2, pred_boxes[pred_ind],
                             cls_ind=pred_objs[pred_ind],
                             text_str=pred_ind2name[pred_ind])
        for gt_ind in gt_ind2name.keys():
            draw2 = draw_box(draw2, gt_boxes[gt_ind],
                             cls_ind=gt_classes[gt_ind],
                             text_str=gt_ind2name[gt_ind])

        recall = int(100 * len(reduce(np.union1d, pred_to_gt)) / gt_rels.shape[0])

        id = '{}-{}'.format(image_id, recall)
        pathname = os.path.join(dir_name,'qualitative', id)
        if not os.path.exists(pathname):
            os.makedirs(pathname)
        theimg.save(os.path.join(pathname, 'img.jpg'), quality=100, subsampling=0)
        theimg2.save(os.path.join(pathname, 'imgbox.jpg'), quality=100, subsampling=0)

        with open(os.path.join(pathname, 'shit.txt'), 'w') as f:
            f.write('good:\n')
            for (o1, o2), p in edges.items():
                f.write('{} - {} - {}\n'.format(o1, p, o2))
            f.write('fn:\n')
            for (o1, o2), p in missededges.items():
                f.write('{} - {} - {}\n'.format(o1, p, o2))
            f.write('shit:\n')
            for (o1, o2), p in badedges.items():
                f.write('{} - {} - {}\n'.format(o1, p, o2))


mAp = val_epoch()