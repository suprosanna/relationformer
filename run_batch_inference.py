import os
import itertools
import yaml
import logging
import ignite
import torch
import torch.nn as nn
from monai.data import DataLoader
from dataset_scene import build_scene_graph_data
from trainer import build_trainer
from models import build_model
from utils import image_graph_collate_scene_graph
from models.matcher_scene import build_matcher
from losses import SetCriterion

def relation_infer_sgrecall(h, out, model):
    # all token except the last one is object token
    object_token = h[..., :-1, :]

    # last token is relation token
    relation_token = h[..., -1:, :]

    # valid tokens
    valid_token = torch.argmax(out['pred_logits'], -1).detach()
    # valid_token = torch.sigmoid(nodes_prob[...,3])>0.5

    pred_nodes = []
    pred_edges = []
    for batch_id in range(h.shape[0]):

        # ID of the valid tokens
        node_id = torch.nonzero(valid_token[batch_id]).squeeze(1)

        # coordinates of the valid tokens
        pred_nodes.append(out['pred_nodes'][batch_id, node_id, :3].detach())
        if node_id.dim() != 0 and node_id.nelement() != 0 and node_id.shape[0] > 1:

            # all possible node pairs in all token ordering
            node_pairs = [list(i) for i in list(itertools.combinations(list(node_id), 2))]
            node_pairs = list(map(list, zip(*node_pairs)))
            # node pairs in valid token order
            node_pairs_valid = torch.tensor(
                [list(i) for i in list(itertools.combinations(list(range(len(node_id))), 2))])

            # concatenate valid object pairs relation feature
            relation_feature = torch.cat((object_token[batch_id, node_pairs[0], :],
                                          object_token[batch_id, node_pairs[1], :],
                                          relation_token[batch_id, ...].repeat(len(node_pairs_valid), 1)), 1)
            # relation_pred = torch.nonzero(torch.softmax(model(relation_feature).detach(), dim=-1)).squeeze(1).cpu().numpy()
            estimation = torch.softmax(model(relation_feature).detach(), dim=-1)
            rel_score = torch.max(estimation, dim=-1)
            rel_inds = torch.argmax(estimation, dim=-1)  # todo: relation indices should have 2 nodes, and relation score just 1 value? then argmax here is wrong
            # pred_edges.append(node_pairs_valid[relation_pred])
        else:
            pred_edges.append(torch.empty(0, 2))

    return pred_nodes, pred_edges

def main_eval_sgrecall(args):
    # Load the config files
    with open(args.config) as f:
        print('\n*** Config file')
        print(args.config)
        config = yaml.load(f, Loader=yaml.FullLoader)
        # print(config['log']['message'])
    config = dict2obj(config)
    os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(map(str, args.cuda_visible_device))

    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True
    torch.multiprocessing.set_sharing_strategy('file_system')
    device = torch.device("cuda") if args.device == 'cuda' else torch.device("cpu")

    net = build_model(config).to(device)

    matcher = build_matcher()
    loss = SetCriterion(config, matcher, net.relation_embed)

    _, val_ds = build_scene_graph_data(
        config, mode='split'
    )

    val_loader = DataLoader(val_ds,
                            batch_size=config.DATA.BATCH_SIZE,
                            shuffle=False,
                            num_workers=config.DATA.NUM_WORKERS,
                            collate_fn=image_graph_collate_scene_graph,
                            pin_memory=True)
    checkpoint = torch.load(args.resume, map_location='cpu')
    net.load_state_dict(checkpoint['net'])
    # evaluator = build_evaluator(
    #     val_loader,
    #     net,
    #     optimizer=None,
    #     scheduler=None,
    #     writer=None,
    #     config=config,
    #     device=device,
    # )
    for i_batch, data_blob in enumerate(val_loader):
        images, target = data_blob

        with torch.no_grad():
            images = torch.stack(images, 0).cuda()
            # todo: maybe list is not a proper format here for the metric SGRecall, need to be investigated later
            boxes = [tgt['boxes'].tolist() for tgt in target]
            edges = [tgt['edges'].tolist() for tgt in target]  # todo: add .tolist() later. Here no .tolist() because it need to be post-processed later for ['pred_rel_inds'] ['rel_scores']
            labels = [tgt['labels'].tolist() for tgt in target]
            h, out = net(images)
            a = 1
            local_container = {}

            local_container['gt_rels'] = edges
            local_container['gt_classes'] = labels
            local_container['gt_boxes'] = boxes

            # local_container['pred_classes'] = torch.argmax(out['pred_logits'], dim=-1)
            local_container['pred_classes'] = [pred[np.where(pred != 0)] for pred in torch.argmax(out['pred_logits'], dim=-1).cpu().numpy()]
            local_container['pred_boxes'] = [pred[local_container['pred_classes'][i]] for i, pred in enumerate(out['pred_nodes'].cpu().numpy())]
            local_container['obj_scores'] = [pred[local_container['pred_classes'][i]] for i, pred in enumerate(torch.max(torch.softmax(out['pred_logits'], dim=-1), dim=-1)[0].cpu().numpy())]

            # local_container['rel_scores'] = relation_infer_sgrecall(h, out, net.relation_embed)
            # local_container['pred_rel_inds']

            local_container['pred_rel_inds'] = [i[:, :2].tolist() for i in local_container['gt_rels']] # dummy-coded for test
            local_container['rel_scores'] = [i[:, 2:].tolist() for i in local_container['gt_rels']] # dummy-coded for test
            local_container['gt_rels'] = [i.tolist() for i in local_container['gt_rels']] # dummy-coded for test

            # # took directly from calculate_recall of SGRecall
            # pred_rel_inds = local_container['pred_rel_inds']
            # rel_scores = local_container['rel_scores']
            # gt_rels = local_container['gt_rels']
            # gt_classes = local_container['gt_classes']
            # gt_boxes = local_container['gt_boxes']
            # pred_classes = local_container['pred_classes']
            # pred_boxes = local_container['pred_boxes']
            # obj_scores = local_container['obj_scores']
            #
            # # iou_thres = global_container['iou_thres']
            #
            # pred_rels = np.column_stack((pred_rel_inds, 1 + rel_scores[:, 1:].argmax(1)))
            # pred_scores = rel_scores[:, 1:].max(1)
            #
            # gt_triplets, gt_triplet_boxes, _ = _triplet(gt_rels, gt_classes, gt_boxes)
            # local_container['gt_triplets'] = gt_triplets
            # local_container['gt_triplet_boxes'] = gt_triplet_boxes
            #
            # pred_triplets, pred_triplet_boxes, pred_triplet_scores = _triplet(
            #     pred_rels, pred_classes, pred_boxes, pred_scores, obj_scores)



if __name__ == '__main__':
    args = parser.parse_args()
    main_eval_sgrecall(args)