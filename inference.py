import torch
import torch.nn.functional as F
from torchvision.ops import batched_nms
import itertools


def relation_infer(h, out, model, obj_token, rln_token, nms=False, map_=False):
    # all token except the last one is object token
    object_token = h[...,:obj_token,:]
    
    # last token is relation token
    if rln_token > 0:
        relation_token = h[..., obj_token:obj_token+rln_token, :]

    # valid tokens
    valid_token = torch.argmax(out['pred_logits'], -1).detach()

    # apply nms on valid tokens
    if nms:
        valid_token_nms = torch.zeros_like(valid_token)
        for idx, (token, logits, nodes) in enumerate(zip(valid_token, out['pred_logits'], out['pred_nodes'])):
            valid_token_id = torch.nonzero(token).squeeze(1)
            
            valid_logits, valid_nodes = logits[valid_token_id], nodes[valid_token_id]
            valid_scores = F.softmax(valid_logits, dim=1)[:, 1]

            # 0 <= x1 < x2 and 0 <= y1 < y2 has to be fulfilled
            valid_nodes[:, 2:] = valid_nodes[:, :2] + 0.5
            
            ids2keep = batched_nms(
                boxes=valid_nodes * 1000, scores=valid_scores, idxs=torch.ones_like(valid_scores, dtype=torch.long), iou_threshold=0.90
            )
            valid_token_id_nms = valid_token_id[ids2keep].sort()[0]
            # print(valid_nodes.shape[0] - ids2keep.shape[0])

            valid_token_nms[idx][valid_token_id_nms] = 1
        valid_token = valid_token_nms

    pred_nodes = []
    pred_edges = []
    if map_:
        pred_nodes_boxes = []
        pred_nodes_boxes_score = []
        pred_nodes_boxes_class = []

        pred_edges_boxes_score = []
        pred_edges_boxes_class = []

    for batch_id in range(h.shape[0]):
        
        # ID of the valid tokens
        node_id = torch.nonzero(valid_token[batch_id]).squeeze(1)
        
        # coordinates of the valid tokens
        pred_nodes.append(out['pred_nodes'][batch_id, node_id, :2].detach())

        if map_:
            pred_nodes_boxes.append(out['pred_nodes'][batch_id, node_id, :].detach().cpu().numpy())
            pred_nodes_boxes_score.append(out['pred_logits'].softmax(-1)[batch_id, node_id, 1].detach().cpu().numpy()) # TODO: generalize over multi-class
            pred_nodes_boxes_class.append(valid_token[batch_id, node_id].cpu().numpy())

        if node_id.dim() !=0 and node_id.nelement() != 0 and node_id.shape[0]>1:
            
            # all possible node pairs in all token ordering
            node_pairs = [list(i) for i in list(itertools.combinations(list(node_id),2))]
            node_pairs = list(map(list, zip(*node_pairs)))
            
            # node pairs in valid token order
            node_pairs_valid = torch.tensor([list(i) for i in list(itertools.combinations(list(range(len(node_id))),2))])

            # concatenate valid object pairs relation feature
            if rln_token>0:
                relation_feature1  = torch.cat((object_token[batch_id,node_pairs[0],:], object_token[batch_id,node_pairs[1],:], relation_token[batch_id,...].repeat(len(node_pairs_valid),1)), 1)
                relation_feature2  = torch.cat((object_token[batch_id,node_pairs[1],:], object_token[batch_id,node_pairs[0],:], relation_token[batch_id,...].repeat(len(node_pairs_valid),1)), 1)
            else:
                relation_feature1  = torch.cat((object_token[batch_id,node_pairs[0],:], object_token[batch_id,node_pairs[1],:]), 1)
                relation_feature2  = torch.cat((object_token[batch_id,node_pairs[1],:], object_token[batch_id,node_pairs[0],:]), 1)

            relation_pred1 = model.relation_embed(relation_feature1).detach()
            relation_pred2 = model.relation_embed(relation_feature2).detach()
            relation_pred = (relation_pred1+relation_pred2)/2.0

            pred_rel = torch.nonzero(torch.argmax(relation_pred, -1)).squeeze(1).cpu().numpy()
            pred_edges.append(node_pairs_valid[pred_rel].cpu().numpy())

            if map_:
                pred_edges_boxes_score.append(relation_pred.softmax(-1)[pred_rel, 1].cpu().numpy())
                pred_edges_boxes_class.append(torch.argmax(relation_pred, -1)[pred_rel].cpu().numpy())
        else:
            pred_edges.append(torch.empty(0,2))

            if map_:
                pred_edges_boxes_score.append(torch.empty(0,1))
                pred_edges_boxes_class.append(torch.empty(0,1))

    if map_:
        return pred_nodes, pred_edges, pred_nodes_boxes, pred_nodes_boxes_score, pred_nodes_boxes_class, pred_edges_boxes_score, pred_edges_boxes_class
    else:
        return pred_nodes, pred_edges