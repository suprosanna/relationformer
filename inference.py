import torch
import itertools
import numpy as np
import pdb

def graph_infer(h, out, relation_embed, freq=None,emb=False, thresh=0.5):
    # all token except the last one is object token
    object_token, relation_token = h
    object_token = object_token.detach()
    relation_token = relation_token.detach()

    # valid tokens
    valid_token = torch.max(out['pred_logits'].softmax(-1).detach(),-1)

    pred_nodes = []
    pred_boxes = []
    pred_boxes_score = []
    pred_boxes_class = []
    pred_rels = []
    pred_rel_score = []
    pred_rel_class = []
    all_node_pairs = []
    all_relation = []
    valid_nodes = []
    for batch_id in range(object_token.shape[0]):
        
        # ID of the valid tokens
        node_id = torch.nonzero(valid_token[1][batch_id]).squeeze(1)
        valid_nodes.append(node_id.cpu().numpy())
        pred_classes = valid_token[1][batch_id, node_id]
        pred_cls_score = valid_token[0][batch_id, node_id]

        # coordinates of the valid tokens
        pred_nodes.append(out['pred_boxes'][batch_id, node_id, :3].detach().cpu().numpy())
        pred_boxes.append(out['pred_boxes'][batch_id, node_id, :].detach().cpu().numpy())
        pred_boxes_score.append(pred_cls_score.detach().cpu().numpy())  # TODO: generalize over multi-class
        pred_boxes_class.append(pred_classes.cpu().numpy() - 1.0)  # TODO: class starts from 0 not 1 for coco matcher

        if node_id.dim() !=0 and node_id.nelement() != 0 and node_id.shape[0]>1:

            # all possible node pairs in all token ordering
            node_pairs = torch.cat((torch.combinations(node_id),torch.combinations(node_id)[:,[1,0]]),0)
            id_rel = torch.tensor(list(range(len(node_id))))
            node_pairs_rel = torch.cat((torch.combinations(id_rel),torch.combinations(id_rel)[:,[1,0]]),0)

            joint_emb = object_token
            rln_feat  = torch.cat((joint_emb[batch_id,node_pairs[:,0],:], joint_emb[batch_id,node_pairs[:,1],:], relation_token[batch_id,...].repeat(len(node_pairs),1)), 1)

            relation_pred = relation_embed(rln_feat).detach()

            if freq is not None:
                relation_pred += freq.index_with_labels(
                    torch.stack((valid_token[1][batch_id,node_pairs[:,0]], valid_token[1][batch_id,node_pairs[:,1]]), 1))

            all_node_pairs.append(node_pairs_rel.cpu().numpy())
            all_relation.append(relation_pred.softmax(-1).detach().cpu().numpy())
            rel_id = torch.nonzero(torch.argmax(relation_pred[:,1:], -1)+1).squeeze(1)
            if rel_id.dim() !=0 and rel_id.nelement() != 0 and rel_id.shape[0]>1:
                rel_id = rel_id.cpu().numpy()
                pred_rels.append(node_pairs_rel[rel_id].cpu().numpy())
                pred_rel_class.append(torch.argmax(relation_pred[:,1:], -1, keepdim=True)[rel_id].cpu().numpy()+1)
            else:
                pred_rels.append(torch.empty(0,2))
                pred_rel_class.append(torch.empty(0,1))
        else:
            all_node_pairs.append(None)
            all_relation.append(None)
            pred_rels.append(torch.empty(0,2))
            pred_rel_class.append(torch.empty(0,1))

    out = {}
    out['node_id']=valid_nodes
    out['pred_boxes'] = pred_boxes
    out['pred_boxes_score'] = pred_boxes_score
    out['pred_boxes_class'] = pred_boxes_class

    out['pred_rels'] = pred_rels
    out['pred_rels_class'] = pred_rel_class
    out['all_node_pairs'] = all_node_pairs
    out['all_relation'] = all_relation

    return out