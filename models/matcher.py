# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Modules to compute the matching cost and solve the corresponding LSAP.
"""
import torch
from scipy.optimize import linear_sum_assignment
from torch import nn
#from lpmp_py import GraphMatchingModule
import pdb
import sys
import os

class HungarianMatcher(nn.Module):
    """This class computes an assignment between the targets and the predictions of the network

    For efficiency reasons, the targets don't include the no_object. Because of this, in general,
    there are more predictions than targets. In this case, we do a 1-to-1 matching of the best predictions,
    while the others are un-matched (and thus treated as non-objects).
    """

    def __init__(self, config):
        """Creates the matcher

        Params:
            cost_class: This is the relative weight of the classification error in the matching cost
            cost_bbox: This is the relative weight of the L1 error of the bounding box coordinates in the matching cost
            cost_giou: This is the relative weight of the giou loss of the bounding box in the matching cost
        """
        super().__init__()
        self.cost_class = config.MODEL.MATCHER.C_CLASS
        self.cost_nodes = config.MODEL.MATCHER.C_NODE

    @torch.no_grad()
    def forward(self, outputs, targets):
        """[summary]

        Args:
            outputs ([type]): [description]
            targets ([type]): [description]

        Returns:
            [type]: [description]
        """
        bs, num_queries = outputs['pred_nodes'].shape[:2]

        # We flatten to compute the cost matrices in a batch
        out_nodes = outputs['pred_nodes'][...,:3].flatten(0, 1)  # [batch_size * num_queries, 3]

        # Also concat the target labels and boxes
        tgt_nodes = torch.cat([v for v in targets['nodes']])

        # Compute the L1 cost between nodes
        cost_nodes = torch.cdist(out_nodes, tgt_nodes, p=1)
        
        # Compute the classification cost.
        tgt_ids = torch.cat([torch.tensor([1]*v.shape[0]).to(out_nodes.device) for v in targets['nodes']])
        cost_class = -outputs["pred_logits"].flatten(0, 1).softmax(-1)[..., tgt_ids]

        # Final cost matrix
        C = self.cost_nodes * cost_nodes + self.cost_class * cost_class
        C = C.view(bs, num_queries, -1).cpu()

        sizes = [len(v) for v in targets['nodes']]
        indices = [linear_sum_assignment(c[i]) for i, c in enumerate(C.split(sizes, -1))]
        return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]


class GraphMatcher(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.solver_params = {'timeout': config.MODEL.MATCHER.PARAMS.timeout,
                              'primalComputationInterval': config.MODEL.MATCHER.PARAMS.primalComputationInterval,
                              'maxIter': config.MODEL.MATCHER.PARAMS.maxIter
                              }
        self.lambda_val = config.MODEL.MATCHER.LAMDA
        self.cost_class = config.MODEL.MATCHER.C_CLASS
        self.cost_nodes = config.MODEL.MATCHER.C_NODE

    @torch.no_grad()
    def forward(self, outputs, targets):
        bs, num_queries = outputs['pred_nodes'].shape[:2]

        out_nodes = outputs['pred_nodes'][...,:3]  # [batch_size * num_queries, 3]

        # Also concat the target labels and boxes
        tgt_nodes = [v for v in targets['nodes']]

        # Compute the L1 cost between nodes
        cost_nodes = [torch.cdist(out_nodes[i], tgt_nodes[i], p=2) for i in range(bs)]
        
        # Compute the classification cost.
        tgt_ids = [torch.tensor([1]*v.shape[0]).to(out_nodes.device) for v in targets['nodes']]
        cost_class = [-outputs["pred_logits"][i].softmax(-1)[..., tgt_ids[i]] for i in range(bs)]
        
        ns_src = [v.shape[0] for v in out_nodes]
        ns_tgt = [len(v) for v in tgt_nodes]

        # make the edge list
        all_right_edges = [t.transpose(0,1) for t in targets['edges']]
        all_left_edges = [t.transpose(0,1) for t in outputs['pred_rels']]

        gm_solver = GraphMatchingModule(
                all_left_edges,
                all_right_edges,
                ns_src,
                ns_tgt,
                self.lambda_val,
                self.solver_params,
            )
        
        # Compute the L1 cost between nodes
        unary_costs = [self.cost_nodes * cost_nodes[i] + self.cost_class * cost_class[i] for i in range(bs)]

        tgt_ids = [torch.tensor([1]*v.shape[0]).to(out_nodes.device) for v in targets['edges']]
        quadratic_costs = [-outputs["pred_rels_score"][i].softmax(-1)[..., tgt_ids[i]] for i in range(bs)]
  
        matchings = gm_solver(unary_costs, quadratic_costs)
        
        indices = [tuple(torch.nonzero(m).transpose(0,1).cpu()) for m in matchings]

        return indices


def build_matcher(config):
    if config.MODEL.MATCHER.NAME=='Hungarian':
        return HungarianMatcher(config)
    elif config.MODEL.MATCHER.NAME=='Graph':
        return GraphMatcher(config)
    else:
        raise('Invalid matcher name')
