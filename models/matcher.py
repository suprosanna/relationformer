# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Modules to compute the matching cost and solve the corresponding LSAP.
"""
import torch
from scipy.optimize import linear_sum_assignment
from torch import nn


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
        self.cost_nodes = config.MODEL.MATCHER.C_NODE
        self.cost_class = config.MODEL.MATCHER.C_CLASS

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
        out_nodes = outputs['pred_nodes'][...,:2].flatten(0, 1)  # [batch_size * num_queries, 2]

        # Also concat the target labels and boxes
        tgt_nodes = torch.cat([v for v in targets['nodes']])

        # Compute the L1 cost between nodes
        cost_nodes = torch.cdist(out_nodes, tgt_nodes, p=1)

        # Compute the cls cost
        tgt_ids = torch.cat([torch.tensor([1]*v.shape[0]).to(out_nodes.device) for v in targets['nodes']])
        cost_class = -outputs["pred_logits"].flatten(0, 1).softmax(-1)[..., tgt_ids]

        # Final cost matrix
        C = self.cost_nodes * cost_nodes + self.cost_class * cost_class
        C = C.view(bs, num_queries, -1).cpu()

        sizes = [len(v) for v in targets['nodes']]
        indices = [linear_sum_assignment(c[i]) for i, c in enumerate(C.split(sizes, -1))]
        return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]


def build_matcher(config):
    return HungarianMatcher(config)
