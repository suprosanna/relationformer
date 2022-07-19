import torch
import torch.nn.functional as F
from torch import nn
import itertools
import pdb
import box_ops_2D
import numpy as np


def sigmoid_focal_loss(inputs, targets, num_boxes, alpha: float = 0.25, gamma: float = 2):
    """
    Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
        alpha: (optional) Weighting factor in range (0,1) to balance
                positive vs negative examples. Default = -1 (no weighting).
        gamma: Exponent of the modulating factor (1 - p_t) to
               balance easy vs hard examples.
    Returns:
        Loss tensor
    """
    prob = inputs.sigmoid()
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    p_t = prob * targets + (1 - prob) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss

    return loss.sum()/num_boxes

@torch.no_grad()
def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    if target.numel() == 0:
        return [torch.zeros([], device=output.device)]
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

class SetCriterion(nn.Module):
    """ This class computes the loss for Graphformer.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth boxes and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    """

    def __init__(self, config, matcher, net):
        """ Create the criterion.
        Parameters:
            num_classes: number of object categories, omitting the special no-object category
            matcher: module able to compute a matching between targets and proposals
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            eos_coef: relative classification weight applied to the no-object category
            losses: list of all the losses to be applied. See get_loss for list of available losses.
        """
        super().__init__()
        self.matcher = matcher
        self.net = net
        self.rln_token = config.MODEL.DECODER.RLN_TOKEN
        self.obj_token = config.MODEL.DECODER.OBJ_TOKEN
        self.losses = config.TRAIN.LOSSES
        self.weight_dict = {'boxes':config.TRAIN.W_BBOX,
                            'class':config.TRAIN.W_CLASS,
                            'cards':config.TRAIN.W_CARD,
                            'nodes':config.TRAIN.W_NODE,
                            'edges':config.TRAIN.W_EDGE,
                            }
        
    def loss_class(self, outputs, indices):
        """Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
           targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]
           The target boxes are expected in format (center_x, center_y, w, h), normalized by the image size.
        """
        weight = torch.tensor([0.2, 0.8]).to(outputs.get_device())
        
        idx = self._get_src_permutation_idx(indices)

        # targets = torch.zeros(outputs.shape[:-1], dtype=outputs.dtype).to(outputs.get_device())
        # targets[idx] = 1.0
        
        # targets = targets.unsqueeze(-1)
        
        # num_nodes = targets.sum()
        # # loss = F.cross_entropy(outputs.permute(0,2,1), targets, weight=weight, reduction='mean')
        # loss = sigmoid_focal_loss(outputs, targets, num_nodes)
        
        targets = torch.zeros(outputs[...,0].shape, dtype=torch.long).to(outputs.get_device())
        targets[idx] = 1.0
        loss = F.cross_entropy(outputs.permute(0,2,1), targets, weight=weight, reduction='mean')
        
        # cls_acc = 100 - accuracy(outputs, targets_one_hot)[0]
        return loss
    
    def loss_cardinality(self, outputs, indices):
        """ Compute the cardinality error, ie the absolute error in the number of predicted non-empty boxes
        This is not really a loss, it is intended for logging purposes only. It doesn't propagate gradients
        """
        idx = self._get_src_permutation_idx(indices)
        targets = torch.zeros(outputs[...,0].shape, dtype=torch.long).to(outputs.get_device())
        targets[idx] = 1.0
        
        tgt_lengths = torch.as_tensor([t.sum() for t in targets], device=outputs.device)
        # Count the number of predictions that are NOT "no-object" (which is the last class)
        card_pred = (outputs.argmax(-1) == outputs.shape[-1] - 1).sum(1)
        # card_pred = (outputs.sigmoid()>0.5).squeeze(-1).sum(1)

        loss = F.l1_loss(card_pred.float(), tgt_lengths.float(), reduction='sum')/(outputs.shape[0]*outputs.shape[1])

        return loss

    def loss_nodes(self, outputs, targets, indices):
        """Classification loss (NLL)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        num_nodes = sum(len(t) for t in targets)
        
        idx = self._get_src_permutation_idx(indices)
        pred_nodes = outputs[idx]
        target_nodes = torch.cat([t[i] for t, (_, i) in zip(targets, indices)], dim=0)
        
        loss = F.l1_loss(pred_nodes, target_nodes, reduction='none') # TODO: check detr for loss function

        loss = loss.sum() / num_nodes

        return loss
    
    def loss_boxes(self, outputs, targets, indices):
        """Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
           targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]
           The target boxes are expected in format (center_x, center_y, w, h), normalized by the image size.
        """
        num_boxes = sum(len(t) for t in targets)
        idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs[idx]

        target_boxes = torch.cat([t[i] for t, (_, i) in zip(targets, indices)], dim=0)
        target_boxes = torch.cat([target_boxes, 0.15*torch.ones(target_boxes.shape, device=target_boxes.device)], dim=-1)

        loss = 1 - torch.diag(box_ops_2D.generalized_box_iou(
            box_ops_2D.box_cxcywh_to_xyxy(src_boxes),
            box_ops_2D.box_cxcywh_to_xyxy(target_boxes)))
        loss = loss.sum() / num_boxes
        return loss

    def loss_edges(self, h, target_nodes, target_edges, indices, num_edges=80):
        """Compute the losses related to the masks: the focal loss and the dice loss.
           targets dicts must contain the key "masks" containing a tensor of dim [nb_target_boxes, h, w]
        """
        try:
            # all token except the last one is object token
            object_token = h[...,:self.obj_token,:]
            
            # last token is relation token
            if self.rln_token > 0:
                relation_token = h[..., self.obj_token:self.rln_token+self.obj_token, :]

            # map the ground truth edge indices by the matcher ordering
            target_edges = [[t for t in tgt if t[0].cpu() in i and t[1].cpu() in i] for tgt, (_, i) in zip(target_edges, indices)]
            target_edges = [torch.stack(t, 0) if len(t)>0 else torch.zeros((0,2), dtype=torch.long).to(h.device) for t in target_edges]

            new_target_edges = []
            for t, (_, i) in zip(target_edges, indices):
                tx = t.clone().detach()
                for idx, k in enumerate(i):
                    t[tx==k]=idx
                new_target_edges.append(t)

            # all_edges = []
            edge_labels = []
            relation_feature = []
            
            # loop through each of batch to collect the edge and node
            for batch_id, (pos_edge, n) in enumerate(zip(new_target_edges, target_nodes)):
                
                # map the predicted object token by the matcher ordering
                rearranged_object_token = object_token[batch_id, indices[batch_id][0],:]
                
                # find the -ve edges for training
                full_adj = torch.ones((n.shape[0],n.shape[0]))-torch.diag(torch.ones(n.shape[0]))
                full_adj[pos_edge[:,0],pos_edge[:,1]]=0
                full_adj[pos_edge[:,1],pos_edge[:,0]]=0
                neg_edges = torch.nonzero(torch.triu(full_adj))

                # shuffle edges for undirected edge
                shuffle = np.random.randn((pos_edge.shape[0]))>0
                to_shuffle = pos_edge[shuffle,:]
                pos_edge[shuffle,:] = to_shuffle[:,[1, 0]]

                # restrict unbalance in the +ve/-ve edge
                if pos_edge.shape[0]>40:
                    # print('Reshaping')
                    pos_edge = pos_edge[:40,:]
                
                # random sample -ve edge
                idx_ = torch.randperm(neg_edges.shape[0])
                neg_edges = neg_edges[idx_, :].to(pos_edge.device)

                # shuffle edges for undirected edge
                shuffle = np.random.randn((neg_edges.shape[0]))>0
                to_shuffle = neg_edges[shuffle,:]
                neg_edges[shuffle,:] = to_shuffle[:,[1, 0]]
                
                # check whether the number of -ve edges are within limit 
                if num_edges-pos_edge.shape[0]<neg_edges.shape[0]:
                    take_neg = num_edges-pos_edge.shape[0]
                    total_edge = num_edges
                else:
                    take_neg = neg_edges.shape[0]
                    total_edge = pos_edge.shape[0]+neg_edges.shape[0]
                all_edges_ = torch.cat((pos_edge, neg_edges[:take_neg]), 0)
                # all_edges.append(all_edges_)
                edge_labels.append(torch.cat((torch.ones(pos_edge.shape[0], dtype=torch.long), torch.zeros(take_neg, dtype=torch.long)), 0))

                # concatenate object token pairs with relation token
                if self.rln_token > 0:
                    relation_feature.append(torch.cat((rearranged_object_token[all_edges_[:,0],:],rearranged_object_token[all_edges_[:,1],:],relation_token[batch_id,...].repeat(total_edge,1)), 1))
                else:
                    relation_feature.append(torch.cat((rearranged_object_token[all_edges_[:,0],:],rearranged_object_token[all_edges_[:,1],:]), 1))

            # [print(e,l) for e,l in zip(all_edges, edge_labels)]

            # torch.tensor(list(itertools.combinations(range(n.shape[0]), 2))).to(e.get_device())
            relation_feature = torch.cat(relation_feature, 0)
            edge_labels = torch.cat(edge_labels, 0).to(h.get_device())

            relation_pred = self.net.relation_embed(relation_feature)

            # valid_edges = torch.argmax(relation_pred, -1)
            # print('valid_edge number', valid_edges.sum())

            loss = F.cross_entropy(relation_pred, edge_labels, reduction='mean')
        except Exception as e:
            print(e)
            pdb.set_trace()

        return loss

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def forward(self, h, out, target):
        """ This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """

        # Retrieve the matching between the outputs of the last layer and the targets
        indices = self.matcher(out, target)
        
        losses = {}
        losses['class'] = self.loss_class(out['pred_logits'], indices)
        losses['nodes'] = self.loss_nodes(out['pred_nodes'][...,:2], target['nodes'], indices)
        losses['boxes'] = self.loss_boxes(out['pred_nodes'], target['nodes'], indices)
        losses['edges'] = self.loss_edges(h, target['nodes'], target['edges'], indices)
        losses['cards'] = self.loss_cardinality(out['pred_logits'], indices)
        
        losses['total'] = sum([losses[key]*self.weight_dict[key] for key in self.losses])

        return losses