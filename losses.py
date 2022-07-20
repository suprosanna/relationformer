import torch
import torch.nn.functional as F
from torch import nn
import itertools
import pdb
import box_ops_2D
from util.misc import is_dist_avail_and_initialized,get_world_size
import copy

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

    def __init__(self, config, matcher, relation_embed, **kwargs):
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
        self.relation_embed = relation_embed
        self.freq = config.MODEL.DECODER.FREQ_BIAS
        if config.MODEL.DECODER.FREQ_BIAS:
            self.freq_baseline = kwargs['freq_baseline']
            self.use_target = kwargs['use_target']
        self.focal_alpha = None if kwargs['focal_alpha']=='' else kwargs['focal_alpha']
        self.num_classes = config.MODEL.NUM_OBJ_CLS if self.focal_alpha is not None else config.MODEL.NUM_OBJ_CLS+1
        self.losses = config.TRAIN.LOSSES

        self.add_emd_rel = config.MODEL.DECODER.ADD_EMB_REL
        self.weight_dict = {'boxes':config.TRAIN.W_BBOX,
                            'class':config.TRAIN.W_CLASS,
                            'cards':config.TRAIN.W_CARD,
                            'nodes':config.TRAIN.W_NODE,
                            'edges':config.TRAIN.W_EDGE,
                            }
        # TODO this is a hack
        if config.MODEL.DECODER.AUX_LOSS:
            aux_weight_dict = {}
            for i in range(config.MODEL.DECODER.DEC_LAYERS - 1):
                aux_weight_dict.update({k + f'_{i}': v for k, v in self.weight_dict.items()})
            aux_weight_dict.update({k + f'_enc': v for k, v in self.weight_dict.items()})
            self.weight_dict.update(aux_weight_dict)
        self.fg_edge = config.DATA.FG_EDGE_PER_IMG
        self.bg_edge = config.DATA.BG_EDGE_PER_IMG

    def loss_class(self, outputs, targets, indices, num_boxes=None):
        """Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
           targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]
           The target boxes are expected in format (center_x, center_y, w, h), normalized by the image size.
        """
        weight = torch.ones(self.num_classes).to(outputs.get_device()) # TODO; fix the class weight
        weight[0] = 0.1
        
        idx = self._get_src_permutation_idx(indices)

        target_classes_o = torch.cat([t[J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.zeros(outputs[...,0].shape, dtype=torch.long).to(outputs.get_device())
        target_classes[idx] = target_classes_o
        if self.focal_alpha is not None:
            target_classes_onehot = torch.zeros([outputs.shape[0], outputs.shape[1], outputs.shape[2]],
                                                dtype=outputs.dtype, layout=outputs.layout,
                                                device=outputs.device)
            target_classes_onehot.scatter_(2, target_classes.unsqueeze(-1), 1)

            # target_classes_onehot = target_classes_onehot[:, :, :-1] #todo here only foreground classes are present,urgent check
            loss = sigmoid_focal_loss(outputs, target_classes_onehot, num_boxes, alpha=self.focal_alpha, gamma=2) * \
                   outputs.shape[1]
        else:
            loss = F.cross_entropy(outputs.permute(0, 2, 1), target_classes, weight=weight, reduction='mean')
        
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

    def l1_loss(self, outputs, targets, indices, num_boxes):
        """Classification loss (NLL)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        idx = self._get_src_permutation_idx(indices)
        pred_boxes = outputs[idx]
        target_nodes = torch.cat([t[i] for t, (_, i) in zip(targets, indices)], dim=0)
        
        loss = F.l1_loss(pred_boxes, target_nodes, reduction='none') # TODO: check detr for loss function

        loss = loss.sum() / num_boxes

        return loss
    
    def giou_loss(self, outputs, targets, indices, num_boxes):
        """Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
           targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]
           The target boxes are expected in format (center_x, center_y, w, h), normalized by the image size.
        """
        idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs[idx]

        target_boxes = torch.cat([t[i] for t, (_, i) in zip(targets, indices)], dim=0)

        loss = 1 - torch.diag(box_ops_2D.generalized_box_iou(
            box_ops_2D.box_cxcywh_to_xyxy(src_boxes),
            box_ops_2D.box_cxcywh_to_xyxy(target_boxes)))
        loss = loss.sum() / num_boxes
        return loss

    def loss_edges(self, object_token, relation_token, target_boxes, pred_classes, tgt_labels, target_edges, indices, cls_dist=None, pred_boxes=None):
        """Compute the losses related to the masks: the focal loss and the dice loss.
           targets dicts must contain the key "masks" containing a tensor of dim [nb_target_boxes, h, w]
        """

        # # last token is relation token
        # relation_token = h[...,-1:,:]
        
        rel_labels = [t[:,2] for t in target_edges]
        target_edges = [t[:,:2] for t in target_edges]
        # map the ground truth edge indices by the matcher ordering
        target_edges = [[t for t in tgt if t[0].cpu() in i and t[1].cpu() in i] for tgt, (_, i) in zip(target_edges, indices)]
        target_edges = [torch.stack(t, 0) if len(t)>0 else torch.zeros((0,2), dtype=torch.long).to(object_token.device) for t in target_edges]

        filtered_edges = []
        for t, (_, i) in zip(target_edges, indices):
            if t.shape[0]>0:
                tx = t.detach().clone()
                for idx, k in enumerate(i):
                    t[tx==k]=idx
            filtered_edges.append(t)
        #now compute predicted class label
        # pred_cls_lbl = [p_c[i[0]] for p_c,i in zip(pred_classes,indices)]
        # target_cls_lbl = [t_c[i[1]] for t_c, i in zip(tgt_labels, indices)]


        all_edge_lbl = []
        relation_feature = []
        freq_dist = []
        total_edge = 0
        total_fg = 0
        # loop through each of batch to collect the edge and node
        for b_id, (filtered_edge, rel_label, n, t_lbl, p_lbl) in enumerate(zip(filtered_edges, rel_labels, target_boxes, tgt_labels, pred_classes)):
            # find the -ve edges for training
            full_adj = torch.ones((n.shape[0],n.shape[0]))-torch.diag(torch.ones(n.shape[0]))
            full_adj[filtered_edge[:,0], filtered_edge[:,1]]=0
            neg_edges = torch.nonzero(full_adj).to(filtered_edge.device)

            # restrict unbalance in the +ve/-ve edge
            if filtered_edge.shape[0]>self.fg_edge:
                idx_ = torch.randperm(filtered_edge.shape[0])[:self.fg_edge]
                filtered_edge = filtered_edge[idx_,:]
                rel_label = rel_label[idx_]

            # check whether the number of -ve edges are within limit
            if neg_edges.shape[0]>=self.bg_edge:# random sample -ve edge
                idx_ = torch.randperm(neg_edges.shape[0])[:self.bg_edge]
                neg_edges = neg_edges[idx_,:]

            all_edges_ = torch.cat((filtered_edge, neg_edges), 0)
            total_edge += all_edges_.shape[0]
            total_fg += filtered_edge.shape[0]
            edge_labels = torch.cat((rel_label, torch.zeros(neg_edges.shape[0], dtype=torch.long).to(object_token.device)), 0)
            #now permute all the combination
            idx_ = torch.randperm(all_edges_.shape[0])
            all_edges_ = all_edges_[idx_,:]
            edge_labels = edge_labels[idx_]
            all_edge_lbl.append(edge_labels)
            # get the valid predicted matching
            pred_ids = indices[b_id][0]

            joint_emb = object_token[b_id, pred_ids, :]

            relation_feature.append(torch.cat((joint_emb[all_edges_[:,0],:],joint_emb[all_edges_[:,1],:],relation_token[b_id,...].repeat(all_edges_.shape[0],1)), 1))

            if self.freq:
                if self.use_target:
                    target_class = t_lbl[indices[b_id][1]]
                    freq_dist.append(self.freq_baseline.index_with_labels(torch.stack((target_class[all_edges_[:, 0]],target_class[all_edges_[:, 1]]),1)))
                else:
                    pred_class=p_lbl[pred_ids]
                    freq_dist.append(self.freq_baseline.index_with_labels(torch.stack((pred_class[all_edges_[:, 0]],pred_class[all_edges_[:, 1]]),1)))


        # torch.tensor(list(itertools.combinations(range(n.shape[0]), 2))).to(e.get_device())
        relation_feature = torch.cat(relation_feature, 0)

        all_edge_lbl = torch.cat(all_edge_lbl, 0).to(object_token.get_device())

        relation_pred = self.relation_embed(relation_feature)

        if len(freq_dist)>0: # add frequeny distribution wd predicted one
            batch_freq_dist = torch.cat(freq_dist, 0)
            relation_pred += batch_freq_dist

            loss =  F.cross_entropy(relation_pred, all_edge_lbl, reduction='mean')

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

    def forward(self, h, outputs, targets):
        """ This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        outputs_without_aux = {k: v for k, v in outputs.items() if k != 'aux_outputs' and k != 'enc_outputs'}
        # Retrieve the matching between the outputs of the last layer and the targets
        indices = self.matcher(outputs_without_aux, targets)

        # Compute the average number of target boxes accross all nodes, for normalization purposes
        num_boxes = sum(len(t["labels"]) for t in targets)
        num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=next(iter(outputs.values())).device)
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_boxes)
        num_boxes = torch.clamp(num_boxes / get_world_size(), min=1).item()

        tgt_labels = [v["labels"] for v in targets]
        tgt_boxes = [v["boxes"] for v in targets]
        tgt_edges = [v["edges"] for v in targets]
        object_token, relation_token = h
        if 'aux_outputs' in outputs:
            final_obj_tkn, aux_obj_tkn = object_token[-1], object_token[:-1]
            final_rel_tkn, aux_rel_tkn = relation_token[-1], relation_token[:-1]
        else:
            final_obj_tkn = object_token
            final_rel_tkn = relation_token
        #calculate losses
        losses = {}
        losses = self.get_loss(final_obj_tkn, final_rel_tkn, outputs, tgt_labels, tgt_boxes, tgt_edges, indices, num_boxes, losses) #todo check normalize boxes and edges

        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if 'aux_outputs' in outputs:
            for i, (obj_tkn,rel_tkn, aux_outputs) in enumerate(zip(aux_obj_tkn, aux_rel_tkn, outputs['aux_outputs'])):
                indices = self.matcher(aux_outputs, targets)
                l_dict = {}
                l_dict = self.get_loss(obj_tkn, rel_tkn, aux_outputs, tgt_labels, tgt_boxes, tgt_edges, indices, num_boxes, l_dict)
                l_dict = {k + f'_{i}': v for k, v in l_dict.items()}
                losses.update(l_dict)
        #in case of two stage loss
        if 'enc_outputs' in outputs:
            enc_outputs = outputs['enc_outputs']
            bin_targets = copy.deepcopy(targets)
            for bt in bin_targets:
                bt['labels'] = torch.zeros_like(bt['labels'])
            indices = self.matcher(enc_outputs, bin_targets)
            l_dict = {}
            l_dict = self.get_loss(None, None, enc_outputs, [v["labels"] for v in bin_targets], tgt_boxes, tgt_edges, indices, num_boxes,l_dict)
            l_dict = {k + f'_enc': v for k, v in l_dict.items()}
            losses.update(l_dict)
        #sum up whole loss
        losses['total'] = sum([losses[key] * self.weight_dict[key] for key in losses if key in self.weight_dict])

        return losses

    def get_loss(self, obj_tkn, rel_tkn, outputs, tgt_labels, tgt_boxes, tgt_edges, indices, num_boxes, losses):
        '''
        calculate losses across all data
        '''
        losses['class'] = self.loss_class(outputs['pred_logits'], tgt_labels, indices, num_boxes)
        losses['cards'] = self.loss_cardinality(outputs['pred_logits'], indices)
        losses['nodes'] = self.l1_loss(outputs['pred_boxes'], tgt_boxes, indices, num_boxes)
        losses['boxes'] = self.giou_loss(outputs['pred_boxes'], tgt_boxes, indices, num_boxes)
        if obj_tkn is not None: # for two stage, we are only interested in
            pred_labels = torch.argmax(outputs['pred_logits'], -1)
            if self.add_emd_rel:
                losses['edges'] = self.loss_edges(obj_tkn, rel_tkn, tgt_boxes, pred_labels, tgt_labels,
                                                  tgt_edges, indices, outputs['pred_logits'], outputs['pred_boxes'])
            else:
                losses['edges'] = self.loss_edges(obj_tkn,rel_tkn, tgt_boxes, pred_labels, tgt_labels, tgt_edges, indices)

        return losses

