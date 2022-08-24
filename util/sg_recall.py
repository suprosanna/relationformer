"""
Adapted from Danfei Xu. In particular, slow code was removed
"""
import os.path
import sys

import numpy as np
import math
import pickle
from functools import reduce
import torch
import pdb
from util.box_ops import intersect_2d,argsort_desc,box_cxcywh_to_xyxy# box_iou as bbox_overlaps
from util.box_ops_np import box_iou_np as bbox_overlaps
import torch.distributed as dist
MODES = ('sgdet',)  #'sgcls', 'predcls')
#proposed relation threshold
iou_thresh= 0.5
Rel_Threshold = 0.5
EDGE_THRESHOLD = 0.4#todo check what they are doing
np.set_printoptions(precision=3)
from abc import ABC
class BasicSceneGraphEvaluator(ABC):
    def __init__(self, mode, topfive_pred=False,**kwargs):
        super(BasicSceneGraphEvaluator, self).__init__()
        self.mode = mode
        self._buffers = {20: [], 50: [], 100: [], 'class_accuracy': []}
        self.rel_pred_acc = []
        self.edge_pred_acc = []
        self.edge_recall = []
        self.rel_recall = []
        self.topfive_pred = topfive_pred
        #todo only for testing the imact ,delete later on
        config = kwargs['config']
        self.sort_only_by_rel = config.DATA.SORT_ONLY_BY_REL
        self.use_gt_filter = config.DATA.USE_GT_FILTER
        self.multiple_preds =  hasattr(config.DATA,'MULTI_PRED') and config.DATA.MULTI_PRED

    @classmethod
    def all_modes(cls, **kwargs):
        evaluators = {m: cls(mode=m, **kwargs) for m in MODES}
        return evaluators

    @classmethod
    def vrd_modes(cls, **kwargs):
        evaluators = {m: cls(mode=m, multiple_preds=True, **kwargs) for m in ('preddet', 'phrdet')}
        return evaluators

    def evaluate_scene_graph_entry(self, gt_entry, pred_entry, viz_dict=None, iou_thresh=iou_thresh, vis=False):
        res = evaluate_from_dict(gt_entry, pred_entry, self=self, iou_thresh=iou_thresh, vis=vis)
        return res

    def reset(self):
        self._buffers = {20: [], 50: [], 100: [],200: [], 'class_accuracy': []}
        self.rel_pred_acc = []
        self.edge_pred_acc = []
        self.edge_recall = []
        self.rel_recall = []

    def print_stats(self, epoch_num=None, writer=None, return_output=False,file_path=None):
        #if not return_output:
        gather_output = [None for _ in range(dist.get_world_size())]
        dist.all_gather_object(gather_output, self._buffers)
        _buffers = {20: [], 50: [], 100: [], 200: [], 'class_accuracy': []}
        keys = list(gather_output[0].keys())
        tmp = [_buffers[key].append(ds[key]) for key in keys for ds in gather_output]
        output = {}
        if dist.get_rank()==0:
            if not return_output:
                print('\n======================' + self.mode + '============================')
            for k, v in _buffers.items():
                if k == 'class_accuracy':
                    if not return_output:
                        print('class accuracy: ',np.mean(v))
                    if writer is not None:
                        writer.add_scalar('obj_det/class accuracy', np.mean(v), epoch_num)
                else:
                    if not return_output:
                        print('R@%i: %f' % (k, np.mean(v)))
                    output['R@%i' % k] = np.mean(v)
                    if writer is not None:
                        writer.add_scalar('rel_pred/R@%i'%(k), np.mean(v), epoch_num)

            if not return_output:
                print('=======================================================')
            if writer is not None:
                writer.flush()
            if file_path is not None:
                f = open(file_path, 'a' if os.path.isfile(file_path) else 'w')
                graph_const = 'recall w/o graph constraint' if self.multiple_preds else 'recall with graph constraint'
                f.write('======================' + graph_const + '============================\n')
                f.write('mR@20: ' + str(np.round(output['R@20'], decimals=3)) + ' \n')
                f.write('mR@50:' + str(np.round(output['R@50'], decimals=3)) + ' \n')
                f.write('mR@100: ' + str(np.round(output['R@100'], decimals=3)) + ' \n')
                f.close()
        if return_output:
            return  output
        # #now write the relation prediction accuracy
        # if len(self.rel_pred_acc)>0 :
        #     print('Relation prediction Accuracy :',np.mean(self.rel_pred_acc))
        #     print('Predicted/ Actual Relation :', np.mean(self.rel_recall))
        #     if writer is not None:
        #         writer.add_scalar('pred_rel_acc', np.mean(self.rel_pred_acc), epoch_num)
        #         writer.add_scalar('pred_vs_actual_rel', np.mean(self.rel_recall), epoch_num)
        # # now write the correct edge prediction accuracy
        # if len(self.edge_pred_acc) > 0:
        #     print('Edge prediction Accuracy :', np.mean(self.edge_pred_acc))
        #     #print('Predicted/ Actual Edge :', np.mean(self.edge_recall))
        #     if writer is not None:
        #         writer.add_scalar('edge_pred_acc', np.mean(self.edge_pred_acc), epoch_num)
        #         #writer.add_scalar('pred_vs_actual_edge', np.mean(self.edge_recall), epoch_num)

def evaluate_from_dict(gt_entry, pred_entry, self=None, topfive=None, vis=False, **kwargs):
    """
    Shortcut to doing evaluate_recall from dict
    :param gt_entry: Dictionary containing gt_relations, gt_boxes, gt_classes
    :param pred_entry: Dictionary containing pred_rels, pred_boxes (if detection), pred_classes
    :param mode: 'det' or 'cls'
    :param result_dict:
    :param viz_dict:
    :param kwargs:
    :return:
    """
    gt_rels = gt_entry['edges']
    gt_boxes = box_cxcywh_to_xyxy(gt_entry['boxes'])
    gt_classes = gt_entry['labels']

    pred_rel_inds = pred_entry[1]['node_pair']
    rel_scores = pred_entry[1]['edge_score']

    pred_classes =  pred_entry[0]['labels'] #to avoid class_accu =1 for predcls

    if rel_scores is None: # if prediction is None
        for k in self._buffers:
            self._buffers[k].append(0.0)
        return None
    else:

        #now  how accurate the relation proposal network is
        if 'p_o_r_m' in pred_entry[1].keys():
            g_o_r_m = pred_entry[1]['g_o_r_m']
            p_o_r_m = pred_entry[1]['p_o_r_m']
            p_o_r_m[p_o_r_m< Rel_Threshold] = 0.01
            p_o_r_m[p_o_r_m >= Rel_Threshold] = 1

        if 'true_edge' in pred_entry[1].keys():
            true_edge = pred_entry['true_edge']
            pred_edge = np.where(pred_entry['pred_edge']>EDGE_THRESHOLD, 1, 0.1)  #0.1 is used that its not equal to 0

        if self.mode == 'predcls':
            pred_boxes = gt_boxes
            pred_classes = gt_classes       #todo for sorting print change here
            obj_scores = np.ones(gt_classes.shape[0])
        elif self.mode == 'sgcls':
            pred_boxes = gt_boxes
            pred_classes = pred_entry[0]['labels']
            obj_scores = pred_entry[0]['scores']
        elif self.mode == 'sgdet' or self.mode == 'phrdet':
            pred_boxes = box_cxcywh_to_xyxy(pred_entry[0]['boxes']).data.cpu().numpy() # if torch.is_tensor(pred_entry[0]['boxes']) else pred_entry[0]['boxes']
            pred_classes = pred_entry[0]['labels'].data.cpu().numpy() if torch.is_tensor(pred_entry[0]['labels']) else pred_entry[0]['labels']
            obj_scores = pred_entry[0]['scores'].data.cpu().numpy() if torch.is_tensor(pred_entry[0]['scores']) else pred_entry[0]['scores']
        elif self.mode == 'preddet':
            # Only extract the indices that appear in GT
            prc = intersect_2d(pred_rel_inds, gt_rels[:, :2])
            if prc.size == 0:
                for k in self._buffers:
                    self._buffers[k].append(0.0)
                return None, None, None
            pred_inds_per_gt = prc.argmax(0)
            pred_rel_inds = pred_rel_inds[pred_inds_per_gt]
            rel_scores = rel_scores[pred_inds_per_gt]

            # Now sort the matching ones
            rel_scores_sorted = argsort_desc(rel_scores[:,1:])
            rel_scores_sorted[:,1] += 1
            rel_scores_sorted = np.column_stack((pred_rel_inds[rel_scores_sorted[:,0]], rel_scores_sorted[:,1]))

            matches = intersect_2d(rel_scores_sorted, gt_rels)
            for k in self._buffers:
                rec_i = float(matches[:k].any(0).sum()) / float(gt_rels.shape[0])
                self._buffers[k].append(rec_i)
            return None, None, None
        else:
            raise ValueError('invalid mode')

        if self.multiple_preds:
            obj_scores_per_rel = obj_scores[pred_rel_inds].prod(1)
            overall_scores = obj_scores_per_rel[:,None] * rel_scores[:,1:]
            score_inds = argsort_desc(overall_scores)[:100]
            pred_rels = np.column_stack((pred_rel_inds[score_inds[:,0]], score_inds[:,1]+1))
            predicate_scores = rel_scores[score_inds[:,0], score_inds[:,1]+1]
        else:
            pred_rels = np.column_stack((pred_rel_inds, 1+rel_scores[:,1:].argmax(1)))
            predicate_scores = rel_scores[:,1:].max(1)

        if self.mode in ('predcls', 'sgcls') and topfive:
            gt_rels_ind = pred_entry['gt_rel_inds']
            concat_gt_pred = np.column_stack((pred_classes[gt_rels_ind[:,:2]],gt_rels_ind[:,2],
                                              1+np.argsort(rel_scores[:, 1:], axis=1)[:, ::-1][:,:5])) #[s,o,r0,r1,r2,r3,r4]
        if self.mode=='sgdet':
            iou_overlap = bbox_overlaps(gt_boxes, pred_boxes)
            if self.use_gt_filter:
                idx = torch.where(iou_overlap >= 0.5)
                valid_rels_idx = np.asarray([i for i, rel in enumerate(pred_rels) if (rel[0] in idx[1]) and (rel[1] in idx[1])]) #filter the junk detections
                if len(valid_rels_idx)>=1:
                    pred_rels = pred_rels[valid_rels_idx,:]
                    predicate_scores = predicate_scores[valid_rels_idx]
            if self.sort_only_by_rel:
                sorted_rel_idx = np.argsort(predicate_scores)[::-1]
                pred_rels = pred_rels[sorted_rel_idx]
                predicate_scores = predicate_scores[sorted_rel_idx]
        pred_to_gt, pred_5ples, rel_scores, sort_idx = evaluate_recall(
                    gt_rels, gt_boxes, gt_classes,
                    pred_rels, pred_boxes, pred_classes,
                    predicate_scores, obj_scores, phrdet= self.mode=='phrdet', vis=vis,
                    **kwargs)

        if self.mode in ('predcls', 'sgcls', 'sgdet', 'phrdet'):
            if self.mode=='sgcls':
                class_accuracy = (gt_classes==pred_classes).sum()/len(gt_classes)
            elif self.mode=='sgdet':
                pred_obj_th0 = pred_to_gt_obj(iou_overlap > iou_thresh, iou_overlap, gt_classes, pred_classes)
                # pred_obj_th1 = pred_to_gt_obj(iou_overlap >iou_thresh+0.1,iou_overlap, gt_classes,pred_classes)
                class_accuracy = len(pred_obj_th0) / len(gt_classes)
            else:
                class_accuracy=0.0
            if 'p_o_r_m' in pred_entry[1].keys():
                self.rel_pred_acc.append((p_o_r_m==g_o_r_m).sum()/g_o_r_m.sum())
                self.rel_recall.append(p_o_r_m.sum()/g_o_r_m.sum())
            if 'true_edge' in pred_entry[1].keys():
                self.edge_pred_acc.append((pred_edge==true_edge).sum()/true_edge.sum())
                self.edge_recall.append(np.count_nonzero(pred_edge==1)  / true_edge.sum())


        for k in self._buffers:
            if k == 'class_accuracy':
                self._buffers[k].append(class_accuracy)
            else:
                match = reduce(np.union1d, pred_to_gt[:k])    #todo check for inreasing match
                rec_i = float(len(match)) / float(gt_rels.shape[0])
                self._buffers[k].append(rec_i)

        return pred_to_gt, pred_5ples, rel_scores, concat_gt_pred if topfive else None, sort_idx


###########################
def evaluate_recall( gt_rels, gt_boxes, gt_classes,
                    pred_rels, pred_boxes, pred_classes, rel_scores=None, cls_scores=None,
                    iou_thresh=0.5, phrdet=False, vis=False):
    """
    Evaluates the recall
    :param gt_rels: [#gt_rel, 3] array of GT relations
    :param gt_boxes: [#gt_box, 4] array of GT boxes
    :param gt_classes: [#gt_box] array of GT classes
    :param pred_rels: [#pred_rel, 3] array of pred rels. Assumed these are in sorted order
                      and refer to IDs in pred classes / pred boxes
                      (id0, id1, rel)
    :param pred_boxes:  [#pred_box, 4] array of pred boxes
    :param pred_classes: [#pred_box] array of predicted classes for these boxes
    :return: pred_to_gt: Matching from predicate to GT
             pred_5ples: the predicted (id0, id1, cls0, cls1, rel)
             rel_scores: [cls_0score, cls1_score, relscore]
                   """
    if pred_rels.size == 0:
        return [[]], np.zeros((0,5)), np.zeros(0)

    num_gt_boxes = gt_boxes.shape[0]
    num_gt_relations = gt_rels.shape[0]
    assert num_gt_relations != 0

    gt_triplets, gt_triplet_boxes, _ = _triplet(gt_rels[:, 2],
                                                gt_rels[:, :2],
                                                gt_classes,
                                                gt_boxes)
    num_boxes = pred_boxes.shape[0]
    assert pred_rels[:,:2].max() < pred_classes.shape[0]

    # Exclude self rels
    # assert np.all(pred_rels[:,0] != pred_rels[:,1])
    assert np.all(pred_rels[:,2] > 0)

    pred_triplets, pred_triplet_boxes, relation_scores = \
        _triplet(pred_rels[:,2], pred_rels[:,:2], pred_classes, pred_boxes,
                 rel_scores, cls_scores)

    scores_overall = relation_scores.prod(1)
    
    if vis==True:
        sort_idx = np.argsort(scores_overall)[::-1]
        pred_triplets = pred_triplets[sort_idx]
        pred_triplet_boxes = pred_triplet_boxes[sort_idx]
        relation_scores = relation_scores[sort_idx]
    else:
        sort_idx=None

    # Compute recall. It's most efficient to match once and then do recall after
    pred_to_gt = _compute_pred_matches(
        gt_triplets,
        pred_triplets,
        gt_triplet_boxes,
        pred_triplet_boxes,
        iou_thresh,
        phrdet=phrdet,
    )

    # Contains some extra stuff for visualization. Not needed.
    pred_5ples = np.column_stack((
        pred_rels[:,:2],
        pred_triplets[:, [0, 2, 1]],
    ))

    return pred_to_gt, pred_5ples, relation_scores, sort_idx


def _triplet(predicates, relations, classes, boxes,
             predicate_scores=None, class_scores=None):
    """
    format predictions into triplets
    :param predicates: A 1d numpy array of num_boxes*(num_boxes-1) predicates, corresponding to
                       each pair of possibilities
    :param relations: A (num_boxes*(num_boxes-1), 2) array, where each row represents the boxes
                      in that relation
    :param classes: A (num_boxes) array of the classes for each thing.
    :param boxes: A (num_boxes,4) array of the bounding boxes for everything.
    :param predicate_scores: A (num_boxes*(num_boxes-1)) array of the scores for each predicate
    :param class_scores: A (num_boxes) array of the likelihood for each object.
    :return: Triplets: (num_relations, 3) array of class, relation, class
             Triplet boxes: (num_relation, 8) array of boxes for the parts
             Triplet scores: num_relation array of the scores overall for the triplets
    """
    assert (predicates.shape[0] == relations.shape[0])

    sub_ob_classes = classes[relations[:, :2]]
    triplets = np.column_stack((sub_ob_classes[:, 0], predicates, sub_ob_classes[:, 1]))
    triplet_boxes = np.column_stack((boxes[relations[:, 0]], boxes[relations[:, 1]]))

    triplet_scores = None
    if predicate_scores is not None and class_scores is not None:
        triplet_scores = np.column_stack((
            class_scores[relations[:, 0]],
            class_scores[relations[:, 1]],
            predicate_scores,
        ))

    return triplets, triplet_boxes, triplet_scores


def _compute_pred_matches(gt_triplets, pred_triplets,
                 gt_boxes, pred_boxes, iou_thresh, phrdet=False):
    """
    Given a set of predicted triplets, return the list of matching GT's for each of the
    given predictions
    :param gt_triplets:
    :param pred_triplets:
    :param gt_boxes:
    :param pred_boxes:
    :param iou_thresh:
    :return:
    """
    # This performs a matrix multiplication-esque thing between the two arrays
    # Instead of summing, we want the equality, so we reduce in that way
    # The rows correspond to GT triplets, columns to pred triplets
    keeps = intersect_2d(gt_triplets[:,:2], pred_triplets[:,:2])
    gt_has_match = keeps.any(1)
    pred_to_gt = [[] for x in range(pred_boxes.shape[0])]
    for gt_ind, gt_box, keep_inds in zip(np.where(gt_has_match)[0],
                                         gt_boxes[gt_has_match],
                                         keeps[gt_has_match],
                                         ):
        boxes = pred_boxes[keep_inds]
        if phrdet:
            # Evaluate where the union box > 0.5
            gt_box_union = gt_box.reshape((2, 4))
            gt_box_union = np.concatenate((gt_box_union.min(0)[:2], gt_box_union.max(0)[2:]), 0)

            box_union = boxes.reshape((-1, 2, 4))
            box_union = np.concatenate((box_union.min(1)[:,:2], box_union.max(1)[:,2:]), 1)

            inds = bbox_overlaps(gt_box_union[None], box_union)[0] >= iou_thresh

        else:
            sub_iou = bbox_overlaps(gt_box[None,:4], boxes[:, :4])[0]
            obj_iou = bbox_overlaps(gt_box[None,4:], boxes[:, 4:])[0]

            inds = (sub_iou >= iou_thresh) & (obj_iou >= iou_thresh)

        for i in np.where(keep_inds)[0][inds]:
            pred_to_gt[i].append(int(gt_ind))
    return pred_to_gt


def calculate_mR_from_evaluator_list(evaluator_list, mode, file_path, save_file=None):
    all_rel_results = {}
    for (pred_id, pred_name, evaluator_rel) in evaluator_list:
        # print('\n')
        # print('relationship: ', pred_name)
        rel_results = evaluator_rel[mode].print_stats(return_output=True)
        all_rel_results[pred_name] = rel_results

    mean_recall = {}
    mR20 = 0.0
    mR50 = 0.0
    mR100 = 0.0
    mR200 = 0.0
    print('------------------meanRecall@100:---------------------')
    for key, value in all_rel_results.items():
        if math.isnan(value['R@100']) or key =='R@200':
            continue
        print("{0} : {1:.2f}".format(key,value['R@100']),end=" :: ")
        mR20 += value['R@20']
        mR50 += value['R@50']
        mR100 += value['R@100']
    rel_num = len(evaluator_list)
    mR20 /= rel_num
    mR50 /= rel_num
    mR100 /= rel_num
    mR200 /= rel_num
    mean_recall['R@20'] = mR20
    mean_recall['R@50'] = mR50
    mean_recall['R@100'] = mR100
    all_rel_results['mean_recall'] = mean_recall

    multi_preds = evaluator_rel[mode].multiple_preds
    recall_mode = 'mean recall without constraint' if multi_preds else 'mean recall with constraint'
    print('\n')
    print('======================' + mode + '  ' + recall_mode + '============================')
    print('mR@20: ', mR20)
    print('mR@50: ', mR50)
    print('mR@100: ', mR100)
    #print('mR@200: ', mR200)
    if file_path is not None and save_file:
        f = open(file_path, 'a' if os.path.isfile(file_path) else 'w')
        f.write('======================'+ recall_mode + '============================\n')
        f.write('mR@20: ' + str(np.round(mR20, decimals=3)) + ' \n')
        f.write('mR@50: ' + str(np.round(mR50, decimals=3)) + ' \n')
        f.write('mR@100: ' + str(np.round(mR100, decimals=3)) + ' \n')
        f.close()


def eval_entry(mode, gt_entry, pred_entry, evaluator, evaluator_list, vis=False):
    evaluator[mode].evaluate_scene_graph_entry(
        gt_entry,
        pred_entry,
        vis=vis,
    )

    for (pred_id, _, evaluator_rel) in evaluator_list:
        gt_entry_rel = gt_entry.copy()
        mask = np.in1d(gt_entry_rel['gt_relations'][:, -1], pred_id)
        gt_entry_rel['gt_relations'] = gt_entry_rel['gt_relations'][mask, :]
        if gt_entry_rel['gt_relations'].shape[0] == 0:
            continue

        evaluator_rel[mode].evaluate_scene_graph_entry(
            gt_entry_rel,
            pred_entry,
            vis=vis,
        )

###########################################
def pred_to_gt_obj(overlap_wd_thresh,overlap_score, gt_classes,pred_classes):
    """
    compute class accuracy with predcited and obj class
    """
    pred_to_gt = {}
    for i,gt in enumerate(overlap_wd_thresh):
        matches = torch.nonzero(gt)
        if len(matches)>0:
            gt_class = gt_classes[i]
            max_iou = 0
            for j, match in enumerate(matches):
                if gt_class==pred_classes[match]:
                    curr_iou = overlap_score[i, match]
                    if curr_iou>max_iou:
                        pred_to_gt[i]=match
        # else:
        #     pred_to_gt.append([])
    return pred_to_gt
