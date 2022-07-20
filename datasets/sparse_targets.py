import torch.nn as nn
import torch
import numpy as np
import os
from datasets.get_dataset_counts import get_counts
from torch.autograd import Variable
import torch.nn.functional as F

class FrequencyBias(nn.Module):
    """
    The goal of this is to provide a simplified way of computing
    P(predicate | obj1, obj2, img).
    """

    def __init__(self, file_path, train_ds, eps=1e-3, dropout=0.0,logsoftmax=False):
        super(FrequencyBias, self).__init__()
        # create freq baseline embedding
        self.num_objs = train_ds.num_classes
        self.logsoftmax=logsoftmax
        self.obj_baseline = nn.Embedding(np.square(train_ds.num_classes), train_ds.num_predicates)
        if dropout >0.0:
            self.drpout = nn.Dropout(p=dropout)
        if os.path.exists(file_path) and file_path=="data/VG.pt" and not logsoftmax:
            pred_dist  = torch.load(file_path,map_location=torch.device('cpu'))
            self.obj_baseline.weight.data=pred_dist
        elif logsoftmax:
            if os.path.exists('data/VG_logsoftmax.pt'):
                pred_dist = torch.load('data/VG_logsoftmax.pt', map_location=torch.device('cpu'))
                self.obj_baseline.weight.data = pred_dist
            else: #todo now only implemented for log softmax
                fg_matrix, bg_matrix = get_counts(train_ds,must_overlap=True)
                #bg_matrix += 1
                fg_matrix[:, :, 0] = bg_matrix

                pred_dist = fg_matrix / (fg_matrix.sum(2)[:, :, None] + eps)

                self.num_objs = pred_dist.shape[0]
                pred_dist = torch.FloatTensor(pred_dist).view(-1, pred_dist.shape[2])
                pred_dist = torch.nn.functional.log_softmax(Variable(pred_dist), dim=-1).data
                torch.save(pred_dist, "data/VG_logsoftmax.pt")
                self.obj_baseline = nn.Embedding(pred_dist.size(0), pred_dist.size(1))
                self.obj_baseline.weight.data = pred_dist

        #self.obj_baseline.weight.requires_grad =False

    def index_with_labels(self, labels):
        """
        :param labels: [batch_size, 2]
        :return:
        """
        if hasattr(self,'drpout'):
            bias =  self.drpout(self.obj_baseline(labels[:, 0] * self.num_objs + labels[:, 1]))
        else:
            bias = self.obj_baseline(labels[:, 0] * self.num_objs + labels[:, 1])
        if self.logsoftmax:
            return torch.sigmoid(bias)
        return bias

    def forward(self, obj_cands0, obj_cands1):
        """
        :param obj_cands0: [batch_size, 151] prob distibution over cands.
        :param obj_cands1: [batch_size, 151] prob distibution over cands.
        :return: [batch_size, #predicates] array, which contains potentials for
        each possibility
        """
        # [batch_size, 151, 151] repr of the joint distribution
        joint_cands = obj_cands0[:, :, None] * obj_cands1[:, None]

        # [151, 151, 51] of targets per.
        baseline = joint_cands.view(joint_cands.size(0), -1) @ self.obj_baseline.weight

        return baseline
