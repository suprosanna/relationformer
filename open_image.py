import json
import logging
import os
import pickle
import random
from collections import defaultdict, OrderedDict, Counter

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt
from util.bounding_box import BoxList

# from pysgg.config import cfg
# from pysgg.data.datasets.visual_genome import resampling_dict_generation, get_VG_statistics, \
#     apply_resampling
# from pysgg.structures.bounding_box import BoxList
# from pysgg.structures.boxlist_ops import split_boxlist, cat_boxlist
# from pysgg.utils.comm import get_rank, synchronize

HEAD = []
BODY = []
TAIL = []

# for i, cate in enumerate(cfg.MODEL.ROI_RELATION_HEAD.LONGTAIL_PART_DICT):
#     if cate == 'h':
#         HEAD.append(i)
#     elif cate == 'b':
#         BODY.append(i)
#     elif cate == 't':
#         TAIL.append(i)


def load_cate_info(dict_file, add_bg=True):
    """
    Loads the file containing the visual genome label meanings
    """
    info = json.load(open(dict_file, 'r'))
    ind_to_predicates_cate = ['__background__'] + info['rel']
    ind_to_entites_cate = ['__background__'] + info['obj']

    # print(len(ind_to_predicates_cate))
    # print(len(ind_to_entites_cate))
    predicate_to_ind = {idx: name for idx, name in enumerate(ind_to_predicates_cate)}
    entites_cate_to_ind = {idx: name for idx, name in enumerate(ind_to_entites_cate)}

    return (ind_to_entites_cate, ind_to_predicates_cate,
            entites_cate_to_ind, predicate_to_ind)


def load_annotations(annotation_file, img_dir, num_img, split,
                     filter_empty_rels, ):
    """

    :param annotation_file:
    :param img_dir:
    :param img_range:
    :param filter_empty_rels:
    :return:
        image_index: numpy array corresponding to the index of images we're using
        boxes: List where each element is a [num_gt, 4] array of ground
                    truth boxes (x1, y1, x2, y2)
        gt_classes: List where each element is a [num_gt] array of classes
        relationships: List where each element is a [num_r, 3] array of
                    (box_ind_1, box_ind_2, predicate) relationships
    """

    annotations = json.load(open(annotation_file, 'r'))

    if num_img == -1:
        num_img = len(annotations)

    annotations = annotations[: num_img]

    empty_list = set()
    if filter_empty_rels:
        for i, each in enumerate(annotations):
            if len(each['rel']) == 0:
                empty_list.add(i)
            if len(each['bbox']) == 0:
                empty_list.add(i)

    print('empty relationship image num: ', len(empty_list))

    boxes = []
    gt_classes = []
    relationships = []
    img_info = []
    for i, anno in enumerate(annotations):

        if i in empty_list:
            continue

        boxes_i = np.array(anno['bbox'])
        gt_classes_i = np.array(anno['det_labels'], dtype=int)

        rels = np.array(anno['rel'], dtype=int)

        gt_classes_i += 1
        rels[:, -1] += 1

        image_info = {
            'width': anno['img_size'][0],
            'height': anno['img_size'][1],
            'img_fn': os.path.join(img_dir, anno['img_fn'] + '.jpg')
        }

        boxes.append(boxes_i)
        gt_classes.append(gt_classes_i)
        relationships.append(rels)
        img_info.append(image_info)

    return boxes, gt_classes, relationships, img_info


class OIDataset(torch.utils.data.Dataset):

    def __init__(self, split, img_dir, ann_file, cate_info_file, transforms=None,
                 num_im=-1, check_img_file=False, filter_duplicate_rels=True, flip_aug=False,DEBUG=False,config=None):
        """
        Torch dataset for VisualGenome
        Parameters:
            split: Must be train, test, or val
            img_dir: folder containing all vg images
            roidb_file:  HDF5 containing the GT boxes, classes, and relationships
            dict_file: JSON Contains mapping of classes/relationships to words
            image_file: HDF5 containing image filenames
            filter_empty_rels: True if we filter out images without relationships between
                             boxes. One might want to set this to false if training a detector.
            filter_duplicate_rels: Whenever we see a duplicate relationship we'll sample instead
            num_im: Number of images in the entire dataset. -1 for all images.
            num_val_im: Number of images in the validation set (must be less than num_im
               unless num_im is -1.)
        """
        # for debug
        if DEBUG:
            num_im = 200
        #
        # num_im = 20000
        # num_val_im = 1000

        assert split in {'train', 'val', 'test'}
        self.flip_aug = flip_aug
        self.split = split
        self.img_dir = img_dir
        self.cate_info_file = cate_info_file
        self.annotation_file = ann_file
        self.filter_duplicate_rels = filter_duplicate_rels and self.split == 'train'
        self.transforms = transforms
        self.repeat_dict = None
        self.check_img_file = check_img_file
        self.remove_tail_classes = False

        (self.ind_to_classes,
         self.ind_to_predicates,
         self.classes_to_ind,
         self.predicates_to_ind) = load_cate_info(self.cate_info_file)  # contiguous 151, 51 containing __background__

        logger = logging.getLogger("pysgg.dataset")
        self.logger = logger
        self.n_queries = config.MODEL.DECODER.NUM_QUERIES


        self.categories = {i: self.ind_to_classes[i]
                           for i in range(len(self.ind_to_classes))}

        self.gt_boxes, self.gt_classes, self.relationships, self.img_info, = load_annotations(
            self.annotation_file, img_dir, num_im, split=split,
            filter_empty_rels= True,
        )

        self.filenames = [img_if['img_fn'] for img_if in self.img_info]
        self.idx_list = list(range(len(self.filenames)))

        self.id_to_img_map = {k: v for k, v in enumerate(self.idx_list)}

        self.pre_compute_bbox = None
        # if cfg.DATASETS.LOAD_PRECOMPUTE_DETECTION_BOX:
        #     """precoompute boxes format:
        #         index by the image id, elem has "scores", "bbox", "cls", 3 fields
        #     """
        #     with open(os.path.join("datasets/vg/stanford_spilt", "detection_precompute_boxes_all.pkl"), 'rb') as f:
        #         self.pre_compute_bbox = pickle.load(f)
        #     self.logger.info("load pre-compute box length %d" %
        #                      (len(self.pre_compute_bbox.keys())))
        #
        # if cfg.MODEL.ROI_RELATION_HEAD.DATA_RESAMPLING and self.split == 'train':
        #     self.resampling_method = cfg.MODEL.ROI_RELATION_HEAD.DATA_RESAMPLING_METHOD
        #     assert self.resampling_method in ['bilvl', 'lvis']
        #
        #     self.global_rf = cfg.MODEL.ROI_RELATION_HEAD.DATA_RESAMPLING_PARAM.REPEAT_FACTOR
        #     self.drop_rate = cfg.MODEL.ROI_RELATION_HEAD.DATA_RESAMPLING_PARAM.INSTANCE_DROP_RATE
        #     # creat repeat dict in main process, other process just wait and load
        #     if get_rank() == 0:
        #         repeat_dict = resampling_dict_generation(self, self.ind_to_predicates, logger)
        #         self.repeat_dict = repeat_dict
        #         with open(os.path.join(cfg.OUTPUT_DIR, "repeat_dict.pkl"), "wb") as f:
        #             pickle.dump(self.repeat_dict, f)
        #
        #     synchronize()
        #     self.repeat_dict = resampling_dict_generation(self, self.ind_to_predicates, logger)

            # duplicate_idx_list = []
            # for idx in range(len(self.filenames)):
            #     r_c = self.repeat_dict[idx]
            #     duplicate_idx_list.extend([idx for _ in range(r_c)])
            # self.idx_list = duplicate_idx_list

    def __getitem__(self, index):
        # if self.split == 'train':
        #    while(random.random() > self.img_info[index]['anti_prop']):
        #        index = int(random.random() * len(self.filenames))
        if self.repeat_dict is not None:
            index = self.idx_list[index]

        img = Image.open(self.filenames[index]).convert("RGB")
        if img.size[0] != self.img_info[index]['width'] or img.size[1] != self.img_info[index]['height']:
            print('=' * 20, ' ERROR index ', str(index), ' ', str(img.size), ' ', str(self.img_info[index]['width']),
                  ' ', str(self.img_info[index]['height']), ' ', '=' * 20)

        flip_img = False

        #target = self.get_groundtruth(index, flip_img)
        img_info = self.img_info[index]
        w, h = img_info['width'], img_info['height']
        gt_boxes = self.gt_boxes[index]
        gt_classes = self.gt_classes[index].copy()
        gt_rels = self.relationships[index].copy()
        if len(gt_boxes)> self.n_queries-2:
            idx = np.random.choice(torch.arange(len(gt_boxes)),self.n_queries-2,replace=False)
            gt_boxes = gt_boxes[idx,:]
            gt_classes = gt_classes[idx]
            gt_rels = [[idx.tolist().index(rel[0]),idx.tolist().index(rel[1]),rel[2]]for rel in gt_rels if rel[0] in idx and rel[1] in idx]
            gt_rels = np.array(gt_rels)
        target = {}
        target["iscrowd"] = torch.zeros(gt_boxes.shape[0])
        target["boxes"] = torch.as_tensor(gt_boxes, dtype=torch.float)
        target["labels"] = torch.as_tensor(gt_classes, dtype=torch.int64)
        target["edges"] = torch.as_tensor(gt_rels, dtype=torch.int64)
        target["image_id"] = img_info['img_fn']
        target["orig_size"] = torch.as_tensor([int(h), int(w)])
        # todo add pre-compute boxes
        pre_compute_boxlist = None

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target #, index

    def get_statistics(self):
        fg_matrix, bg_matrix, rel_counter_init = None #get_VG_statistics(self,
                                                                   #must_overlap=True)
        eps = 1e-3
        bg_matrix += 1
        fg_matrix[:, :, 0] = bg_matrix
        pred_dist = fg_matrix / fg_matrix.sum(2)[:, :, None] + eps

        result = {
            'fg_matrix': torch.from_numpy(fg_matrix),
            'pred_dist': torch.from_numpy(pred_dist).float(),
            'obj_classes': self.ind_to_classes,
            'rel_classes': self.ind_to_predicates,
            'att_classes': [],
        }

        rel_counter = Counter()

        for i in tqdm(self.idx_list):

            relation = self.relationships[i].copy()  # (num_rel, 3)
            if self.filter_duplicate_rels:
                # Filter out dupes!
                assert self.split == 'train'
                old_size = relation.shape[0]
                all_rel_sets = defaultdict(list)
                for (o0, o1, r) in relation:
                    all_rel_sets[(o0, o1)].append(r)
                relation = [(k[0], k[1], np.random.choice(v))
                            for k, v in all_rel_sets.items()]
                relation = np.array(relation, dtype=np.int32)

            if self.repeat_dict is not None:
                relation, _ = apply_resampling(i,
                                               relation,
                                               self.repeat_dict,
                                               self.drop_rate, )

            for i in relation[:, -1]:
                if i > 0:
                    rel_counter[i] += 1

        cate_num = []
        cate_num_init = []
        cate_set = []
        counter_name = []

        sorted_cate_list = [i[0] for i in rel_counter_init.most_common()]
        lt_part_dict = cfg.MODEL.ROI_RELATION_HEAD.LONGTAIL_PART_DICT
        for cate_id in sorted_cate_list:
            if lt_part_dict[cate_id] == 'h':
                cate_set.append(0)
            if lt_part_dict[cate_id] == 'b':
                cate_set.append(1)
            if lt_part_dict[cate_id] == 't':
                cate_set.append(2)

            counter_name.append(self.ind_to_predicates[cate_id])  # list start from 0
            cate_num.append(rel_counter[cate_id])  # dict start from 1
            cate_num_init.append(rel_counter_init[cate_id])  # dict start from 1

        pallte = ['r', 'g', 'b']
        color = [pallte[idx] for idx in cate_set]

        fig, axs_c = plt.subplots(2, 1, figsize=(13, 10), tight_layout=True)
        fig.set_facecolor((1, 1, 1))

        axs_c[0].bar(counter_name, cate_num_init, color=color, width=0.6, zorder=0)
        axs_c[0].grid()
        plt.sca(axs_c[0])
        plt.xticks(rotation=-90, )

        axs_c[1].bar(counter_name, cate_num, color=color, width=0.6, zorder=0)
        axs_c[1].grid()
        axs_c[1].set_ylim(0, 50000)
        plt.sca(axs_c[1])
        plt.xticks(rotation=-90, )

        save_file = os.path.join(cfg.OUTPUT_DIR, f"rel_freq_dist.png")
        fig.savefig(save_file, dpi=300)

        return result

    def get_img_info(self, index):
        # WARNING: original image_file.json has several pictures with false image size
        # use correct function to check the validity before training
        # it will take a while, you only need to do it once

        # correct_img_info(self.img_dir, self.image_file)
        return self.img_info[index]

    def get_groundtruth(self, index, evaluation=False, flip_img=False, inner_idx=True):
        if not inner_idx:
            # here, if we pass the index after resampeling, we need to map back to the initial index
            if self.repeat_dict is not None:
                index = self.idx_list[index]

        img_info = self.img_info[index]
        w, h = img_info['width'], img_info['height']
        box = self.gt_boxes[index]
        box = torch.from_numpy(box)  # guard against no boxes
        if flip_img:
            new_xmin = w - box[:, 2]
            new_xmax = w - box[:, 0]
            box[:, 0] = new_xmin
            box[:, 2] = new_xmax
        target = BoxList(box, (w, h), 'xyxy')  # xyxy

        target.add_field("labels", torch.from_numpy(self.gt_classes[index]))
        target.add_field("attributes", torch.from_numpy(np.zeros((len(self.gt_classes[index]), 10))))

        relation = self.relationships[index].copy()  # (num_rel, 3)
        if self.filter_duplicate_rels:
            # Filter out dupes!
            assert self.split == 'train'
            old_size = relation.shape[0]
            all_rel_sets = defaultdict(list)
            for (o0, o1, r) in relation:
                all_rel_sets[(o0, o1)].append(r)
            relation = [(k[0], k[1], np.random.choice(v))
                        for k, v in all_rel_sets.items()]
            relation = np.array(relation, dtype=np.int32)

        relation_non_masked = None
        if self.repeat_dict is not None:
            relation, relation_non_masked = apply_resampling(index,
                                                             relation,
                                                             self.repeat_dict,
                                                             self.drop_rate, )
        # add relation to target
        num_box = len(target)
        relation_map_non_masked = None
        if self.repeat_dict is not None:
            relation_map_non_masked = torch.zeros((num_box, num_box), dtype=torch.long)

        relation_map = torch.zeros((num_box, num_box), dtype=torch.long)
        for i in range(relation.shape[0]):
            # Sometimes two objects may have multiple different ground-truth predicates in VisualGenome.
            # In this case, when we construct GT annotations, random selection allows later predicates
            # having the chance to overwrite the precious collided predicate.
            if relation_map[int(relation[i, 0]), int(relation[i, 1])] != 0:
                if (random.random() > 0.5):
                    relation_map[int(relation[i, 0]), int(relation[i, 1])] = int(relation[i, 2])
                    if relation_map_non_masked is not None:
                        relation_map_non_masked[int(relation_non_masked[i, 0]),
                                                int(relation_non_masked[i, 1])] = int(relation_non_masked[i, 2])
            else:
                relation_map[int(relation[i, 0]), int(relation[i, 1])] = int(relation[i, 2])
                if relation_map_non_masked is not None:
                    relation_map_non_masked[int(relation_non_masked[i, 0]),
                                            int(relation_non_masked[i, 1])] = int(relation_non_masked[i, 2])

        target.add_field("relation", relation_map, is_triplet=True)
        if relation_map_non_masked is not None:
            target.add_field("relation_non_masked", relation_map_non_masked.long(), is_triplet=True)

        if evaluation:
            target = target.clip_to_image(remove_empty=False)
            target.add_field("relation_tuple", torch.LongTensor(relation))  # for evaluation
            return target
        else:
            target = target.clip_to_image(remove_empty=True)
            return target

    def __len__(self):
        return len(self.idx_list)


if __name__ == '__main__':
    oi_ds = OIDataset(split='train',img_dir='data/open-imagev6/images',
                      ann_file='data/open-imagev6/annotations/vrd-train-anno.json',
                      cate_info_file='data/open-imagev6/annotations/categories_dict.json')
    print(oi_ds)