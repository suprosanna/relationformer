"""
File that involves dataloaders for the Visual Genome dataset.
"""

import json
import os
import random
import h5py
import numpy as np
import pickle
from PIL import Image
from torch.utils.data import Dataset
from collections import defaultdict
from pathlib import Path
import torch
import torch.utils.data
import transforms as T
from box_ops_2D import box_iou
from open_image import OIDataset


class visual_genome_loader(Dataset):
    def __init__(self, data_dicts, transforms, config, is_train=False):
        self.data_dicts = data_dicts
        self.transforms = transforms
        self.BOX_SCALE = config.DATA.BOX_SCALE
        self.BG_EDGE_PER_IMG = config.DATA.BG_EDGE_PER_IMG
        self.FG_EDGE_PER_IMG = config.DATA.FG_EDGE_PER_IMG
        self.num_classes = config.MODEL.NUM_OBJ_CLS+1
        self.num_predicates = config.MODEL.NUM_REL_CLS+1
        self.is_train=is_train
        self.dataset = config.DATA.DATASET
        self.n_queries = config.MODEL.DECODER.NUM_QUERIES

    def __getitem__(self, index):

        ##################### Load the image data
        image_unpadded = Image.open(self.data_dicts[index]['image']).convert('RGB')
        w, h = image_unpadded.size

        ##################### Load the box location data
        gt_boxes = self.data_dicts[index]['boxes'].copy()
        gt_classes = self.data_dicts[index]['label'].copy()
        ##################### Load the relationship data
        gt_rels = self.data_dicts[index]['edges'].copy()
        if len(gt_boxes)> self.n_queries-2:
            idx = np.random.choice(torch.arange(len(gt_boxes)),self.n_queries-2,replace=False)
            gt_boxes = gt_boxes[idx,:]
            gt_classes = gt_classes[idx]
            gt_rels = [[idx.tolist().index(rel[0]),idx.tolist().index(rel[1]),rel[2]]for rel in gt_rels if rel[0] in idx and rel[1] in idx]
            gt_rels = np.array(gt_rels)
        # Boxes are already at BOX_SCALE
        if self.dataset=='VG':
            gt_boxes = gt_boxes/self.BOX_SCALE

            # crop boxes that are too large. This seems to be only a problem for image heights, but whatevs
            gt_boxes = gt_boxes.clip(0, 1)

                # # crop the image for data augmentation
                # image_unpadded, gt_boxes = random_crop(image_unpadded, gt_boxes, self.BOX_SCALE, round_boxes=True)
            scale_ = np.maximum(h,w)
            # scale it back to image dimension
            gt_boxes[:, [1, 3]] = gt_boxes[:, [1, 3]]*scale_
            gt_boxes[:, [0, 2]] = gt_boxes[:, [0, 2]]*scale_
        #crop boxes that are too large
        if self.is_train:
            gt_boxes[:, [1, 3]] = gt_boxes[:, [1, 3]].clip(None,h)
            gt_boxes[:, [0, 2]] = gt_boxes[:, [0, 2]].clip(None,w)



        # unique_comb, fwd_rels, inv_rels = get_obj_comb_and_rels(gt_rels, self.gt_classes[index].copy(), gt_boxes, self.use_bg_rels, self.require_overlap, self.FG_EDGE_PER_IMG, self.BG_EDGE_PER_IMG)
        # obj_rel_mat = get_obj_rels_mat(gt_rels, self.gt_classes[index])
        # gt_norm_boxes = get_normalized_boxes(gt_boxes.copy(), image_unpadded, self.BOX_SCALE)   #todo change as per im scale

        target = {}
        target["iscrowd"] = torch.zeros(gt_boxes.shape[0])
        target["boxes"] = torch.as_tensor(gt_boxes, dtype=torch.float)
        target["labels"] = torch.as_tensor(gt_classes, dtype=torch.int64)
        target["edges"] = torch.as_tensor(gt_rels, dtype=torch.int64)
        target["image_id"] = self.data_dicts[index]['id']
        target["orig_size"] = torch.as_tensor([int(h), int(w)])

        if self.transforms is not None:
            img, target = self.transforms(image_unpadded, target)

        return img, target

    def __len__(self):
        return len(self.data_dicts)


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# MISC. HELPER FUNCTIONS ~~~~~~~~~~~~~~~~~~~~~
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def get_obj_comb_and_rels(rels, gt_classes, gt_boxes, use_bg_rels, require_overlap, FG_EDGE_PER_IMG, BG_EDGE_PER_IMG):
    '''
    It create all posiible combination of rels which have some intersction or they have rels in gt
    :param rels:
    :param gt_classes:
    :param gt_boxes:
    :param use_bg_rels:
    :return:
    '''
    #create mix of bg and fg rels in sorted order unique does that
    unique_fg_edge = np.unique(np.sort(rels[:, :2], axis=1),axis=0)
    if use_bg_rels:
        all_comb =  np.arange(len(gt_classes))[:, None] != np.arange(len(gt_classes))[None]
        if require_overlap:
            all_comb = np.stack(np.nonzero(np.triu(all_comb & (box_iou_union_2d(gt_boxes, gt_boxes) > 0))), axis=-1)
            #all_comb = np.concatenate((all_comb,rels[:, 0:2]))
            #all_comb = np.concatenate((all_comb, unique_fg_edge))
            #unique_bg_edge = np.unique(all_comb, axis=0)
            #all_unique_rel = unique_bg_edge
        else:
            all_comb = np.stack(np.nonzero(np.triu(all_comb)), axis=-1)
    else:
        all_comb = unique_fg_edge
        assert 'This condition is not yet implemented'

    #now filter out the fg edge
    unique_bg_edge = []
    for edge in all_comb:
        if not (edge == unique_fg_edge).all(axis=1).any():
            unique_bg_edge.append(edge)
    #check if the number of rels exceeds the limit, then sample
    if len(unique_fg_edge)>FG_EDGE_PER_IMG:
        unique_fg_edge = np.asarray(random.sample(list(unique_fg_edge),k=FG_EDGE_PER_IMG))
    if len(unique_bg_edge)>BG_EDGE_PER_IMG and use_bg_rels:
        unique_bg_edge = np.asarray(random.sample(list(unique_bg_edge),k=BG_EDGE_PER_IMG))

    if len(unique_bg_edge)>0:
        all_comb = np.unique(np.concatenate((np.asarray(unique_bg_edge), unique_fg_edge)), axis=0)
    else:
        all_comb = unique_fg_edge

    all_comb = np.append(all_comb, np.zeros(all_comb.shape[0], dtype=np.int64)[:, None], axis=1)

    return get_fwd_inv_rels(all_comb, rels)

# def get_obj_rels_mat(gt_rels, gt_classes):
#     obj_rel_mat = np.full((len(gt_classes), len(gt_classes)), 0)
#     for rel in gt_rels:
#         obj_rel_mat[rel[0],rel[1]] = 1
#         obj_rel_mat[rel[1], rel[0]] = 1
#     return  obj_rel_mat


# def get_normalized_boxes(norm_boxes, image_unpadded, BOX_SCALE):
#     unscaled_img = np.array([BOX_SCALE / max(image_unpadded.size) * image_unpadded.size[0],
#                              BOX_SCALE / max(image_unpadded.size) * image_unpadded.size[1]])
#     norm_boxes = norm_boxes.astype(float)
#     norm_boxes[:, 0] /= unscaled_img[0]
#     norm_boxes[:, 1] /= unscaled_img[1]
#     norm_boxes[:, 2] /= unscaled_img[0]
#     norm_boxes[:, 3] /= unscaled_img[1]

#     return norm_boxes


def load_image_filenames(image_file, image_dir):
    """
    Loads the image filenames from visual genome from the JSON file that contains them.
    This matches the preprocessing in scene-graph-TF-release/data_tools/vg_to_imdb.py.
    :param image_file: JSON file. Elements contain the param "image_id".
    :param image_dir: directory where the VisualGenome images are located
    :return: List of filenames corresponding to the good images
    """
    with open(image_file, 'r') as f:
        im_data = json.load(f)

    corrupted_ims = ['1592.jpg', '1722.jpg', '4616.jpg', '4617.jpg']
    fns = []
    idx = []
    all_files = os.listdir(image_dir)
    for i, img in enumerate(im_data):
        basename = '{}.jpg'.format(img['image_id'])
        if basename in corrupted_ims:
            continue

        filename = os.path.join(image_dir, basename)
        if basename in all_files:
            fns.append(filename)
            idx.append(img['image_id'])
            # print('found',filename)
    #assert len(fns) == 108073
    return fns, idx


def load_graphs(graphs_file,
                filter_duplicate_rels=True,
                filter_empty_rels=True,
                filter_non_overlap=False,
                BOX_SCALE=1024,
                mode='train'):
    """
    Load the file containing the GT boxes and relations, as well as the dataset split
    :param graphs_file: HDF5
    :param mode: (train, val, or test)
    :param num_im: Number of images we want
    :param num_val_im: Number of validation images
    :param filter_empty_rels: (will be filtered otherwise.)
    :param filter_non_overlap: If training, filter images that dont overlap.
    :return: image_index: numpy array corresponding to the index of images we're using
             boxes: List where each element is a [num_gt, 4] array of ground 
                    truth boxes (x1, y1, x2, y2)
             gt_classes: List where each element is a [num_gt] array of classes
             relationships: List where each element is a [num_r, 3] array of 
                    (box_ind_1, box_ind_2, predicate) relationships
    """

    roi_h5 = h5py.File(graphs_file, 'r')
    data_split = roi_h5['split'][:]
    split = 2 if mode == 'test' else 0
    split_mask = data_split == split

    # Filter out images without bounding boxes
    split_mask &= roi_h5['img_to_first_box'][:] >= 0
    if filter_empty_rels:
        split_mask &= roi_h5['img_to_first_rel'][:] >= 0

    image_index = np.where(split_mask)[0]

    split_mask = np.zeros_like(data_split).astype(bool)
    split_mask[image_index] = True

    # Get box information
    all_labels = roi_h5['labels'][:, 0]
    all_boxes = roi_h5['boxes_{}'.format(BOX_SCALE)][:]  # will index later
    assert np.all(all_boxes[:, :2] >= 0)  # sanity check
    assert np.all(all_boxes[:, 2:] > 0)  # no empty box

    # convert from xc, yc, w, h to x1, y1, x2, y2
    all_boxes[:, :2] = all_boxes[:, :2] - all_boxes[:, 2:] / 2
    all_boxes[:, 2:] = all_boxes[:, :2] + all_boxes[:, 2:]

    im_to_first_box = roi_h5['img_to_first_box'][split_mask]
    im_to_last_box = roi_h5['img_to_last_box'][split_mask]
    im_to_first_rel = roi_h5['img_to_first_rel'][split_mask]
    im_to_last_rel = roi_h5['img_to_last_rel'][split_mask]

    # load relation labels
    _relations = roi_h5['relationships'][:]
    _relation_predicates = roi_h5['predicates'][:, 0]
    assert (im_to_first_rel.shape[0] == im_to_last_rel.shape[0])
    assert (_relations.shape[0] == _relation_predicates.shape[0])  # sanity check

    # Get everything by image.
    boxes = []
    gt_classes = []
    relationships = []
    for i in range(len(image_index)):
        boxes_i = all_boxes[im_to_first_box[i]:im_to_last_box[i] + 1, :]
        gt_classes_i = all_labels[im_to_first_box[i]:im_to_last_box[i] + 1]

        if im_to_first_rel[i] >= 0:
            predicates = _relation_predicates[im_to_first_rel[i]:im_to_last_rel[i] + 1]
            obj_idx = _relations[im_to_first_rel[i]:im_to_last_rel[i] + 1] - im_to_first_box[i]
            assert np.all(obj_idx >= 0)
            assert np.all(obj_idx < boxes_i.shape[0])
            rels = np.column_stack((obj_idx, predicates))
        else:
            assert not filter_empty_rels
            rels = np.zeros((0, 3), dtype=np.int32)

        if filter_duplicate_rels:
            # Filter out dupes!
            all_rel_sets = defaultdict(list)
            for (o0, o1, r) in rels:
                all_rel_sets[(o0, o1)].append(r)
            rels = [(k[0], k[1], np.random.choice(v)) for k,v in all_rel_sets.items()]
            rels = np.array(rels)

        if filter_non_overlap:
            inters = box_iou_union_2d(boxes_i, boxes_i)  # TODO: check whether this option is necessary
            rel_overs = inters[rels[:, 0], rels[:, 1]]
            inc = np.where(rel_overs > 0.0)[0]

            if inc.size > 0:
                rels = rels[inc]
            else:
                split_mask[image_index[i]] = 0
                continue
        boxes.append(boxes_i)
        gt_classes.append(gt_classes_i)
        relationships.append(rels)

    return split_mask, boxes, gt_classes, relationships


def load_info(dir_name, filter_non_overlap=False,
              filter_empty_rels=True,
              filter_duplicate_rels=True, ):
    """
    Loads the file containing the visual genome label meanings
    :param info_file: JSON
    :return: ind_to_classes: sorted list of classes
             ind_to_predicates: sorted list of predicates
    """
    if os.path.exists(os.path.join('data','VG_14.pickle')):
        vg_ds = pickle.load(open(os.path.join('data','VG_14.pickle'),'rb'))
        data_dicts = vg_ds['data_dicts']
        class_to_ind = vg_ds['obj_label_map']
        predicate_to_ind =  vg_ds['rel_label_map']
    else:
        all_frames = json.load(open(os.path.join(dir_name, 'objects.json'), 'r'))
        all_rels = json.load(open(os.path.join(dir_name, 'relationships.json'), 'r'))
        dict_info = json.load(open(os.path.join(dir_name, 'VG-SGG-dicts.json'), 'r'))
        class_to_ind = dict_info['label_to_idx']
        predicate_to_ind = dict_info['predicate_to_idx']
        # Get everything from image.
        data_dicts = []
        mismatched_rels = 0
        total_rels = 0
        for i, (each_frame, each_frame_rels) in enumerate(zip(all_frames,all_rels)):
            assert each_frame['image_id'] == each_frame_rels['image_id']
            # if 'image_url' in each_frame.keys():
            #     image_path = each_frame['image_url']
            #     if  'VG_100K_2' in image_path:#todo temporary
            #         continue
            image_name='data/VG_100K/'+str(each_frame['image_id'])+'.jpg'
            relationships = each_frame_rels['relationships']

            label = []
            boxes = []
            rels = []
            obj_id_2_pos = []

            for object in each_frame['objects']: #iterate over objects
                name = object['names'][0]
                if name in class_to_ind:
                    x = object['x']
                    y = object['y']
                    w = object['w']
                    h = object['h']
                    boxes.append([x,y,x+w,y+h])
                    label.append(class_to_ind[name])
                    obj_id_2_pos.append(object['object_id'])

            for framewise_rels in relationships: #iterate over relations
                total_rels +=1
                predicate = framewise_rels['predicate']
                subject = framewise_rels['subject']['object_id']
                object = framewise_rels['object']['object_id']
                if subject in obj_id_2_pos and object in obj_id_2_pos and subject!=object: # filter self and wrong rels
                    if filter_duplicate_rels:
                        if predicate in predicate_to_ind:
                            sub_rel_pos = obj_id_2_pos.index(subject)
                            obj_rel_pos = obj_id_2_pos.index(object)
                            if len(rels)==0:
                                rels.append([sub_rel_pos, obj_rel_pos, predicate_to_ind[predicate]])
                            else:
                                if not [sub_rel_pos,obj_rel_pos] in np.asarray(rels)[:,:2].tolist():
                                    rels.append([sub_rel_pos, obj_rel_pos, predicate_to_ind[predicate]])
                        else:
                            mismatched_rels +=1
            #append to data dicts
            if filter_empty_rels :
                if len(rels)>0:
                    data_dicts.append({'id':each_frame['image_id'], "image": image_name, "boxes": np.asarray(boxes), "label":label, "edges": np.asarray(rels)})

        print('Number of mismatched relations : ',mismatched_rels/total_rels)
    vg_14 = {}
    vg_14['data_dicts']=data_dicts
    vg_14['obj_label_map'] = class_to_ind
    vg_14['rel_label_map'] = predicate_to_ind
    pickle_out = open("data/VG_14.pickle", "wb")
    pickle.dump(vg_14, pickle_out)
    pickle_out.close()

        # #info['predicate_to_idx']['__background__'] = 0
        # class_to_ind = info['label_to_idx']
        # predicate_to_ind = info['predicate_to_idx']
        # ind_to_classes = sorted(class_to_ind, key=lambda k: class_to_ind[k])
        # ind_to_predicates = sorted(predicate_to_ind, key=lambda k: predicate_to_ind[k])

    return data_dicts , class_to_ind, predicate_to_ind


def get_fwd_inv_rels(all_comb, rels, val=False):
    def stack_and_cat(to_rels, indices, obj1, obj2):
        """
        concatinate with previous relation if they exists, else make relation for background
        """
        if len(indices) > 0:
            if to_rels is not None:
                rel_temp = np.column_stack((np.full((len(indices)), i), rels[indices]))
                to_rels = np.concatenate([to_rels, rel_temp], axis=0)
            else:
                to_rels = np.column_stack((np.full((len(indices)), i), rels[indices]))
        else:
            if not val:  #for validation and test only gt rels
                if to_rels is not None:
                    rel_temp = np.column_stack(( i, obj1, obj2, 0))
                    to_rels = np.concatenate([to_rels, rel_temp], axis=0)
                else:
                    to_rels = np.column_stack(( i, obj1, obj2, 0))

        return to_rels
        # rearrange relation as per object combination order, format [pos of unq comb, rel number]

    #split to fwd and inv relation and add bg relation
    fwd_rels = None
    inv_rels = None
    if all_comb.shape[1]<3:
        all_comb = np.concatenate((all_comb,np.zeros((all_comb.shape[0],1),dtype=int)),1)
    for i, each_comb in enumerate(all_comb):
        fwd_indices = np.where((rels[:, :2] == each_comb[:2]).all(axis=1))[0]
        fwd_rels = stack_and_cat(fwd_rels, fwd_indices, each_comb[0], each_comb[1])

        inv_indices = np.where((rels[:, :2] == each_comb[:2][[1, 0]]).all(axis=1))[0]
        inv_rels = stack_and_cat(inv_rels, inv_indices, each_comb[1], each_comb[0])

        if len(fwd_indices) >0 or len(inv_indices)>0:
            each_comb[2]=1

    return all_comb, fwd_rels, inv_rels

def get_dataset(config,filter_non_overlap, filter_empty_rels, filter_duplicate_rels,split='train'):
    # get VG data
    split_mask, gt_boxes, gt_classes, relationships = load_graphs(
        graphs_file=config.DATA.VG_SGG_FN,
        BOX_SCALE=config.DATA.BOX_SCALE,
        filter_non_overlap=filter_non_overlap,
        filter_empty_rels=filter_empty_rels,
        filter_duplicate_rels=filter_duplicate_rels,mode=split
    )
    filenames, idx = load_image_filenames(config.DATA.IM_DATA_FN, image_dir=config.DATA.VG_IMAGES)
    filenames = [filenames[i] for i in np.where(split_mask)[0]]
    idx = [idx[i] for i in np.where(split_mask)[0]]

    # print(len(filenames), split_mask.shape, len(gt_boxes), len(gt_classes), len(relationships))
    data_dicts = [
        {"id": ids, "image": img, "boxes": boxes, "label": labels, "edges": edges} for
        ids, img, boxes, labels, edges in zip(idx, filenames, gt_boxes, gt_classes, relationships)
    ]

    return data_dicts


def build_scene_graph_data(config,
                            mode='train',
                            split=0.8,
                            num_val_im=5000,
                            filter_empty_rels=True,
                            filter_duplicate_rels=True,
                            filter_non_overlap=False,
                            debug=False,
                            ):

    normalize = T.Compose([
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    scales = [480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800]

    train_transform = T.Compose([
            T.RandomHorizontalFlip(),
            T.RandomSelect(
                T.RandomResize(scales, max_size=800),
                #T.FixedResize(config.DATA.IM_SCALE, max),
                T.Compose([
                    T.RandomResize(scales),
                    #T.RandomResize([400, 500, 600]),
                    T.RandomSizeCrop(384, 600),
                    T.RandomResize(scales, max_size=800),
                    #T.FixedResize(config.DATA.IM_SCALE, max),
                ])
            ),
            normalize,
        ])
    val_transform = T.Compose([
            T.RandomResize([800], max_size=800),
            #T.FixedResize(config.DATA.IM_SCALE, max),
            normalize,
        ])
    if config.DATA.DATASET=='VG':
        data_dicts = get_dataset(config,filter_non_overlap=filter_non_overlap,filter_empty_rels=filter_empty_rels,
                                                                    filter_duplicate_rels=filter_duplicate_rels)
        if config.DATA.VG_TEST:
            test_data_dicts = get_dataset(config,filter_non_overlap=filter_non_overlap,filter_empty_rels=filter_empty_rels,
                                                                    filter_duplicate_rels=filter_duplicate_rels,split='test')
    elif config.DATA.DATASET=='OI':#for open image V6
        train_ds = OIDataset(split='train',img_dir='data/open-imagev6/images',
                      ann_file='data/open-imagev6/annotations/vrd-train-anno.json',
                      cate_info_file='data/open-imagev6/annotations/categories_dict.json',
                             transforms=train_transform,DEBUG=debug,config=config)
        val_ds = OIDataset(split='val', img_dir='data/open-imagev6/images',
                             ann_file='data/open-imagev6/annotations/vrd-val-anno.json',
                             cate_info_file='data/open-imagev6/annotations/categories_dict.json',
                           transforms=val_transform,DEBUG=debug,config=config)
        return train_ds, val_ds
    else:
        data_dicts, obj_label_map,rel_label_map = load_info(
            dir_name=os.path.dirname(config.DATA.VG_SGG_FN),
            #BOX_SCALE=config.DATA.BOX_SCALE,
            filter_non_overlap=filter_non_overlap,
            filter_empty_rels=filter_empty_rels,
            filter_duplicate_rels=filter_duplicate_rels,
        )
    if config.TRAIN.FOCAL_LOSS_ALPHA:    # for focall loss class staets wd 0 as noo extra classes
        gt_classes = list(map(lambda x: x - 1, gt_classes))
    if mode=='train':
        ds = visual_genome_loader(data_dicts, train_transform, config)
        return ds
    elif mode=='val':
        ds = visual_genome_loader(data_dicts, val_transform, config)
        return ds
    elif mode=='split':
        random.seed(config.DATA.SEED)

        if not config.DATA.VG_TEST: # load full training and full test set
            train_files, val_files = data_dicts, test_data_dicts
        else: # load train and validation
            train_files, val_files = data_dicts[num_val_im:], data_dicts[:num_val_im]
        random.shuffle(train_files) #random shuffle training
        if debug:  # todo donly for debugging, remove in final version
            train_files = train_files[:30]
            val_files = val_files[:30]
        train_ds = visual_genome_loader(train_files, train_transform, config,is_train=True)
        val_ds = visual_genome_loader(val_files, val_transform, config)
        return train_ds, val_ds
