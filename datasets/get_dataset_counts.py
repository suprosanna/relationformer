"""
Get counts of all of the examples in the dataset. Used for creating the baseline
dictionary model
"""

import numpy as np
from metrics.box_ops_np import intersect_2d, box_iou_2d_np as bbox_overlaps

def get_counts(train_data, must_overlap=True):
    """
    Get counts of all of the relations. Used for modeling directly P(rel | o1, o2)
    :param train_data:
    :param must_overlap:
    :return:
    """
    fg_matrix = np.zeros((
        train_data.num_classes,
        train_data.num_classes,
        train_data.num_predicates,
    ), dtype=np.int64)

    bg_matrix = np.zeros((
        train_data.num_classes,
        train_data.num_classes,
    ), dtype=np.int64)

    for sample in train_data.data_dicts:
        gt_classes = sample['label'].copy()
        gt_relations = sample['edges'].copy()
        gt_boxes = sample['boxes'].copy().copy()

        # For the foreground, we'll just look at everything
        o1o2 = gt_classes[gt_relations[:, :2]]
        for (o1, o2), gtr in zip(o1o2, gt_relations[:,2]):
            fg_matrix[o1, o2, gtr] += 1

            # For the background, get all of the things that overlap.
        o1o2_total = gt_classes[np.array(
            box_filter(gt_boxes, must_overlap=must_overlap), dtype=int)]
        mask = intersect_2d(o1o2_total, o1o2).any(1)
        index = np.where(mask)[0]
        o1o2_bg = o1o2_total[index]
        for (o1, o2) in o1o2_bg:
            bg_matrix[o1, o2] += 1

    return fg_matrix, bg_matrix


def box_filter(boxes, must_overlap=False):
    """ Only include boxes that overlap as possible relations.
    If no overlapping boxes, use all of them."""
    n_cands = boxes.shape[0]

    overlaps = bbox_overlaps(boxes.astype(np.float), boxes.astype(np.float)) > 0
    np.fill_diagonal(overlaps, 0)

    all_possib = np.ones_like(overlaps, dtype=np.bool)
    np.fill_diagonal(all_possib, 0)

    if must_overlap:
        possible_boxes = np.column_stack(np.where(overlaps))

        if possible_boxes.size == 0:
            possible_boxes = np.column_stack(np.where(all_possib))
    else:
        possible_boxes = np.column_stack(np.where(all_possib))
    return possible_boxes

if __name__ == '__main__':
    fg, bg = get_counts(must_overlap=False)
