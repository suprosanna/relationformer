from functools import partial
import pdb
import numpy as np


def box_iou_2d_np(boxes1, boxes2):
    area1 = box_area_2d_np(boxes1)
    area2 = box_area_2d_np(boxes2)

    x1 = np.maximum(boxes1[:, None, 0], boxes2[:, 0])  # [N, M]
    y1 = np.maximum(boxes1[:, None, 1], boxes2[:, 1])  # [N, M]
    x2 = np.minimum(boxes1[:, None, 2], boxes2[:, 2])  # [N, M]
    y2 = np.minimum(boxes1[:, None, 3], boxes2[:, 3])  # [N, M]

    inter = np.clip((x2 - x1), a_min=0, a_max=None) * np.clip((y2 - y1), a_min=0, a_max=None)  # [N, M]

    with np.errstate(invalid='raise'):
        try:
            return inter / (area1[:, None] + area2 - inter + 1e-8)
        except RuntimeWarning as e:
            print(e)


def box_area_2d_np(boxes):
    return (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])

class BBoxEvaluator:
    def __init__(
        self,
        classes,
        iou_fn=box_iou_2d_np,
        max_detections=100
    ):
        """
        Class for evaluate detection metrics
        Args:
            metrics (Sequence[DetectionMetric]: detection metrics to evaluate
            iou_fn (Callable[[np.ndarray, np.ndarray], np.ndarray]): compute overlap for each pair
            max_detections (int): number of maximum detections per image (reduces computation)
        """
        self.iou_fn = iou_fn
        self.max_detections = max_detections

        self.results_list = []  # store results of each image

        self.metrics = [
            Metric(
                classes=classes,
                iou_list=np.arange(0.1, 1.0, 0.1),  # for individual APs
                iou_range=(0.5, 0.95, 0.05), # for mAP - different from coco (0.5, 0.95, 0.05)
                per_class=True,
                max_detection=(100, ) # different from nndet (100, )
            )
        ]

        self.iou_thresholds = self.get_unique_iou_thresholds()
        self.iou_mapping = self.get_indices_of_iou_for_each_metric()

    def get_unique_iou_thresholds(self):
        """
        Compute unique set of iou thresholds
        """
        iou_thresholds = [_i for i in self.metrics for _i in i.get_iou_thresholds()]
        iou_thresholds = list(set(iou_thresholds))
        iou_thresholds.sort()
        return iou_thresholds

    def get_indices_of_iou_for_each_metric(self):
        """
        Find indices of iou thresholds for each metric
        """
        return [[self.iou_thresholds.index(th) for th in m.get_iou_thresholds()]
                for m in self.metrics]

    def add(
        self,
        pred_boxes,
        pred_classes,
        pred_scores,
        gt_boxes,
        gt_classes,
        gt_ignore=None
    ):
        """
        Preprocess batch results for final evaluation
        Args:
            pred_boxes (Sequence[np.ndarray]): predicted boxes from single batch; List[[D, dim * 2]], D number of
                predictions
            pred_classes (Sequence[np.ndarray]): predicted classes from a single batch; List[[D]], D number of
                predictions
            pred_scores (Sequence[np.ndarray]): predicted score for each bounding box; List[[D]], D number of
                predictions
            gt_boxes (Sequence[np.ndarray]): ground truth boxes; List[[G, dim * 2]], G number of ground truth
            gt_classes (Sequence[np.ndarray]): ground truth classes; List[[G]], G number of ground truth
            gt_ignore (Sequence[Sequence[bool]]): specified if which ground truth boxes are not counted as true
                positives (detections which match theses boxes are not counted as false positives either);
                List[[G]], G number of ground truth
        Returns
            dict: empty dict... detection metrics can only be evaluated at the end
        """
        # reduce class ids by 1 to start with 0
        gt_classes = [batch_elem_classes -1 for batch_elem_classes in gt_classes]
        pred_classes = [batch_elem_classes -1 for batch_elem_classes in pred_classes]

        if gt_ignore is None:   # only zeros -> don't ignore anything
            n = [0 if gt_boxes_img.size == 0 else gt_boxes_img.shape[0] for gt_boxes_img in gt_boxes]
            gt_ignore = [np.zeros(_n).reshape(-1) for _n in n]

        self.results_list.extend(matching_batch(
            self.iou_fn, self.iou_thresholds, pred_boxes=pred_boxes, pred_classes=pred_classes,
            pred_scores=pred_scores, gt_boxes=gt_boxes, gt_classes=gt_classes, gt_ignore=gt_ignore,
            max_detections=self.max_detections))

        return {}

    def eval(self):
        """
        Accumulate results of individual batches and compute final metrics
        Returns:
            Dict[str, float]: dictionary with scalar values for evaluation
            Dict[str, np.ndarray]: dictionary with arrays, e.g. for visualization of graphs
        """
        metric_scores = {}
        metric_curves = {}
        for metric_idx, metric in enumerate(self.metrics):
            _filter = partial(self.iou_filter, iou_idx=self.iou_mapping[metric_idx])
            iou_filtered_results = list(map(_filter, self.results_list))    # no filtering
            
            score, curve = metric(iou_filtered_results)
            
            if score is not None:
                metric_scores.update(score)
            
            if curve is not None:
                metric_curves.update(curve)
        return metric_scores

    @staticmethod
    def iou_filter(image_dict, iou_idx,
                   filter_keys=('dtMatches', 'gtMatches', 'dtIgnore')):
        """
        This functions can be used to filter specific IoU values from the results
        to make sure that the correct IoUs are passed to metric
        
        Parameters
        ----------
        image_dict : dict
            dictionary containin :param:`filter_keys` which contains IoUs in the first dimension
        iou_idx : List[int]
            indices of IoU values to filter from keys
        filter_keys : tuple, optional
            keys to filter, by default ('dtMatches', 'gtMatches', 'dtIgnore')
        
        Returns
        -------
        dict
            filtered dictionary
        """
        iou_idx = list(iou_idx)
        filtered = {}
        for cls_key, cls_item in image_dict.items():
            filtered[cls_key] = {key: item[iou_idx] if key in filter_keys else item
                                 for key, item in cls_item.items()}
        return filtered

    def reset(self):
        """
        Reset internal state of evaluator
        """
        self.results_list = []


def matching_batch(
    iou_fn, 
    iou_thresholds, 
    pred_boxes,
    pred_classes, 
    pred_scores,
    gt_boxes, 
    gt_classes,
    gt_ignore,
    max_detections
):
    """
    Match boxes of a batch to corresponding ground truth for each category
    independently.
    Args:
        iou_fn: compute overlap for each pair
        iou_thresholds: defined which IoU thresholds should be evaluated
        pred_boxes: predicted boxes from single batch; List[[D, dim * 2]],
            D number of predictions
        pred_classes: predicted classes from a single batch; List[[D]],
            D number of predictions
        pred_scores: predicted score for each bounding box; List[[D]],
            D number of predictions
        gt_boxes: ground truth boxes; List[[G, dim * 2]], G number of ground
            truth
        gt_classes: ground truth classes; List[[G]], G number of ground truth
        gt_ignore: specified if which ground truth boxes are not counted as
            true positives
            (detections which match theses boxes are not counted as false
            positives either); List[[G]], G number of ground truth
        max_detections: maximum number of detections which should be evaluated
    Returns:
        List[Dict[int, Dict[str, np.ndarray]]]
            matched detections [dtMatches] and ground truth [gtMatches]
            boxes [str, np.ndarray] for each category (stored in dict keys)
            for each image (list)
    """
    results = []
    # iterate over images/batches
    for pboxes, pclasses, pscores, gboxes, gclasses, gignore in zip(
        pred_boxes, pred_classes, pred_scores, gt_boxes, gt_classes, gt_ignore
    ):
        img_classes = np.union1d(pclasses, gclasses)
        result = {}  # dict contains results for each class in one image
        for c in img_classes:
            pred_mask = pclasses == c # mask predictions with current class
            gt_mask = gclasses == c # mask ground trtuh with current class
            try:
                if not np.any(gt_mask): # no ground truth
                    result[c] = _matching_no_gt(
                        iou_thresholds=iou_thresholds,
                        pred_scores=pscores[pred_mask],
                        max_detections=max_detections)
                elif pred_mask.shape[0]==0: # no predictions
                    result[c] = _matching_no_pred(
                        iou_thresholds=iou_thresholds,
                        gt_ignore=gignore[gt_mask],
                    )
                else: # at least one prediction and one ground truth
                    result[c] = _matching_single_image_single_class(
                        iou_fn=iou_fn,
                        pred_boxes=pboxes[pred_mask],
                        pred_scores=pscores[pred_mask],
                        gt_boxes=gboxes[gt_mask],
                        gt_ignore=gignore[gt_mask],
                        max_detections=max_detections,
                        iou_thresholds=iou_thresholds,
                    )
            except:
                pdb.set_trace()
        results.append(result)
    return results


def _matching_no_gt(
    iou_thresholds,
    pred_scores,
    max_detections,
):
    """
    Matching result with not ground truth in image
    Args:
        iou_thresholds: defined which IoU thresholds should be evaluated
        dt_scores: predicted scores
        max_detections: maximum number of allowed detections per image.
            This functions uses this parameter to stay consistent with
            the actual matching function which needs this limit.
    Returns:
        dict: computed matching
            `dtMatches`: matched detections [T, D], where T = number of
                thresholds, D = number of detections
            `gtMatches`: matched ground truth boxes [T, G], where T = number
                of thresholds, G = number of ground truth
            `dtScores`: prediction scores [D] detection scores
            `gtIgnore`: ground truth boxes which should be ignored
                [G] indicate whether ground truth should be ignored
            `dtIgnore`: detections which should be ignored [T, D],
                indicate which detections should be ignored
    """
    dt_ind = np.argsort(-pred_scores, kind='mergesort')
    dt_ind = dt_ind[:max_detections]
    dt_scores = pred_scores[dt_ind]

    num_preds = len(dt_scores)

    gt_match = np.array([[]] * len(iou_thresholds))
    dt_match = np.zeros((len(iou_thresholds), num_preds))
    dt_ignore = np.zeros((len(iou_thresholds), num_preds))

    return {
        'dtMatches': dt_match,  # [T, D], where T = number of thresholds, D = number of detections
        'gtMatches': gt_match,  # [T, G], where T = number of thresholds, G = number of ground truth
        'dtScores': dt_scores,  # [D] detection scores
        'gtIgnore': np.array([]).reshape(-1),  # [G] indicate whether ground truth should be ignored
        'dtIgnore': dt_ignore,  # [T, D], indicate which detections should be ignored
    }


def _matching_no_pred(
    iou_thresholds,
    gt_ignore,
):
    """
    Matching result with no predictions
    Args:
        iou_thresholds: defined which IoU thresholds should be evaluated
        gt_ignore: specified if which ground truth boxes are not counted as
            true positives (detections which match theses boxes are not
            counted as false positives either); [G], G number of ground truth
    Returns:
        dict: computed matching
            `dtMatches`: matched detections [T, D], where T = number of
                thresholds, D = number of detections
            `gtMatches`: matched ground truth boxes [T, G], where T = number
                of thresholds, G = number of ground truth
            `dtScores`: prediction scores [D] detection scores
            `gtIgnore`: ground truth boxes which should be ignored
                [G] indicate whether ground truth should be ignored
            `dtIgnore`: detections which should be ignored [T, D],
                indicate which detections should be ignored
    """
    dt_scores = np.array([])
    dt_match = np.array([[]] * len(iou_thresholds))
    dt_ignore = np.array([[]] * len(iou_thresholds))

    n_gt = 0 if gt_ignore.size == 0 else gt_ignore.shape[0]
    gt_match = np.zeros((len(iou_thresholds), n_gt))

    return {
        'dtMatches': dt_match,  # [T, D], where T = number of thresholds, D = number of detections
        'gtMatches': gt_match,  # [T, G], where T = number of thresholds, G = number of ground truth
        'dtScores': dt_scores,  # [D] detection scores
        'gtIgnore': gt_ignore.reshape(-1),  # [G] indicate whether ground truth should be ignored
        'dtIgnore': dt_ignore,  # [T, D], indicate which detections should be ignored
    }


def _matching_single_image_single_class(
    iou_fn,
    pred_boxes,
    pred_scores,
    gt_boxes,
    gt_ignore,
    max_detections,
    iou_thresholds,    
):
    """
    Adapted from https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocotools/cocoeval.py
    Args:
        iou_fn: compute overlap for each pair
        iou_thresholds: defined which IoU thresholds should be evaluated
        pred_boxes: predicted boxes from single batch; [D, dim * 2], D number
            of predictions
        pred_scores: predicted score for each bounding box; [D], D number of
            predictions
        gt_boxes: ground truth boxes; [G, dim * 2], G number of ground truth
        gt_ignore: specified if which ground truth boxes are not counted as
            true positives (detections which match theses boxes are not
            counted as false positives either); [G], G number of ground truth
        max_detections: maximum number of detections which should be evaluated
    Returns:
        dict: computed matching
            `dtMatches`: matched detections [T, D], where T = number of
                thresholds, D = number of detections
            `gtMatches`: matched ground truth boxes [T, G], where T = number
                of thresholds, G = number of ground truth
            `dtScores`: prediction scores [D] detection scores
            `gtIgnore`: ground truth boxes which should be ignored
                [G] indicate whether ground truth should be ignored
            `dtIgnore`: detections which should be ignored [T, D],
                indicate which detections should be ignored
    """
    # filter for max_detections highest scoring predictions to speed up computation
    dt_ind = np.argsort(-pred_scores, kind='mergesort')
    dt_ind = dt_ind[:max_detections]    # only take up to max number of detections

    pred_boxes = pred_boxes[dt_ind] # sort by highest score
    pred_scores = pred_scores[dt_ind]

    # sort ignored ground truth to last positions
    gt_ind = np.argsort(gt_ignore, kind='mergesort')
    gt_boxes = gt_boxes[gt_ind]
    gt_ignore = gt_ignore[gt_ind]

    # ious between sorted(!) predictions and ground truth
    ious = iou_fn(pred_boxes, gt_boxes)

    num_preds, num_gts = ious.shape[0], ious.shape[1]
    gt_match = np.zeros((len(iou_thresholds), num_gts))
    dt_match = np.zeros((len(iou_thresholds), num_preds))
    dt_ignore = np.zeros((len(iou_thresholds), num_preds))

    for tind, t in enumerate(iou_thresholds):
        for dind, _d in enumerate(pred_boxes):  # iterate detections starting from highest scoring one
            # information about best match so far (m=-1 -> unmatched)
            iou = min([t, 1-1e-10]) # iou threshold
            m = -1

            for gind, _g in enumerate(gt_boxes):  # iterate ground truth
                # if this gt already matched, continue (no duplicate detections)
                if gt_match[tind, gind] > 0:
                    continue

                # if dt matched to reg gt, and on ignore gt, stop
                if m > -1 and gt_ignore[m] == 0 and gt_ignore[gind] == 1:
                    break

                # continue to next gt unless better match made
                if ious[dind, gind] < iou:
                    continue

                # if match successful and best so far, store appropriately
                iou = ious[dind, gind]
                m = gind

            # if match made, store id of match for both dt and gt
            if m == -1:
                continue
            else:
                dt_ignore[tind, dind] = int(gt_ignore[m])
                dt_match[tind, dind] = 1
                gt_match[tind, m] = 1

    # store results for given image and category
    return {
            'dtMatches': dt_match,  # [T, D], where T = number of thresholds, D = number of detections
            'gtMatches': gt_match,  # [T, G], where T = number of thresholds, G = number of ground truth
            'dtScores': pred_scores,  # [D] detection scores
            'gtIgnore': gt_ignore.reshape(-1),  # [G] indicate whether ground truth should be ignored
            'dtIgnore': dt_ignore,  # [T, D], indicate which detections should be ignored
        }


class Metric:
    def __init__(
        self,
        classes,
        iou_list=(0.1, 0.5, 0.75),
        iou_range=(0.1, 0.5, 0.05),
        max_detection=(1, 5, 100),
        per_class=True
    ):
        """
        Class to compute COCO metrics
        Metrics computed:
            mAP over the IoU range specified by :param:`iou_range` at last value of :param:`max_detection`
            AP values at IoU thresholds specified by :param:`iou_list` at last value of :param:`max_detection`
            AR over max detections thresholds defined by :param:`max_detection` (over iou range)
        Args:
            classes (Sequence[str]): name of each class (index needs to correspond to predicted class indices!)
            iou_list (Sequence[float]): specific thresholds where ap is evaluated and saved
            iou_range (Sequence[float]): (start, stop, step) for mAP iou thresholds
            max_detection (Sequence[int]): maximum number of detections per image
        """
        self.classes = classes
        self.per_class = per_class

        iou_list = np.array(iou_list)
        _iou_range = np.linspace(
            iou_range[0], iou_range[1], int(np.round((iou_range[1] - iou_range[0]) / iou_range[2])) + 1, endpoint=True
        )
        self.iou_thresholds = np.union1d(iou_list, _iou_range)
        self.iou_range = iou_range

        # get indices of iou values of ious range and ious list for later evaluation
        self.iou_list_idx = np.nonzero(iou_list[:, np.newaxis] == self.iou_thresholds[np.newaxis])[1]
        self.iou_range_idx = np.nonzero(_iou_range[:, np.newaxis] == self.iou_thresholds[np.newaxis])[1]

        assert (self.iou_thresholds[self.iou_list_idx] == iou_list).all()
        assert (self.iou_thresholds[self.iou_range_idx] == _iou_range).all()

        self.recall_thresholds = np.linspace(.0, 1.00, int(np.round((1.00 - .0) / .01)) + 1, endpoint=True)
        self.max_detections = max_detection

    def __call__(self, *args, **kwargs):
        return self.compute(*args, **kwargs)

    def get_iou_thresholds(self):
        """
        Return IoU thresholds needed for this metric in an numpy array
        Returns:
            Sequence[float]: IoU thresholds [M], M is the number of thresholds
        """
        return self.iou_thresholds

    def compute(
        self,
        results_list
    ):
        """
        Compute COCO metrics
        Args:
            results_list (List[Dict[int, Dict[str, np.ndarray]]]): list with result s per image (in list)
                per category (dict). Inner Dict contains multiple results obtained by :func:`box_matching_batch`.
                `dtMatches`: matched detections [T, D], where T = number of thresholds, D = number of detections
                `gtMatches`: matched ground truth boxes [T, G], where T = number of thresholds, G = number of 
                    ground truth
                `dtScores`: prediction scores [D] detection scores
                `gtIgnore`: ground truth boxes which should be ignored [G] indicate whether ground truth
                    should be ignored
                `dtIgnore`: detections which should be ignored [T, D], indicate which detections should be ignored
        Returns:
            Dict[str, float]: dictionary with coco metrics
            Dict[str, np.ndarray]: None
        """
        dataset_statistics = self.compute_statistics(results_list=results_list)

        results = {}
        results.update(self.compute_ap(dataset_statistics))
        results.update(self.compute_ar(dataset_statistics))

        return results, None

    def compute_ap(self, dataset_statistics):
        """
        Compute AP metrics
        Args:
            results_list (List[Dict[int, Dict[str, np.ndarray]]]): list with result s per image (in list)
                per category (dict). Inner Dict contains multiple results obtained by :func:`box_matching_batch`.
                `dtMatches`: matched detections [T, D], where T = number of thresholds, D = number of detections
                `gtMatches`: matched ground truth boxes [T, G], where T = number of thresholds, G = number of
                    ground truth
                `dtScores`: prediction scores [D] detection scores
                `gtIgnore`: ground truth boxes which should be ignored [G] indicate whether ground truth
                    should be ignored
                `dtIgnore`: detections which should be ignored [T, D], indicate which detections should be ignored
        """
        results = {}
        if self.iou_range:  # mAP
            key = (f"mAP_IoU_{self.iou_range[0]:.2f}_{self.iou_range[1]:.2f}_{self.iou_range[2]:.2f}_"
                   f"MaxDet_{self.max_detections[-1]}")
            results[key] = self.select_ap(dataset_statistics, iou_idx=self.iou_range_idx, max_det_idx=-1)

            if self.per_class:
                for cls_idx, cls_str in enumerate(self.classes):  # per class results
                    key = (f"{cls_str}_"
                           f"mAP_IoU_{self.iou_range[0]:.2f}_{self.iou_range[1]:.2f}_{self.iou_range[2]:.2f}_"
                           f"MaxDet_{self.max_detections[-1]}")
                    results[key] = self.select_ap(dataset_statistics, iou_idx=self.iou_range_idx,
                                                  cls_idx=cls_idx, max_det_idx=-1)

        for idx in self.iou_list_idx:   # AP@IoU
            key = f"AP_IoU_{self.iou_thresholds[idx]:.2f}_MaxDet_{self.max_detections[-1]}"
            results[key] = self.select_ap(dataset_statistics, iou_idx=[idx], max_det_idx=-1)

            if self.per_class:
                for cls_idx, cls_str in enumerate(self.classes):  # per class results
                    key = (f"{cls_str}_"
                           f"AP_IoU_{self.iou_thresholds[idx]:.2f}_"
                           f"MaxDet_{self.max_detections[-1]}")
                    results[key] = self.select_ap(dataset_statistics,
                                                  iou_idx=[idx], cls_idx=cls_idx, max_det_idx=-1)
        return results

    def compute_ar(self, dataset_statistics):
        """
        Compute AR metrics
        Args:
            results_list (List[Dict[int, Dict[str, np.ndarray]]]): list with result s per image (in list)
                per category (dict). Inner Dict contains multiple results obtained by :func:`box_matching_batch`.
                `dtMatches`: matched detections [T, D], where T = number of thresholds, D = number of detections
                `gtMatches`: matched ground truth boxes [T, G], where T = number of thresholds, G = number of
                    ground truth
                `dtScores`: prediction scores [D] detection scores
                `gtIgnore`: ground truth boxes which should be ignored [G] indicate whether ground truth
                    should be ignored
                `dtIgnore`: detections which should be ignored [T, D], indicate which detections should be ignored
        """
        results = {}
        for max_det_idx, max_det in enumerate(self.max_detections):  # mAR
            key = f"mAR_IoU_{self.iou_range[0]:.2f}_{self.iou_range[1]:.2f}_{self.iou_range[2]:.2f}_MaxDet_{max_det}"
            results[key] = self.select_ar(dataset_statistics, max_det_idx=max_det_idx, iou_idx=self.iou_range_idx)

            if self.per_class:
                for cls_idx, cls_str in enumerate(self.classes):  # per class results
                    key = (f"{cls_str}_"
                           f"mAR_IoU_{self.iou_range[0]:.2f}_{self.iou_range[1]:.2f}_{self.iou_range[2]:.2f}_"
                           f"MaxDet_{max_det}")
                    results[key] = self.select_ar(dataset_statistics,
                                                  cls_idx=cls_idx, max_det_idx=max_det_idx, iou_idx=self.iou_range_idx)

        for idx in self.iou_list_idx:   # AR@IoU
            key = f"AR_IoU_{self.iou_thresholds[idx]:.2f}_MaxDet_{self.max_detections[-1]}"
            results[key] = self.select_ar(dataset_statistics, iou_idx=idx, max_det_idx=-1)

            if self.per_class:
                for cls_idx, cls_str in enumerate(self.classes):  # per class results
                    key = (f"{cls_str}_"
                           f"AR_IoU_{self.iou_thresholds[idx]:.2f}_"
                           f"MaxDet_{self.max_detections[-1]}")
                    results[key] = self.select_ar(dataset_statistics, 
                                                 iou_idx=idx, cls_idx=cls_idx, max_det_idx=-1)
        return results

    @staticmethod
    def select_ap(
        dataset_statistics,
        iou_idx=None,
        cls_idx=None,
        max_det_idx=-1
    ):
        """
        Compute average precision
        Args:
            dataset_statistics (dict): computed statistics over dataset
                `counts`: Number of thresholds, Number recall thresholds, Number of classes, Number of max
                    detection thresholds
                `recall`: Computed recall values [num_iou_th, num_classes, num_max_detections]
                `precision`: Precision values at specified recall thresholds
                    [num_iou_th, num_recall_th, num_classes, num_max_detections]
                `scores`: Scores corresponding to specified recall thresholds
                    [num_iou_th, num_recall_th, num_classes, num_max_detections]
            iou_idx: index of IoU values to select for evaluation(if None, all values are used)
            cls_idx: class indices to select, if None all classes will be selected
            max_det_idx (int): index to select max detection threshold from data
        Returns:
            np.ndarray: AP value
        """
        prec = dataset_statistics["precision"]
        if iou_idx is not None:
            prec = prec[iou_idx]
        if cls_idx is not None:
            prec = prec[..., cls_idx, :]
        prec = prec[..., max_det_idx]
        return np.mean(prec)

    @staticmethod
    def select_ar(
        dataset_statistics,
        iou_idx=None,
        cls_idx=None,
        max_det_idx=-1
    ):
        """
        Compute average recall
        Args:
            dataset_statistics (dict): computed statistics over dataset
                `counts`: Number of thresholds, Number recall thresholds, Number of classes, Number of max
                    detection thresholds
                `recall`: Computed recall values [num_iou_th, num_classes, num_max_detections]
                `precision`: Precision values at specified recall thresholds
                    [num_iou_th, num_recall_th, num_classes, num_max_detections]
                `scores`: Scores corresponding to specified recall thresholds
                    [num_iou_th, num_recall_th, num_classes, num_max_detections]
            iou_idx: index of IoU values to select for evaluation(if None, all values are used)
            cls_idx: class indices to select, if None all classes will be selected
            max_det_idx (int): index to select max detection threshold from data
        Returns:
            np.ndarray: recall value
        """
        rec = dataset_statistics["recall"]
        if iou_idx is not None:
            rec = rec[iou_idx]
        if cls_idx is not None:
            rec = rec[..., cls_idx, :]
        rec = rec[..., max_det_idx]

        if len(rec[rec > -1]) == 0:
            rec = -1
        else:
            rec = np.mean(rec[rec > -1])
        return rec
    
    def check_number_of_iou(self, *args) -> None:
        """
        Check if shape of input in first dimension is consistent with expected IoU values
        (assumes IoU dimension is the first dimension)
        Args:
            args: array like inputs with shape function
        """
        num_ious = len(self.get_iou_thresholds())
        for arg in args:
            assert arg.shape[0] == num_ious

    def compute_statistics(self, results_list):
        """
        Compute statistics needed for COCO metrics (mAP, AP of individual classes, mAP@IoU_Thresholds, AR)
        Adapted from https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocotools/cocoeval.py
        Args:
            results_list (List[Dict[int, Dict[str, np.ndarray]]]): list with result s per image (in list) 
                per cateory (dict). Inner Dict contains multiple results obtained by :func:`box_matching_batch`.
                `dtMatches`: matched detections [T, D], where T = number of thresholds, D = number of detections
                `gtMatches`: matched ground truth boxes [T, G], where T = number of thresholds, G = number of
                    ground truth
                `dtScores`: prediction scores [D] detection scores
                `gtIgnore`: ground truth boxes which should be ignored [G] indicate whether ground truth should be 
                    ignored
                `dtIgnore`: detections which should be ignored [T, D], indicate which detections should be ignored
        Returns:
            dict: computed statistics over dataset
                `counts`: Number of thresholds, Number recall thresholds, Number of classes, Number of max
                    detection thresholds
                `recall`: Computed recall values [num_iou_th, num_classes, num_max_detections]
                `precision`: Precision values at specified recall thresholds
                    [num_iou_th, num_recall_th, num_classes, num_max_detections]
                `scores`: Scores corresponding to specified recall thresholds
                    [num_iou_th, num_recall_th, num_classes, num_max_detections]
        """
        num_iou_th = len(self.iou_thresholds)
        num_recall_th = len(self.recall_thresholds)
        num_classes = len(self.classes)
        num_max_detections = len(self.max_detections)

        # -1 for the precision of absent categories
        precision = -np.ones((num_iou_th, num_recall_th, num_classes, num_max_detections))
        recall = -np.ones((num_iou_th, num_classes, num_max_detections))
        scores = -np.ones((num_iou_th, num_recall_th, num_classes, num_max_detections))

        for cls_idx, cls_i in enumerate(self.classes):  # for each class
            for maxDet_idx, maxDet in enumerate(self.max_detections):  # for each maximum number of detections
                results = [r[cls_idx] for r in results_list if cls_idx in r]    # get results for each class

                if len(results) == 0:
                    continue

                dt_scores = np.concatenate([r['dtScores'][0:maxDet] for r in results])  # get class dt scores 

                # different sorting method generates slightly different results.
                # mergesort is used to be consistent as Matlab implementation.
                inds = np.argsort(-dt_scores, kind='mergesort')
                dt_scores_sorted = dt_scores[inds]  # scores sorte by value

                # r['dtMatches'] [T, R], where R = sum(all detections) and T = iou_thresholds + sorted by score
                dt_matches = np.concatenate([r['dtMatches'][:, 0:maxDet] for r in results], axis=1)[:, inds]
                dt_ignores = np.concatenate([r['dtIgnore'][:, 0:maxDet] for r in results], axis=1)[:, inds]
                self.check_number_of_iou(dt_matches, dt_ignores)
                gt_ignore = np.concatenate([r['gtIgnore'] for r in results])
                num_gt = np.count_nonzero(gt_ignore == 0)  # number of ground truth boxes (non ignored)
                if num_gt == 0:
                    continue

                # ignore cases need to be handled differently for tp and fp
                tps = np.logical_and(dt_matches,  np.logical_not(dt_ignores))
                fps = np.logical_and(np.logical_not(dt_matches), np.logical_not(dt_ignores))

                tp_sum = np.cumsum(tps, axis=1).astype(dtype=np.float32)
                fp_sum = np.cumsum(fps, axis=1).astype(dtype=np.float32)

                for th_ind, (tp, fp) in enumerate(zip(tp_sum, fp_sum)):  # for each threshold th_ind
                    tp, fp = np.array(tp), np.array(fp)
                    r, p, s = compute_stats_single_threshold(tp, fp, dt_scores_sorted, self.recall_thresholds, num_gt)
                    recall[th_ind, cls_idx, maxDet_idx] = r
                    precision[th_ind, :, cls_idx, maxDet_idx] = p   # basically the precision recall curve
                    scores[th_ind, :, cls_idx, maxDet_idx] = s      # corresponding score thresholds for recall steps

        return {
            'counts': [num_iou_th, num_recall_th, num_classes, num_max_detections],  # [4]
            'recall':   recall,  # [num_iou_th, num_classes, num_max_detections]
            'precision': precision,  # [num_iou_th, num_recall_th, num_classes, num_max_detections]
            'scores': scores,  # [num_iou_th, num_recall_th, num_classes, num_max_detections]
        }

def compute_stats_single_threshold(
    tp,
    fp,
    dt_scores_sorted,
    recall_thresholds,
    num_gt
):
    """
    Compute recall value, precision curve and scores thresholds
    Adapted from https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocotools/cocoeval.py
    Args:
        tp (np.ndarray): cumsum over true positives [R], R is the number of detections
        fp (np.ndarray): cumsum over false positives [R], R is the number of detections
        dt_scores_sorted (np.ndarray): sorted (descending) scores [R], R is the number of detections
        recall_thresholds (Sequence[float]): recall thresholds which should be evaluated
        num_gt (int): number of ground truth bounding boxes (excluding boxes which are ignored)
    Returns:
        float: overall recall for given IoU value
        np.ndarray: precision values at defined recall values
            [RTH], where RTH is the number of recall thresholds
        np.ndarray: prediction scores corresponding to recall values
            [RTH], where RTH is the number of recall thresholds
    """
    num_recall_th = len(recall_thresholds)

    rc = tp / num_gt    # equal to def of: tp / (tp + fn)
    # np.spacing(1) is the smallest representable epsilon with float
    pr = tp / (fp + tp + np.spacing(1))

    if len(tp):
        recall = rc[-1]
    else:
        # no prediction
        recall = 0

    # array where precision values nearest to given recall th are saved
    precision = np.zeros((num_recall_th,))  # precision-recall curve
    # save scores for corresponding recall value in here
    th_scores = np.zeros((num_recall_th,))
    # numpy is slow without cython optimization for accessing elements
    # use python array gets significant speed improvement
    pr = pr.tolist(); precision = precision.tolist()

    # smooth precision curve (create box shape)
    for i in range(len(tp) - 1, 0, -1):
        if pr[i] > pr[i-1]:
            pr[i-1] = pr[i]

    # get indices to nearest given recall threshold (nn interpolation!)
    inds = np.searchsorted(rc, recall_thresholds, side='left')
    try:    # breaks bc of IndexError for array_index
        for save_idx, array_index in enumerate(inds):
            precision[save_idx] = pr[array_index]
            th_scores[save_idx] = dt_scores_sorted[array_index]
    except:
        pass

    return recall, np.array(precision), np.array(th_scores)
