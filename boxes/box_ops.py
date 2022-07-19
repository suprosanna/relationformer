import torch

from torch import Tensor
from numpy import ndarray
from typing import Union, Sequence, Tuple


def box_cxcyczwhd_to_xyxyzz(x):
    x_c, y_c, z_c, w, h, d = x.unbind(-1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h),
         (z_c - 0.5 * d), (z_c + 0.5 * d)]
    return torch.stack(b, dim=-1)


# def box_xyxy_to_cxcywh(x):
#     x0, y0, x1, y1 = x.unbind(-1)
#     b = [(x0 + x1) / 2, (y0 + y1) / 2,
#          (x1 - x0), (y1 - y0)]
#     return torch.stack(b, dim=-1)


def box_area_3d(boxes: Tensor) -> Tensor:
    """
    Computes the area of a set of bounding boxes, which are specified by its
    (x1, y1, x2, y2, z1, z2) coordinates.
    
    Arguments:
        boxes (Union[Tensor, ndarray]): boxes for which the area will be computed. They
            are expected to be in (x1, y1, x2, y2, z1, z2) format. [N, 6]
    Returns:
        area (Union[Tensor, ndarray]): area for each box [N]
    """
    return (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 5] - boxes[:, 4])


def box_area(boxes: Union[Tensor, ndarray]) -> Union[Tensor, ndarray]:
    """
    Computes the area of a set of bounding boxes
    
    Args:
        boxes (Union[Tensor, ndarray]): boxes of shape; (x1, y1, x2, y2, (z1, z2))[N, dim * 2]
    
    Returns:
        Union[Tensor, ndarray]: area of boxes
    
    See Also:
        :func:`box_area_3d`, :func:`torchvision.ops.boxes.box_area`
    """

    return box_area_3d(boxes)


def box_iou_union_3d(boxes1: Tensor, boxes2: Tensor, eps: float = 0) -> Tuple[Tensor, Tensor]:
    """
    Return intersection-over-union (Jaccard index) and  of boxes.
    Both sets of boxes are expected to be in (x1, y1, x2, y2, z1, z2) format.
    
    Args:
        boxes1: set of boxes (x1, y1, x2, y2, z1, z2)[N, 6]
        boxes2: set of boxes (x1, y1, x2, y2, z1, z2)[M, 6]
        eps: optional small constant for numerical stability
    Returns:
        Tensor[N, M]: the NxM matrix containing the pairwise
            IoU values for every element in boxes1 and boxes2
        Tensor[N, M]: the nxM matrix containing the pairwise union
            values
    """
    vol1 = box_area_3d(boxes1)
    vol2 = box_area_3d(boxes2)

    x1 = torch.max(boxes1[:, None, 0], boxes2[:, 0])  # [N, M]
    y1 = torch.max(boxes1[:, None, 1], boxes2[:, 1])  # [N, M]
    x2 = torch.min(boxes1[:, None, 2], boxes2[:, 2])  # [N, M]
    y2 = torch.min(boxes1[:, None, 3], boxes2[:, 3])  # [N, M]
    z1 = torch.max(boxes1[:, None, 4], boxes2[:, 4])  # [N, M]
    z2 = torch.min(boxes1[:, None, 5], boxes2[:, 5])  # [N, M]

    inter = ((x2 - x1).clamp(min=0) * (y2 - y1).clamp(min=0) * (z2 - z1).clamp(min=0)) + eps  # [N, M]
    union = (vol1[:, None] + vol2 - inter)
    return inter / union, union


def generalized_box_iou_3d(boxes1: Tensor, boxes2: Tensor, eps: float = 0) -> Tensor:
    """
    Computes the generalized box iou between given bounding boxes
    Args:
        boxes1: set of boxes (x1, y1, x2, y2, z1, z2)[N, 6]
        boxes2: set of boxes (x1, y1, x2, y2, z1, z2)[M, 6]
        eps: optional small constant for numerical stability
    Returns:
        Tensor[N, M]: the NxM matrix containing the pairwise
            generalized IoU values for every element in boxes1 and boxes2
    """
    iou, union = box_iou_union_3d(boxes1, boxes2)

    x1 = torch.min(boxes1[:, None, 0], boxes2[:, 0])  # [N, M]
    y1 = torch.min(boxes1[:, None, 1], boxes2[:, 1])  # [N, M]
    x2 = torch.max(boxes1[:, None, 2], boxes2[:, 2])  # [N, M]
    y2 = torch.max(boxes1[:, None, 3], boxes2[:, 3])  # [N, M]
    z1 = torch.min(boxes1[:, None, 4], boxes2[:, 4])  # [N, M]
    z2 = torch.max(boxes1[:, None, 5], boxes2[:, 5])  # [N, M]

    vol = ((x2 - x1).clamp(min=0) * (y2 - y1).clamp(min=0) * (z2 - z1).clamp(min=0)) + eps  # [N, M]
    return iou - (vol - union) / vol