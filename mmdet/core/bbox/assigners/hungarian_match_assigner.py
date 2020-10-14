import torch
from scipy.optimize import linear_sum_assignment

from ..builder import BBOX_ASSIGNERS
from ..transforms import bbox_cxcywh_to_xyxy, bbox_xyxy_to_cxcywh
from .assign_result import AssignResult
from .base_assigner import BaseAssigner


@BBOX_ASSIGNERS.register_module()
class HungarianMatcher(BaseAssigner):

    def __init__(self, cls_wei=1., bbox_wei=1., giou_wei=1.):
        self.cls_wei = cls_wei
        self.bbox_wei = bbox_wei
        self.giou_wei = giou_wei

    # TODO check torch.no_grad in DETR
    def assign(self,
               bbox_pred,
               cls_pred,
               gt_bboxes,
               gt_bboxes_ignore=None,
               gt_labels=None,
               img_meta=None,
               eps=1e-7):
        """Forward function.

        Args:
            bbox_pred (Tensor): Shape [num_query, 4].
            cls_pred (Tensor): Shape [num_query, num_class].

        Returns:
            Tensor: Output results.
        """
        assert gt_bboxes_ignore is None, 'Not supported for gt_bboxes_ignore'
        assert gt_labels is not None and img_meta is not None
        num_gts, num_bboxes = gt_bboxes.size(0), bbox_pred.size(0)
        img_h, img_w, _ = img_meta['img_shape']
        assigned_gt_inds = bbox_pred.new_full((num_bboxes, ),
                                              -1,
                                              dtype=torch.long)
        assigned_labels = bbox_pred.new_full((num_bboxes, ),
                                             -1,
                                             dtype=torch.long)
        if num_gts == 0 or num_bboxes == 0:
            # No ground truth or boxes, return empty assignment
            if num_gts == 0:
                # No ground truth, assign all to background
                assigned_gt_inds[:] = 0
            return AssignResult(
                num_gts, assigned_gt_inds, None, labels=assigned_labels)

        # classification cost
        cls_score = cls_pred.softmax(-1)
        cls_cost = -cls_score[:, gt_labels]  # [num_query, num_gt]

        # regression L1 cost
        factor = torch.Tensor([img_w, img_h, img_w,
                               img_h]).unsqueeze(0).to(gt_bboxes.device)
        gt_bboxes_normalized = gt_bboxes / factor
        bbox_cost = torch.cdist(
            bbox_pred, bbox_xyxy_to_cxcywh(gt_bboxes_normalized),
            p=1)  # [num_query, num_gt]

        # regression giou cost
        bboxes = bbox_cxcywh_to_xyxy(bbox_pred) * factor
        lt = torch.max(bboxes[:, None, :2], gt_bboxes[None, :, :2])
        rb = torch.min(bboxes[:, None, 2:], gt_bboxes[None, :, 2:])
        wh = (rb - lt).clamp(min=0)  # [num_query, num_gt, 2]
        overlap = wh[..., 0] * wh[..., 1]
        # calculate ious
        area1 = (bboxes[:, 2] - bboxes[:, 0]) * (bboxes[:, 3] - bboxes[:, 1])
        area2 = (gt_bboxes[:, 2] - gt_bboxes[:, 0]) * (
            gt_bboxes[:, 3] - gt_bboxes[:, 1])
        union = area1[:, None] + area2[None, :] - overlap
        # eps = union.new_tensor([eps])
        union = union.clamp(min=eps)
        ious = overlap / union
        # calculate enclose_area
        enclose_lt = torch.min(bboxes[:, None, :2], gt_bboxes[None, :, :2])
        enclose_rb = torch.max(bboxes[:, None, 2:], gt_bboxes[None, :, 2:])
        enclose_wh = (enclose_rb - enclose_lt).clamp(min=0)
        enclose_area = enclose_wh[..., 0] * enclose_wh[..., 1]
        enclose_area = enclose_area.clamp(min=eps)
        gious = ious - (enclose_area - union) / enclose_area
        giou_cost = -gious

        # weighted sum of above three costs
        cost = self.cls_wei * cls_cost + self.bbox_wei * bbox_cost
        cost = cost + self.giou_wei * giou_cost

        # assign all indices to background first
        assigned_gt_inds[:] = 0
        # do HungarianMatcher on CPU using linear_sum_assignment
        cost = cost.detach().cpu()
        matched_row_inds, matched_col_inds = linear_sum_assignment(cost)
        matched_row_inds = torch.from_numpy(matched_row_inds).to(
            bbox_pred.device)
        matched_col_inds = torch.from_numpy(matched_col_inds).to(
            bbox_pred.device)
        assigned_gt_inds[matched_row_inds] = matched_col_inds + 1
        assigned_labels[matched_row_inds] = gt_labels[matched_col_inds]
        return AssignResult(
            num_gts, assigned_gt_inds, None, labels=assigned_labels)
