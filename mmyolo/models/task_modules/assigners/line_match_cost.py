"""
Latest version 2025-04-03-16:00:00, (1) fix IoULoss for loss_by_feat_single and HungarianAssigner, add IoUCost (为了支持ciou)
Created on Tue Jan 16 09:47:56 2024

@author: zcf
"""
from typing import Optional, Union

import torch
from mmengine.structures import InstanceData
from torch import Tensor

from mmyolo.registry import TASK_UTILS
from mmdet.models.task_modules.assigners.match_cost import BaseMatchCost
from mmyolo.models.losses import bbox_overlaps

pts_dim=2
    
@TASK_UTILS.register_module()
class LaneL1Cost(BaseMatchCost):
    r"""
    Notes
    -----
    Adapted from https://github.com/open-mmlab/mmdetection/blob/master/mmdet/core/bbox/match_costs/match_cost.py#L11.
    一条线上的点的差值是相加的, 因为L1OrderLoss，L1Loss对一条线上的点的差值, 也是相加的

    """
    def __init__(self, normalize=False, weight: Union[float, int] = 1., clc_order='Chamfer') -> None:
        self.normalize = normalize
        self.weight = weight
        self.clc_order = clc_order

    def __call__(self,
                 pred_instances: InstanceData,
                 gt_instances: InstanceData,
                 img_meta: Optional[dict] = None,
                 **kwargs) -> Tensor:
        line_pred = pred_instances.lines #bboxes #----------for iouloss
        gt_lines = gt_instances.lines #.bboxes #----------for iouloss
        if self.normalize:
            img_h, img_w = img_meta['img_shape']
            factor = gt_lines.new_tensor([img_w, img_h]).unsqueeze(0)
            gt_lines = gt_lines / factor
            line_pred = line_pred / factor
        # if self.clc_order == 'minDist' and (line_pred.shape[1] // pts_dim !=2):
        #      self.clc_order = 'default'
        #      print('!!!!!!!!!!!self.clc_order = default')
        if self.clc_order == 'default':
            if len(line_pred.shape)==len(gt_lines.shape)==2:
                lane_cost = torch.cdist(line_pred, gt_lines, p=1) #[num_pred_line,num_gt_line]
            else:
                gt_lines = gt_lines.permute(1,0,2) 
                lane_cost = (line_pred - gt_lines).abs().sum(dim=-1)
        else:
            if len(line_pred.shape)==len(gt_lines.shape)==2:
                num_pred_line = line_pred.shape[0]
                num_gt_line = gt_lines.shape[0]
                #[num_pred_line,num_gt_line,pred_line_numPts,gt_line_numPts]
                dist_mat = torch.cdist(line_pred.reshape(num_pred_line,-1,pts_dim).unsqueeze(1), gt_lines.reshape(num_gt_line,-1,pts_dim).unsqueeze(0), p=1)
                if self.clc_order == 'Chamfer':
                    dist_pred = dist_mat.min(-1)[0].sum(-1) #[num_pred_line,num_gt_line]
                    dist_gt = dist_mat.min(-2)[0].sum(-1) #[num_pred_line,num_gt_line]
                    lane_cost = (dist_pred + dist_gt) / 2
            else:
                print('not implement')
            # if self.clc_order == 'ChamferMax':
            #     dist_pred = dist_mat.min(-1)[0] #[num_pred_line,num_gt_line,pred_line_numPts]
            #     dist_gt = dist_mat.min(-2)[0] #[num_pred_line,num_gt_line,gt_line_numPts]
            #     lane_cost = torch.max(dist_pred, dist_gt) #[num_pred_line,num_gt_line,pred_line_numPts]
            #     lane_cost = lane_cost.sum(-1) #[num_pred_line,num_gt_line]
            # elif self.clc_order == 'minDist':
            #     # 计算斜对角线元素的和
            #     diag1_sum = dist_mat[..., 0, 0] + dist_mat[..., 1, 1]  # a + d
            #     diag2_sum = dist_mat[..., 0, 1] + dist_mat[..., 1, 0]  # b + c
            #     # 找到两条斜对角线和的最小值
            #     lane_cost = torch.min(diag1_sum, diag2_sum) #[num_pred_line,num_gt_line]
        return lane_cost * self.weight
    
# @TASK_UTILS.register_module()
# class LineCostv2(BaseMatchCost):
#     r"""
#     Notes
#     -----
#     Adapted from https://github.com/open-mmlab/mmdetection/blob/master/mmdet/core/bbox/match_costs/match_cost.py#L11.

#     """
#     def __init__(self, normalize=False, weight: Union[float, int] = 1.) -> None:
#         self.normalize = normalize
#         self.weight = weight

#     def __call__(self,
#                  pred_instances: InstanceData,
#                  gt_instances: InstanceData,
#                  img_meta: Optional[dict] = None,
#                  **kwargs) -> Tensor:
#         line_pred = pred_instances.bboxes #[num_queries,2+3]
#         gt_lines = gt_instances.bboxes #[num_gt,num_points,2]
#         centerXY = line_pred[:,:2]
#         findMinPts(centerXY,gt_lines)
#         centerXY_cost=
        
            
#         lane_cost = torch.cdist(line_pred, gt_lines, p=1)
#         return lane_cost * self.weight
'''
copy from mmdetection_v3d3d0/mmdet/models/task_modules/assigners/match_cost.py
only for change bbox_overlaps: from mmdet.structures.bbox import bbox_overlaps ---> from mmyolo.models.losses import bbox_overlaps
'''
@TASK_UTILS.register_module()
class IoUCost(BaseMatchCost):
    """IoUCost.

    Note: ``bboxes`` in ``InstanceData`` passed in is of format 'xyxy'
    and its coordinates are unnormalized.

    Args:
        iou_mode (str): iou mode such as 'iou', 'giou'. Defaults to 'giou'.
        weight (Union[float, int]): Cost weight. Defaults to 1.

    Examples:
        >>> from mmdet.models.task_modules.assigners.
        ... match_costs.match_cost import IoUCost
        >>> import torch
        >>> self = IoUCost()
        >>> bboxes = torch.FloatTensor([[1,1, 2, 2], [2, 2, 3, 4]])
        >>> gt_bboxes = torch.FloatTensor([[0, 0, 2, 4], [1, 2, 3, 4]])
        >>> self(bboxes, gt_bboxes)
        tensor([[-0.1250,  0.1667],
            [ 0.1667, -0.5000]])
    """

    def __init__(self, iou_mode: str = 'giou', weight: Union[float, int] = 1.):
        super().__init__(weight=weight)
        self.iou_mode = iou_mode

    def __call__(self,
                 pred_instances: InstanceData,
                 gt_instances: InstanceData,
                 img_meta: Optional[dict] = None,
                 **kwargs):
        """Compute match cost.

        Args:
            pred_instances (:obj:`InstanceData`): ``bboxes`` inside is
                predicted boxes with unnormalized coordinate
                (x, y, x, y).
            gt_instances (:obj:`InstanceData`): ``bboxes`` inside is gt
                bboxes with unnormalized coordinate (x, y, x, y).
            img_meta (Optional[dict]): Image information. Defaults to None.

        Returns:
            Tensor: Match Cost matrix of shape (num_preds, num_gts).
        """
        pred_bboxes = pred_instances.bboxes
        gt_bboxes = gt_instances.bboxes

        # avoid fp16 overflow
        if pred_bboxes.dtype == torch.float16:
            fp16 = True
            pred_bboxes = pred_bboxes.to(torch.float32)
        else:
            fp16 = False

        # overlaps = bbox_overlaps(
        #     pred_bboxes, gt_bboxes, mode=self.iou_mode, is_aligned=False)
        overlaps = bbox_overlaps(
            pred_bboxes.unsqueeze(1), gt_bboxes.unsqueeze(0), iou_mode=self.iou_mode, bbox_format='xyxy') #.clamp(0)

        if fp16:
            overlaps = overlaps.to(torch.float16)

        # The 1 is a constant that doesn't change the matching, so omitted.
        iou_cost = -overlaps
        return iou_cost * self.weight