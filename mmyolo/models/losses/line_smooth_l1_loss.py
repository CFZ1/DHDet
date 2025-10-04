# Copyright (c) OpenMMLab. All rights reserved.
from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor

from mmyolo.registry import MODELS
from mmdet.models import weighted_loss
pts_dim=2
@weighted_loss
def l1_order_loss(pred: Tensor, target: Tensor, clc_order: str) -> Tensor:
    """L1 loss.

    Args:
        pred (Tensor): The prediction.
        target (Tensor): The learning target of the prediction.

    Returns:
        Tensor: Calculated loss
    """
    if target.numel() == 0:
        return pred.sum() * 0

    assert pred.size() == target.size()
    if clc_order == 'default':
        loss = torch.abs(pred - target) #[num_pred_line,pred_line_numPts*pts_dim]
    else:
        num_pred_line = pred.shape[0]
        # [num_pred_line,pred_line_numPts,tg_line_numPts,pts_dim]
        dist_mat = torch.abs(pred.reshape(num_pred_line,-1,pts_dim).unsqueeze(2) - target.reshape(num_pred_line,-1,pts_dim).unsqueeze(1))
        if clc_order == 'Chamfer':
            dist_1 = dist_mat.min(2)[0] #[num_pred_line,pred_line_numPts,num_gt_line]
            dist_2 = dist_mat.min(1)[0] #[num_pred_line,pred_line_numPts,num_gt_line]
            loss = (dist_1 + dist_2) / 2 #[num_pred_line,pred_line_numPts,num_gt_line
            loss = loss.reshape(num_pred_line,-1)
        # elif clc_order == 'minDist':
        #     # 计算斜对角线元素的和
        #     diag1_sum = dist_mat[:, 0, 0] + dist_mat[..., 1, 1]  # a + d
        #     diag2_sum = dist_mat[:, 0, 1] + dist_mat[..., 1, 0]  # b + c
        #     # 找到两条斜对角线和的最小值
        #     lane_cost = torch.min(diag1_sum, diag2_sum) #[num_pred_line,num_gt_line]
    return loss

@MODELS.register_module()
class L1OrderLoss(nn.Module):
    """L1 loss.

    Args:
        reduction (str, optional): The method to reduce the loss.
            Options are "none", "mean" and "sum".
        loss_weight (float, optional): The weight of loss.
    """

    def __init__(self,
                 reduction: str = 'mean',
                 loss_weight: float = 1.0, clc_order='Chamfer') -> None:
        super().__init__()
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.clc_order = clc_order

    def forward(self,
                pred: Tensor,
                target: Tensor,
                weight: Optional[Tensor] = None,
                avg_factor: Optional[int] = None,
                reduction_override: Optional[str] = None) -> Tensor:
        """Forward function.

        Args:
            pred (Tensor): The prediction.
            target (Tensor): The learning target of the prediction.
            weight (Tensor, optional): The weight of loss for each
                prediction. Defaults to None.
            avg_factor (int, optional): Average factor that is used to average
                the loss. Defaults to None.
            reduction_override (str, optional): The reduction method used to
                override the original reduction method of the loss.
                Defaults to None.

        Returns:
            Tensor: Calculated loss
        """
        # if weight is not None and not torch.any(weight > 0): #ref mmdet.L1Loss(smooth_l1_loss.py) for fast 240826
        #     return (pred * weight).sum()                     #ref mmdet.L1Loss(smooth_l1_loss.py) for fast 240826
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        loss_bbox = self.loss_weight * l1_order_loss(
            pred, target, weight, clc_order=self.clc_order, reduction=reduction, avg_factor=avg_factor)
        num_pts = pred.shape[-1]//pts_dim
        # 
        return loss_bbox/num_pts
