"""
Created on Sun Feb 18 15:15:42 2024

@author: zcf
西北B区 / 285机, 20241017_092241
"""
import math
from typing import List, Sequence, Tuple, Union

import torch
import torch.nn as nn
from mmcv.cnn import ConvModule
from mmdet.models.utils import multi_apply
from mmdet.utils import (ConfigType, OptConfigType, OptInstanceList,
                         OptMultiConfig)
from mmengine.dist import get_dist_info
from mmengine.model import BaseModule
from mmengine.structures import InstanceData
from torch import Tensor

from mmyolo.registry import MODELS, TASK_UTILS
from mmdet.structures.bbox import bbox_cxcywh_to_xyxy, bbox_xyxy_to_cxcywh
from mmengine.config import ConfigDict
from ..utils import gt_instances_preprocess, make_divisible
from .yolov5_head import YOLOv5Head
import torch.nn.functional as F
from mmcv.cnn import Scale
import copy
from typing import Optional
from mmdet.models.utils import filter_scores_and_topk
from mmdet.models.utils import unpack_gt_instances
from mmengine.logging import MMLogger
from mmcv.cnn import build_conv_layer
@MODELS.register_module()
class YOLOv8HeadModuleWithInit_dcn(BaseModule):
    def __init__(self,
                 num_classes: int,
                 in_channels: Union[int, Sequence],
                 widen_factor: float = 1.0,
                 featmap_strides: Sequence[int] = (8, 16, 32),
                 norm_cfg: ConfigType = dict(
                     type='BN', momentum=0.03, eps=0.001),
                 act_cfg: ConfigType = dict(type='SiLU', inplace=True),
                 img_scale=(640, 640), base_sizes=None,
                 dcn_on_last_conv = False, 
                 reg_mid_ratio = 1/4,
                 conv_cfg = None,
                 stacked_convs = 2,
                 reg_max=1,
                 kernel_last = 1,
                 stacked_conv_groups=1,
                 convType_notUse_groups=[],
                 conv_bias: Union[bool, str] = 'auto',
                 # last_conv_cfg=None,
                 init_cfg: OptMultiConfig = None):
        super().__init__(init_cfg=init_cfg)
        self.num_classes = num_classes
        self.featmap_strides = featmap_strides
        self.num_levels = len(self.featmap_strides)
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        self.in_channels = in_channels
        self.reg_max = reg_max

        in_channels = []
        for channel in self.in_channels:
            channel = make_divisible(channel, widen_factor)
            in_channels.append(channel)
        self.in_channels = in_channels
        #------------
        self.dcn_on_last_conv = dcn_on_last_conv
        self.reg_mid_ratio = reg_mid_ratio
        #------------
        self.stacked_convs = stacked_convs
        self.conv_cfg = conv_cfg
        self.kernel_last = kernel_last
        assert kernel_last in [1, 3], "kernel_last must be 1 or 3"
        self.conv_bias = conv_bias
        # self.last_conv_cfg = last_conv_cfg
        self.stacked_conv_groups = stacked_conv_groups
        self.convType_notUse_groups = convType_notUse_groups
        self._init_layers()
        #------------only for init_weights
        self.img_scale = img_scale
        self.base_sizes = base_sizes
        #------------only for init_weights
        '''
        base_sizes: 
        需要修改dataset_type=mmdet.CocoDataset
        debugfile('/media/zcf/extra/zcf/code/231108_FoundationModels/mmyolo_forv8/mmyolo/tools/analysis_tools/optimize_anchors.py', wdir='/media/zcf/extra/zcf/code/231108_FoundationModels/mmyolo_forv8/mmyolo/tools/analysis_tools',args='/media/zcf/extra/zcf/code/231108_FoundationModels/mmyolo_forv8/mmyolo/0solar_configs/N11Ours.py --input-shape 1024 1024 --algorithm v5-k-means')
        '''
        
    def init_weights(self, prior_prob=0.01):
        """Initialize the weight and bias of PPYOLOE head."""
        # 模型结构打印到终端和log文件中
        logger: MMLogger = MMLogger.get_current_instance()
        logger.info('YOLOv8HeadModuleWithInit_dcn')
        logger.info(self)
        super().init_weights()
        # from mmengine.model import bias_init_with_prob
        # bias_init = bias_init_with_prob(0.01)
        if self.base_sizes is None:
            self.base_sizes = [(2, 2)] * self.num_levels
        base_size_index = 0
        assert len(self.base_sizes) == self.num_levels, '初始化的box的个数应该等于特征图的个数，即len(self.base_sizes) == self.num_levels'
        for reg_pred, cls_pred, stride in zip(self.reg_preds, self.cls_preds, self.featmap_strides):
            # 设置回归预测的偏置
            # 选择当前 reg_pred 对应的 1 个 base_sizes
            selected_base_sizes = self.base_sizes[base_size_index : base_size_index + 1]
            # 计算 x1, y1, x2, y2
            # reg_pred[-1].weight.data[:] = 0.0
            reg_pred[-1].bias.data[:] = torch.tensor(selected_base_sizes).repeat(1,2).view(-1) / (2.0 * stride)
            # 更新 base_size_index 为下一组 base_sizes 的起始索引
            base_size_index += 1
            # cls (.01 objects, 80 classes, 640 img)
            # 设置分类预测的偏置
            # cls_pred[-1].bias.data.fill_(bias_init)
            cls_pred[-1].bias.data[:self.num_classes] = math.log(
                5 / self.num_classes / (self.img_scale[0] / stride) / (self.img_scale[1] / stride))
    def _init_layers(self):
        """initialize conv layers in YOLOv8 head."""
        # Init decouple head
        self.cls_preds = nn.ModuleList()
        self.reg_preds = nn.ModuleList()
        
        # reg_out_channels = max(
        #     (16, self.in_channels[0] // 4, self.reg_max * 4))
        reg_out_channels = max(
            (16, int(self.in_channels[0]*self.reg_mid_ratio)))
        cls_out_channels = max(self.in_channels[0], self.num_classes)
        padding_last = 0 if self.kernel_last == 1 else 1
        
        for j in range(self.num_levels):
            reg_convs_single, cls_preds_single = [], []
            for i in range(self.stacked_convs):
                reg_chn = self.in_channels[j] if i == 0 else reg_out_channels
                cls_chn = self.in_channels[j] if i == 0 else cls_out_channels
                if self.dcn_on_last_conv and i == self.stacked_convs - 1:
                    conv_cfg = dict(type='DCN')
                    use_bias = 'auto' if self.conv_bias else self.conv_bias
                else:
                    conv_cfg = self.conv_cfg
                    use_bias=self.conv_bias
                #-----------------
                conv_type = conv_cfg.get('type', None) if conv_cfg is not None else None
                conv_groups = 1 if conv_type in self.convType_notUse_groups else self.stacked_conv_groups
                reg_convs_single.append(
                    ConvModule(
                        reg_chn,
                        reg_out_channels,
                        3,
                        stride=1,
                        padding=1,
                        conv_cfg=conv_cfg,
                        groups=conv_groups,
                        norm_cfg=self.norm_cfg,
                        act_cfg=self.act_cfg,
                        bias=use_bias))
                cls_preds_single.append(
                    ConvModule(
                        cls_chn,
                        cls_out_channels,
                        3,
                        stride=1,
                        padding=1,
                        conv_cfg=conv_cfg,
                        groups=conv_groups,
                        norm_cfg=self.norm_cfg,
                        act_cfg=self.act_cfg,
                        bias=use_bias))
            # if self.last_conv_cfg == None:
            reg_convs_single.append(
                nn.Conv2d(
                        in_channels=reg_out_channels,
                        out_channels=4,
                        kernel_size=self.kernel_last,
                        padding=padding_last))
            cls_preds_single.append(
                nn.Conv2d(
                        in_channels=cls_out_channels,
                        out_channels=self.num_classes,
                        kernel_size=self.kernel_last,
                        padding=padding_last))
            # else:
            #     reg_convs_single.append(
            #         build_conv_layer(
            #                 self.last_conv_cfg,
            #                 in_channels=reg_out_channels,
            #                 out_channels=4,
            #                 kernel_size=self.kernel_last,
            #                 padding=padding_last))
            #     cls_preds_single.append(
            #         build_conv_layer(
            #                 self.last_conv_cfg,
            #                 in_channels=cls_out_channels,
            #                 out_channels=self.num_classes,
            #                 kernel_size=self.kernel_last,
            #                 padding=padding_last))
            self.reg_preds.append(nn.Sequential(*reg_convs_single))
            self.cls_preds.append(nn.Sequential(*cls_preds_single))

    def forward(self, x: Tuple[Tensor]) -> Tuple[List]:
        """Forward features from the upstream network.

        Args:
            x (Tuple[Tensor]): Features from the upstream network, each is
                a 4D-tensor.
        Returns:
            Tuple[List]: A tuple of multi-level classification scores, bbox
            predictions
        """
        assert len(x) == self.num_levels
        return multi_apply(self.forward_single, x, self.cls_preds,
                           self.reg_preds)
    def forward_single(self, x: torch.Tensor, cls_pred: nn.ModuleList,
                        reg_pred: nn.ModuleList) -> Tuple:
        """Forward feature of a single scale level."""
        b, _, h, w = x.shape
        cls_logit = cls_pred(x)
        bbox_dist_preds = reg_pred(x)
        bbox_preds = bbox_dist_preds
        if self.training:
            return cls_logit, bbox_preds, bbox_dist_preds
        else:
            return cls_logit, bbox_preds
# copy from mmyolo/mmyolo/models/dense_heads/yolov8_head.py: YOLOv8Head
# 直接改变YOLOv8Head，因为不希望嵌套太多
@MODELS.register_module()
class YOLOv8HeadFlexibleLoss(YOLOv5Head):
    """YOLOv8Head head used in `YOLOv8`.

    Args:
        head_module(:obj:`ConfigDict` or dict): Base module used for YOLOv8Head
        prior_generator(dict): Points generator feature maps
            in 2D points-based detectors.
        bbox_coder (:obj:`ConfigDict` or dict): Config of bbox coder.
        loss_cls (:obj:`ConfigDict` or dict): Config of classification loss.
        loss_bbox (:obj:`ConfigDict` or dict): Config of localization loss.
        loss_dfl (:obj:`ConfigDict` or dict): Config of Distribution Focal
            Loss.
        train_cfg (:obj:`ConfigDict` or dict, optional): Training config of
            anchor head. Defaults to None.
        test_cfg (:obj:`ConfigDict` or dict, optional): Testing config of
            anchor head. Defaults to None.
        init_cfg (:obj:`ConfigDict` or list[:obj:`ConfigDict`] or dict or
            list[dict], optional): Initialization config dict.
            Defaults to None.
    """

    def __init__(self,
                 head_module: ConfigType,
                 prior_generator: ConfigType = dict(
                     type='mmdet.MlvlPointGenerator',
                     offset=0.5,
                     strides=[8, 16, 32]),
                 bbox_coder: ConfigType = dict(type='DistancePointBBoxCoder'),
                 loss_cls: ConfigType = dict(
                     type='mmdet.CrossEntropyLoss',
                     use_sigmoid=True,
                     reduction='none',
                     loss_weight=0.5),
                 loss_bbox: Union[List[ConfigType],ConfigType] = dict(
                     type='IoULoss',
                     iou_mode='ciou',
                     bbox_format='xyxy',
                     reduction='sum',
                     loss_weight=7.5,
                     return_iou=False),
                 loss_dfl=None, # dict(type='mmdet.DistributionFocalLoss',reduction='mean',loss_weight=1.5 / 4),
                 train_cfg: OptConfigType = None,
                 test_cfg: OptConfigType = None,
                 init_cfg: OptMultiConfig = None):
        if isinstance(loss_bbox, (ConfigDict,dict)):
            loss_bbox = [loss_bbox]
        loss_bbox_dict = loss_bbox[0]     
        super().__init__(
            head_module=head_module,
            prior_generator=prior_generator,
            bbox_coder=bbox_coder,
            loss_cls=loss_cls,
            loss_bbox=loss_bbox_dict, #-----------zhou
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            init_cfg=init_cfg)
        del self.loss_bbox
        self.loss_bbox = [
            MODELS.build(loss_bbo) for loss_bbo in loss_bbox
        ]
        self.loss_dfl = None
        if loss_dfl is not None:
            self.loss_dfl = MODELS.build(loss_dfl)
        # YOLOv8 doesn't need loss_obj
        self.loss_obj = None

    def special_init(self):
        """Since YOLO series algorithms will inherit from YOLOv5Head, but
        different algorithms have special initialization process.

        The special_init function is designed to deal with this situation.
        """
        if self.train_cfg:
            self.assigner = TASK_UTILS.build(self.train_cfg.assigner)

            # Add common attributes to reduce calculation
            self.featmap_sizes_train = None
            self.num_level_priors = None
            self.flatten_priors_train = None
            self.stride_tensor = None
            
    def loss_by_feat(
            self,
            cls_scores: Sequence[Tensor],
            bbox_preds: Sequence[Tensor],
            bbox_dist_preds: Sequence[Tensor],
            batch_gt_instances: Sequence[InstanceData],
            batch_img_metas: Sequence[dict],
            batch_gt_instances_ignore: OptInstanceList = None) -> dict:
        """Calculate the loss based on the features extracted by the detection
        head.

        Args:
            cls_scores (Sequence[Tensor]): Box scores for each scale level,
                each is a 4D-tensor, the channel number is
                num_priors * num_classes.
            bbox_preds (Sequence[Tensor]): Box energies / deltas for each scale
                level, each is a 4D-tensor, the channel number is
                num_priors * 4.
            bbox_dist_preds (Sequence[Tensor]): Box distribution logits for
                each scale level with shape (bs, reg_max + 1, H*W, 4).
            batch_gt_instances (list[:obj:`InstanceData`]): Batch of
                gt_instance. It usually includes ``bboxes`` and ``labels``
                attributes.
            batch_img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            batch_gt_instances_ignore (list[:obj:`InstanceData`], optional):
                Batch of gt_instances_ignore. It includes ``bboxes`` attribute
                data that is ignored during training and testing.
                Defaults to None.
        Returns:
            dict[str, Tensor]: A dictionary of losses.
        """
        num_imgs = len(batch_img_metas)
        current_featmap_sizes = [
            cls_score.shape[2:] for cls_score in cls_scores
        ]
        # If the shape does not equal, generate new one
        if current_featmap_sizes != self.featmap_sizes_train:
            self.featmap_sizes_train = current_featmap_sizes

            mlvl_priors_with_stride = self.prior_generator.grid_priors(
                self.featmap_sizes_train,
                dtype=cls_scores[0].dtype,
                device=cls_scores[0].device,
                with_stride=True)
            self.num_level_priors = [len(n) for n in mlvl_priors_with_stride]
            self.flatten_priors_train = torch.cat(
                mlvl_priors_with_stride, dim=0)
            self.stride_tensor = self.flatten_priors_train[..., [2]]
        # gt info
        gt_info = gt_instances_preprocess(batch_gt_instances, num_imgs)
        gt_labels = gt_info[:, :, :1]
        gt_bboxes = gt_info[:, :, 1:]  # xyxy
        pad_bbox_flag = (gt_bboxes.sum(-1, keepdim=True) > 0).float()

        # pred info
        flatten_cls_preds = [
            cls_pred.permute(0, 2, 3, 1).reshape(num_imgs, -1,
                                                 self.num_classes)
            for cls_pred in cls_scores
        ]
        flatten_pred_bboxes = [
            bbox_pred.permute(0, 2, 3, 1).reshape(num_imgs, -1, 4)
            for bbox_pred in bbox_preds
        ]
        # (bs, n, 4 * reg_max)
        flatten_pred_dists = [
            bbox_pred_org.reshape(num_imgs, -1, self.head_module.reg_max * 4)
            for bbox_pred_org in bbox_dist_preds
        ]

        flatten_dist_preds = torch.cat(flatten_pred_dists, dim=1)
        flatten_cls_preds = torch.cat(flatten_cls_preds, dim=1)
        flatten_pred_bboxes = torch.cat(flatten_pred_bboxes, dim=1)
        flatten_pred_bboxes = self.bbox_coder.decode(
            self.flatten_priors_train[..., :2], flatten_pred_bboxes,
            self.stride_tensor[..., 0])

        assigned_result = self.assigner(
            (flatten_pred_bboxes.detach()).type(gt_bboxes.dtype),
            flatten_cls_preds.detach().sigmoid(), self.flatten_priors_train,
            gt_labels, gt_bboxes, pad_bbox_flag)

        assigned_bboxes = assigned_result['assigned_bboxes']
        assigned_scores = assigned_result['assigned_scores']
        fg_mask_pre_prior = assigned_result['fg_mask_pre_prior']

        assigned_scores_sum = assigned_scores.sum().clamp(min=1)
        # use_sample = True
        # if use_sample:
        #     flatten_cls_preds = flatten_cls_preds.view(-1, 6)
        #     assigned_scores_sample = assigned_scores.view(-1, 6)
        #     # 对assigned_scores的最后一维求和，得到形状为(2, 3150)
        #     scores_sum = assigned_scores_sample.sum(dim=-1)
        #     # 获取 >0 的索引
        #     positive_indices = torch.nonzero(scores_sum > 0).squeeze() 
        #     # 获取 =0 的索引
        #     zero_indices = torch.nonzero(scores_sum == 0).squeeze()
        #     num_to_keep = max(positive_indices.numel(), zero_indices.numel(), 256)
        #     perm = torch.randperm(zero_indices.numel())
        #     random_neg_indices = zero_indices[perm[:num_to_keep]]
        #     indices_to_keep = torch.cat([positive_indices, random_neg_indices])
        #     # 根据索引选择元素
        #     selected_cls_preds = flatten_cls_preds[indices_to_keep]
        #     selected_scores = assigned_scores_sample[indices_to_keep]
        #     loss_cls = self.loss_cls(selected_cls_preds, selected_scores).sum()
        # else:
            # print('flatten_cls_preds',flatten_cls_preds.shape,assigned_scores.shape)
        loss_cls = self.loss_cls(flatten_cls_preds, assigned_scores).sum()
        
        # if 'Focal' in self.loss_cls.__class__.__name__:
        #     num_pos = fg_mask_pre_prior.sum()
        #     if num_pos > 0:
        #         assigned_labels = F.one_hot(assigned_result['assigned_labels'], self.num_classes)
        #         loss_cls = self.loss_cls(flatten_cls_preds, assigned_labels).sum()
        #     else:
        #         loss_cls = flatten_cls_preds.sum() * 0
        # else:
        #     num_pos = fg_mask_pre_prior.sum()
        #     if num_pos > 0:
        #         loss_cls = self.loss_cls(flatten_cls_preds, assigned_scores).sum()
        #     else:
        #         loss_cls = flatten_cls_preds.sum() * 0
        loss_cls /= assigned_scores_sum

        # rescale bbox
        assigned_bboxes /= self.stride_tensor
        flatten_pred_bboxes /= self.stride_tensor

        # select positive samples mask
        num_pos = fg_mask_pre_prior.sum()
        if num_pos > 0:
            # when num_pos > 0, assigned_scores_sum will >0, so the loss_bbox
            # will not report an error
            # iou loss
            prior_bbox_mask = fg_mask_pre_prior.unsqueeze(-1).repeat([1, 1, 4])
            pred_bboxes_pos = torch.masked_select(
                flatten_pred_bboxes, prior_bbox_mask).reshape([-1, 4])
            assigned_bboxes_pos = torch.masked_select(
                assigned_bboxes, prior_bbox_mask).reshape([-1, 4])
            bbox_weight = torch.masked_select(
                assigned_scores.sum(-1), fg_mask_pre_prior).unsqueeze(-1)
            
            #-----------------loss 240530------step 1
            # loss_bbox = self.loss_bbox(
            #     pred_bboxes_pos, assigned_bboxes_pos,
            #     weight=bbox_weight) / assigned_scores_sum
            loss_bbox = dict()
            for loss_index in range(len(self.loss_bbox)):
                loss_name = self.loss_bbox[loss_index].__class__.__name__
                loss_bbox['loss_bbox_'+loss_name] = self.loss_bbox[loss_index](
                    pred_bboxes_pos, assigned_bboxes_pos,
                    weight=bbox_weight.repeat([1, 4])) / assigned_scores_sum        
                
            #-----------------loss 240530------step 2  
            if self.loss_dfl is not None:
                # dfl loss
                pred_dist_pos = flatten_dist_preds[fg_mask_pre_prior]
                assigned_ltrb = self.bbox_coder.encode(
                    self.flatten_priors_train[..., :2] / self.stride_tensor,
                    assigned_bboxes,
                    max_dis=self.head_module.reg_max - 1,
                    eps=0.01)
                assigned_ltrb_pos = torch.masked_select(
                    assigned_ltrb, prior_bbox_mask).reshape([-1, 4])
                loss_dfl = self.loss_dfl(
                    pred_dist_pos.reshape(-1, self.head_module.reg_max),
                    assigned_ltrb_pos.reshape(-1),
                    weight=bbox_weight.expand(-1, 4).reshape(-1),
                    avg_factor=assigned_scores_sum)
        else:
            #-----------------loss 240530------step 3
            # loss_bbox = flatten_pred_bboxes.sum() * 0
            # loss_dfl = flatten_pred_bboxes.sum() * 0
            loss_bbox = dict()
            for loss_index in range(len(self.loss_bbox)):
                loss_name = self.loss_bbox[loss_index].__class__.__name__
                loss_bbox['loss_bbox_'+loss_name] = flatten_pred_bboxes.sum() * 0
            if self.loss_dfl is not None:
                loss_dfl = flatten_pred_bboxes.sum() * 0
                
        _, world_size = get_dist_info()
        
        #-----------------loss 240530------step 4
        if self.loss_dfl is None:
            return_loss =  dict(loss_cls=loss_cls)
        else:
            return_loss =  dict(loss_cls=loss_cls,
                                loss_dfl=loss_dfl)
        return_loss.update(loss_bbox)
        for loss_name,loss_value in return_loss.items():
            return_loss[loss_name] = loss_value* num_imgs * world_size
        return return_loss
    def predict_by_feat(self,
                        cls_scores: List[Tensor],
                        bbox_preds: List[Tensor],
                        objectnesses: Optional[List[Tensor]] = None,
                        batch_img_metas: Optional[List[dict]] = None,
                        cfg: Optional[ConfigDict] = None,
                        rescale: bool = True,
                        with_nms: bool = True) -> List[InstanceData]:
        """Transform a batch of output features extracted by the head into
        bbox results.
        Args:
            cls_scores (list[Tensor]): Classification scores for all
                scale levels, each is a 4D-tensor, has shape
                (batch_size, num_priors * num_classes, H, W).
            bbox_preds (list[Tensor]): Box energies / deltas for all
                scale levels, each is a 4D-tensor, has shape
                (batch_size, num_priors * 4, H, W).
            objectnesses (list[Tensor], Optional): Score factor for
                all scale level, each is a 4D-tensor, has shape
                (batch_size, 1, H, W).
            batch_img_metas (list[dict], Optional): Batch image meta info.
                Defaults to None.
            cfg (ConfigDict, optional): Test / postprocessing
                configuration, if None, test_cfg would be used.
                Defaults to None.
            rescale (bool): If True, return boxes in original image space.
                Defaults to False.
            with_nms (bool): If True, do nms before return boxes.
                Defaults to True.

        Returns:
            list[:obj:`InstanceData`]: Object detection results of each image
            after the post process. Each item usually contains following keys.

            - scores (Tensor): Classification scores, has a shape
              (num_instance, )
            - labels (Tensor): Labels of bboxes, has a shape
              (num_instances, ).
            - bboxes (Tensor): Has a shape (num_instances, 4),
              the last dimension 4 arrange as (x1, y1, x2, y2).
        """
        assert len(cls_scores) == len(bbox_preds)
        if objectnesses is None:
            with_objectnesses = False
        else:
            with_objectnesses = True
            assert len(cls_scores) == len(objectnesses)

        cfg = self.test_cfg if cfg is None else cfg
        cfg = copy.deepcopy(cfg)

        multi_label = cfg.multi_label
        multi_label &= self.num_classes > 1
        cfg.multi_label = multi_label

        num_imgs = len(batch_img_metas)
        featmap_sizes = [cls_score.shape[2:] for cls_score in cls_scores]

        # If the shape does not change, use the previous mlvl_priors
        if featmap_sizes != self.featmap_sizes:
            self.mlvl_priors = self.prior_generator.grid_priors(
                featmap_sizes,
                dtype=cls_scores[0].dtype,
                device=cls_scores[0].device)
            self.featmap_sizes = featmap_sizes
        flatten_priors = torch.cat(self.mlvl_priors)

        mlvl_strides = [
            flatten_priors.new_full(
                (featmap_size.numel(), ), stride) for
            featmap_size, stride in zip(featmap_sizes, self.featmap_strides)
        ]
        flatten_stride = torch.cat(mlvl_strides)

        # flatten cls_scores, bbox_preds and objectness
        flatten_cls_scores = [
            cls_score.permute(0, 2, 3, 1).reshape(num_imgs, -1,
                                                  self.num_classes)
            for cls_score in cls_scores
        ]
        flatten_bbox_preds = [
            bbox_pred.permute(0, 2, 3, 1).reshape(num_imgs, -1, 4)
            for bbox_pred in bbox_preds
        ]

        flatten_cls_scores = torch.cat(flatten_cls_scores, dim=1).sigmoid()
        flatten_bbox_preds = torch.cat(flatten_bbox_preds, dim=1)
        flatten_decoded_bboxes = self.bbox_coder.decode(
            flatten_priors[None], flatten_bbox_preds, flatten_stride)

        if with_objectnesses:
            flatten_objectness = [
                objectness.permute(0, 2, 3, 1).reshape(num_imgs, -1)
                for objectness in objectnesses
            ]
            flatten_objectness = torch.cat(flatten_objectness, dim=1).sigmoid()
        else:
            flatten_objectness = [None for _ in range(num_imgs)]

        results_list = []
        for (bboxes, scores, objectness,
             img_meta) in zip(flatten_decoded_bboxes, flatten_cls_scores,
                              flatten_objectness, batch_img_metas):
            ori_shape = img_meta['ori_shape']
            scale_factor = img_meta['scale_factor']
            if 'pad_param' in img_meta:
                pad_param = img_meta['pad_param']
            else:
                pad_param = None

            score_thr = cfg.get('score_thr', -1)
            # yolox_style does not require the following operations
            if objectness is not None and score_thr > 0 and not cfg.get(
                    'yolox_style', False):
                conf_inds = objectness > score_thr
                bboxes = bboxes[conf_inds, :]
                scores = scores[conf_inds, :]
                objectness = objectness[conf_inds]

            if objectness is not None:
                # conf = obj_conf * cls_conf
                scores *= objectness[:, None]

            if scores.shape[0] == 0:
                empty_results = InstanceData()
                empty_results.bboxes = bboxes
                empty_results.scores = scores[:, 0]
                empty_results.labels = scores[:, 0].int()
                results_list.append(empty_results)
                continue

            nms_pre = cfg.get('nms_pre', 100000)
            if cfg.multi_label is False:
                scores, labels = scores.max(1, keepdim=True)
                scores, _, keep_idxs, results = filter_scores_and_topk(
                    scores,
                    score_thr,
                    nms_pre,
                    results=dict(labels=labels[:, 0]))
                labels = results['labels']
            else:
                scores, labels, keep_idxs, _ = filter_scores_and_topk(
                    scores, score_thr, nms_pre)

            results = InstanceData(
                scores=scores, labels=labels, bboxes=bboxes[keep_idxs])

            if rescale:
                if pad_param is not None:
                    results.bboxes -= results.bboxes.new_tensor([
                        pad_param[2], pad_param[0], pad_param[2], pad_param[0]
                    ])
                results.bboxes /= results.bboxes.new_tensor(
                    scale_factor).repeat((1, 2))

            if cfg.get('yolox_style', False):
                # do not need max_per_img
                cfg.max_per_img = len(results)

            results = self._bbox_post_process(
                results=results,
                cfg=cfg,
                rescale=False,
                with_nms=with_nms,
                img_meta=img_meta)
            results.bboxes[:, 0::2].clamp_(0, ori_shape[1])
            results.bboxes[:, 1::2].clamp_(0, ori_shape[0])

            results_list.append(results)
        return results_list  