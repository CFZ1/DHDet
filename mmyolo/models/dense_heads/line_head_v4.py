"""
Latest version 2025-04-05-09:00:00 
(1) line_head_v4.py版本回退， Latest version 2025-03-22-21:16:00, 
(1.1)只保留 lineRefPts,  
(1.2) 而且HungarianAssigner的输入labels=gt_labels, lines=gt_lanes, bboxes=gt_lanes_box，保存lines，这样line_match_cost.py的输入就不修改了
"""
import copy
from typing import Sequence, Tuple, Union

import torch
import torch.nn as nn
from mmdet.models.dense_heads.base_dense_head import BaseDenseHead
from mmdet.models.utils import multi_apply
from mmdet.utils import (ConfigType, OptInstanceList,InstanceList,OptMultiConfig)
from mmengine.structures import InstanceData
from torch import Tensor

from mmyolo.registry import MODELS, TASK_UTILS
from mmcv.cnn import Linear
from typing import Dict
from mmdet.structures import OptSampleList
import torch.nn.functional as F
from mmdet.models import DeformableDetrTransformerDecoder,DeformableDetrTransformerEncoder
from mmcv.cnn.bricks.transformer import MultiScaleDeformableAttention
from mmengine.model import xavier_init
from torch.nn.init import normal_
from mmdet.models import inverse_sigmoid
from mmdet.utils import reduce_mean
from mmengine.utils import digit_version
from math import factorial
import numpy as np
from shapely.geometry import LineString
from mmcv.cnn import ConvModule
from mmengine.model import bias_init_with_prob, constant_init
from mmyolo.models import prepare_line_points
from mmyolo.models.layers import correct_reference_points_by_length_scaling, correct_lines_batch_origin
import math
from mmdet.models import MLP
from mmdet.structures.bbox import bbox_cxcywh_to_xyxy

# proposals_init_mode = 1 #[0,1,2]
# transformer=dict(
#     type='PETRTransformer',
#     decoder=dict(
#         type='PETRTransformerDecoder',
#         return_intermediate=True,
#         num_layers=6,
#         transformerlayers=dict(
#             type='PETRTransformerDecoderLayer',
#             attn_cfgs=[
#                 dict(
#                     type='MultiheadAttention',
#                     embed_dims=256,
#                     num_heads=8,
#                     dropout=0.1),
#                 dict(
#                     type='PETRMultiheadAttention',
#                     embed_dims=256,
#                     num_heads=8,
#                     dropout=0.1),
#                 ],
#             feedforward_channels=2048,
#             ffn_dropout=0.1,
#             operation_order=('self_attn', 'norm', 'cross_attn', 'norm',
#                              'ffn', 'norm')),
#     )),

# transformer=dict(
#     type='TopoNetTransformerDecoderOnly',
#     embed_dims=_dim_,
#     pts_dim=pts_dim,
#     decoder=dict(
#         type='TopoNetSGNNDecoder',
#         num_layers=6,
#         return_intermediate=True,
#         transformerlayers=dict(
#             type='SGNNDecoderLayer',
#             attn_cfgs=[
#                 dict(
#                     type='MultiheadAttention',
#                     embed_dims=_dim_,
#                     num_heads=8,
#                     dropout=0.1),
#                  dict(
#                     type='CustomMSDeformableAttention',
#                     embed_dims=_dim_,
#                     num_levels=1),
#             ],
#             ffn_cfgs=dict(
#                 type='FFN_SGNN',
#                 embed_dims=_dim_,
#                 feedforward_channels=_ffn_dim_,
#                 num_te_classes=13,
#                 edge_weight=0.6),
#             operation_order=('self_attn', 'norm', 'cross_attn', 'norm',
#                              'ffn', 'norm')))),

# copy from mmdet/models/detectors/deformable_detr.py; copy from mmdet/models/detectors/deformable_detr.py;
# copy from TopoNet/projects/toponet/models/dense_heads/toponet_head.py
@MODELS.register_module()
class coLineHeadv4(BaseDenseHead):
    r"""

    Args:
        with_box_refine (bool, optional): Whether to refine the references
            in the decoder. Defaults to `False`.
        as_two_stage (bool, optional): Whether to generate the proposal
            from the outputs of encoder. Defaults to `False`.
        num_feature_levels (int, optional): Number of feature levels.
            Defaults to 3.
    3个最重要的函数：loss\ predict \forward
    toponet and topomlp: 神经网络输出的边框预测缩放到真实尺度上，然后计算损失，predict输出
    本模型与detr一样: 边框预测缩放到img shape（输入神经网络的尺寸）上，然后1.计算损失，2.需要对缩放的图像尺寸还原，predict输出，
    """
    def __init__(self,
                 num_classes: int,
                 in_channels: Union[int, Sequence],
                 embed_dims: int = 256,
                 num_feature_levels: int = 3,
                 num_queries: int = 300,
                 with_box_refine: bool = True,
                 as_two_stage: bool = False,
                 num_reg_fcs=2,
                 share_pred_layer: bool = False,
                 code_size = 2 * 11,  #默认11个点，每个点两个坐标
                 code_weights=None,
                 sync_cls_avg_factor: bool = True,
                 fineLength: bool = False,
                 lineRefPts: int = 2,
                 correctAngle: bool = False,
                 correctAngleSigFirst: bool = False,
                 transformer=None,
                 positional_encoding=dict(
                     type='SinePositionalEncoding',
                     num_feats=128,
                     normalize=True,
                     offset=-0.5),
                 # loss_cls=dict(
                 #     type='mmdet.CrossEntropyLoss',
                 #     use_sigmoid=True,
                 #     reduction='none',
                 #     loss_weight=1.0),
                 loss_bbox_pre=dict(line_points_inter_method ='lineSegmentUni',# lineSegmentUni,bezier
                                    points_for_lossMetric = 4,
                                    inter_reg = False),
                 loss_cls=dict(
                     type='mmdet.FocalLoss',
                     use_sigmoid=True,
                     gamma=2.0,
                     alpha=0.25,
                     loss_weight=2.0),
                 loss_bbox=dict(type='mmdet.L1Loss', loss_weight=5.0),
                 loss_iou=dict(type='mmdet.GIoULoss', loss_weight=2.0), #not use
                 train_cfg: ConfigType = dict(
                     assigner=dict(
                         type='mmdet.HungarianAssigner',
                         match_costs=[
                             dict(type='mmdet.FocalLossCost', weight=1.5),
                             dict(type='LaneL1Cost', weight=0.025)]),
                     sampler=None),
                 test_cfg: ConfigType = dict(max_per_img=100), #TODO! not use it!! TOponet and topomlp donot use it, but DeformableDETRHead use it
                 init_cfg: OptMultiConfig = None
                ):
        super().__init__(init_cfg=init_cfg)
        self.pts_dim = 2 # 默认一条线(line)上2个点, 一个点x,y两个维度
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.num_reg_fcs = num_reg_fcs
        self.num_feature_levels = num_feature_levels
        self.embed_dims = embed_dims 
        self.as_two_stage = as_two_stage # TODO!
        self.num_queries = num_queries
        self.with_box_refine = with_box_refine
        # assert self.with_box_refine is False, 'only support False now'
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        
        self.bg_cls_weight = 0
        self.sync_cls_avg_factor = sync_cls_avg_factor
        class_weight = loss_cls.get('class_weight', None)
        if class_weight is not None and (self.__class__ is coLineHeadv4):
            assert isinstance(class_weight, float), 'Expected ' \
                                                    'class_weight to have type float. Found ' \
                                                    f'{type(class_weight)}.'
            # NOTE following the official DETR rep0, bg_cls_weight means relative classification weight of the no-object class.
            bg_cls_weight = loss_cls.get('bg_cls_weight', class_weight)
            assert isinstance(bg_cls_weight, float), 'Expected ' \
                                                     'bg_cls_weight to have type float. Found ' \
                                                     f'{type(bg_cls_weight)}.'
            class_weight = torch.ones(num_classes + 1) * class_weight
            # set background class as the last indice
            class_weight[num_classes] = bg_cls_weight
            loss_cls.update({'class_weight': class_weight})
            if 'bg_cls_weight' in loss_cls:
                loss_cls.pop('bg_cls_weight')
            self.bg_cls_weight = bg_cls_weight
        #---------bg filter 240622    
        self.use_bg = loss_cls.get('use_bg', False) 
        if 'use_bg' in loss_cls:
            loss_cls.pop('use_bg')
        if (not self.use_bg) and loss_cls.use_sigmoid:
        # if loss_cls.use_sigmoid:
        #---------bg filter 240622
            self.cls_out_channels = num_classes
        else:
            self.cls_out_channels = num_classes + 1   
            
        self.loss_cls: nn.Module = MODELS.build(loss_cls)
        self.loss_bbox: nn.Module = MODELS.build(loss_bbox)
        self.loss_iou: nn.Module = MODELS.build(loss_iou)
        if self.train_cfg:
            assert 'assigner' in self.train_cfg, 'assigner should be provided when train_cfg is set.'
            assigner = self.train_cfg['assigner']
            self.assigner = TASK_UTILS.build(assigner)
            # DETR sampling=False, so use PseudoSampler
            self.sampler = None
            if self.train_cfg.get('sampler', None) is not None:
                self.sampler = TASK_UTILS.build(self.train_cfg['sampler'], default_args=dict(context=self)) #dict(type='PseudoSampler')
        # 统一多尺度特征的通道数目
        self.input_proj = None
        if self.in_channels != self.embed_dims:
            self.in_channels = [in_channels] if isinstance(in_channels, int) else in_channels
            self.input_proj = nn.ModuleList()
            for i in range(len(self.in_channels)):
                self.input_proj.append(
                    nn.Sequential(
                        ConvModule(
                            in_channels=self.in_channels[i],
                            out_channels=self.embed_dims,
                            kernel_size=3,
                            stride=1,
                            padding=1,
                            norm_cfg=dict(type='BN'),
                            act_cfg=dict(type='ReLU', inplace=True)),
                        nn.Conv2d(
                            in_channels=self.embed_dims,
                            out_channels=self.embed_dims,
                            kernel_size=1)))
        if transformer:
            # self.transformer = MODELS.build(transformer) # TODO!
            self.encoder, self.decoder = None, None
            if transformer.get('encoder', None):
                if transformer.encoder.type =='DeformableDetrTransformerEncoder':
                    transformer.encoder.pop('type')
                    self.encoder = DeformableDetrTransformerEncoder(**transformer.encoder)
            if transformer.decoder.type =='DeformableDetrTransformerDecoder':
                transformer.decoder.pop('type')
                self.decoder = DeformableDetrTransformerDecoder(**transformer.decoder) #return_intermediate=True
            else:
                #----------zcf self.as_two_stage 240520----------step 4, 传输一些参数给self.decoder
                transformer.decoder['as_two_stage'] = self.as_two_stage
                transformer.decoder['ref_numPts'] = lineRefPts
                #----------zcf self.as_two_stage 240520----------step 4, 传输一些参数给self.decoder
                self.decoder = MODELS.build(transformer.decoder)
            self.positional_encoding = MODELS.build(positional_encoding)
            if self.encoder is not None:
                self.level_embed = nn.Parameter(
                    torch.Tensor(self.num_feature_levels, self.embed_dims))
            if not self.as_two_stage:
                self.query_embedding = nn.Embedding(self.num_queries,
                                                    self.embed_dims * 2)
            if self.as_two_stage:
                self.memory_trans_fc = nn.Linear(self.embed_dims, self.embed_dims)
                self.memory_trans_norm = nn.LayerNorm(self.embed_dims)
                # self.line_proposal = MLP(self.embed_dims,self.embed_dims//2,2, 2)
                self.query_embedding = nn.Embedding(self.num_queries, self.embed_dims)
                self.Fea_LineClassifier = nn.Linear(self.embed_dims, 1) 
                # NOTE In DINO, the query_embedding only contains content
                # queries, while in Deformable DETR, the query_embedding
                # contains both content and spatial queries, and in DETR,
                # it only contains spatial queries.
                # self.pos_trans_fc = nn.Linear(self.embed_dims * 2,
                #                               self.embed_dims * 2)
                # self.pos_trans_norm = nn.LayerNorm(self.embed_dims * 2)
            else:
                self.reference_points_fc = nn.Linear(self.embed_dims, lineRefPts*self.pts_dim)
                
        self.code_size = code_size
        if loss_bbox_pre:
            self.points_for_lossMetric = loss_bbox_pre.points_for_lossMetric #计算损失的时候，统一预测和gt的点数，两者都是points_for_lossMetric=11个
            self.line_points_inter_method = loss_bbox_pre.line_points_inter_method 
            self.inter_reg = loss_bbox_pre.inter_reg 
            self.decoder.line_points_inter_method = self.line_points_inter_method
            # if not self.inter_reg:
            #     self.points_for_lossMetric =  self.code_size //self.pts_dim
        else:
           self.points_for_lossMetric = 2 # 默认一条线(line)上2个点 
           self.inter_reg = False # 默认不对边框分支输出的线（n个点）进行插点
        self.decoder.points_for_lossMetric = self.points_for_lossMetric
        if code_weights is not None:
            self.code_weights = code_weights
        else:
            self.code_weights = [1.0, ] * self.points_for_lossMetric*self.pts_dim #TopoMLP and toponet are different
        self.code_weights = nn.Parameter(torch.tensor(
            self.code_weights, requires_grad=False), requires_grad=False)
        
        cls_branch = []
        for _ in range(self.num_reg_fcs):
            cls_branch.append(Linear(self.embed_dims, self.embed_dims))
            cls_branch.append(nn.LayerNorm(self.embed_dims))
            cls_branch.append(nn.ReLU(inplace=True))
        cls_branch.append(Linear(self.embed_dims, self.cls_out_channels))
        fc_cls = nn.Sequential(*cls_branch)
        #--------for v3-----------
        self.fineLength = fineLength
        self.correctAngle = correctAngle
        self.correctAngleSigFirst = correctAngleSigFirst
        self.decoder.correctAngle = self.correctAngle
        self.decoder.correctAngleSigFirst = self.correctAngleSigFirst
        self.lineRefPts = lineRefPts
        if self.correctAngle:
            self.correctAngleIndex = lineRefPts*self.pts_dim
            self.decoder.correctAngleIndex = self.correctAngleIndex
            reg_branch_outLength = lineRefPts*self.pts_dim+1 #LPx+LPy+RPx+RPy+Angle=5
        else:
            reg_branch_outLength = lineRefPts*self.pts_dim #LPx+LPy+RPx+RPy=4
        #--------for v3-----------
        reg_branch = []
        for _ in range(self.num_reg_fcs):
            reg_branch.append(Linear(self.embed_dims, self.embed_dims))
            reg_branch.append(nn.ReLU())
        reg_branch.append(Linear(self.embed_dims, reg_branch_outLength))
        reg_branch = nn.Sequential(*reg_branch)
        
        # self.num_pred_layer = transformer.decoder.num_layers
        self.num_pred_layer = (transformer.decoder.num_layers + 1) if self.as_two_stage else transformer.decoder.num_layers
        self.share_pred_layer = share_pred_layer
        if self.share_pred_layer:
            self.cls_branches = nn.ModuleList(
                [fc_cls for _ in range(self.num_pred_layer)])
            self.reg_branches = nn.ModuleList(
                [reg_branch for _ in range(self.num_pred_layer)])
        else:
            self.cls_branches = nn.ModuleList(
                [copy.deepcopy(fc_cls) for _ in range(self.num_pred_layer)])
            self.reg_branches = nn.ModuleList([
                copy.deepcopy(reg_branch) for _ in range(self.num_pred_layer)
            ])
        
    # copy from mmdet/models/detectors/deformable_detr.py
    def init_weights(self) -> None:
        """Initialize weights for Transformer and other components."""
        super().init_weights()
        for coder in self.encoder, self.decoder:
            if coder is not None: #-----add by zhou
                for p in coder.parameters():
                    if p.dim() > 1:
                        nn.init.xavier_uniform_(p)
        for m in self.modules():
            if isinstance(m, MultiScaleDeformableAttention):
                m.init_weights()
        if self.as_two_stage:
            nn.init.xavier_uniform_(self.memory_trans_fc.weight)
            nn.init.xavier_uniform_(self.query_embedding.weight)
            # nn.init.xavier_uniform_(self.pos_trans_fc.weight)
        else:
            xavier_init(
                self.reference_points_fc, distribution='uniform', bias=0.)
        if self.encoder is not None:
            normal_(self.level_embed)
        # copy from mmdet/models/dense_heads/deformable_detr_head.py
        """Initialize weights of the Deformable DETR head."""
        if self.loss_cls.use_sigmoid:
            bias_init = bias_init_with_prob(0.01)
            for m in self.cls_branches:
                if isinstance(m, nn.Sequential):#---zhou
                    nn.init.constant_(m[-1].bias, bias_init) #---zhou m.bias-->m[-1].bias
                else:
                    nn.init.constant_(m.bias, bias_init) #---zhou
        # 确保角度修正分支的bias的初始化=0.0
        if self.correctAngle:
            for m in self.reg_branches:
                m[-1].bias.data[-1] = 0.0  # box
        # for m in self.reg_branches:
        #     constant_init(m[-1], 0, bias=0)
        # nn.init.constant_(self.reg_branches[0][-1].bias.data[2:], -2.0)
         #---not used in toponet and topomlp
        # for m in self.reg_branches:
        #     constant_init(m[-1], 0, bias=0)
        # nn.init.constant_(self.reg_branches[0][-1].bias.data[2:], -2.0)
        # if self.as_two_stage:
        #     for m in self.reg_branches:
        #         nn.init.constant_(m[-1].bias.data[2:], 0.0)           
    def loss(self, x: Tuple[Tensor], batch_data_samples: Union[list,dict], return_queries=False) -> dict:
        """Perform forward propagation and loss calculation of the detection
        head on the features of the upstream network.

        Args:
            x (tuple[Tensor]): Features from the upstream network, each is
                a 4D-tensor. 
            batch_data_samples (List[:obj:`DetDataSample`], dict): The Data
                Samples. It usually includes information such as
                `gt_instance`, `gt_panoptic_seg` and `gt_sem_seg`.

        Returns:
            dict: A dictionary of loss components.
        参考mmyolo/mmyolo/models/dense_heads/yolov5_head.py; 
        toponet/models/dense_heads/toponet_head.py; mmdet/models/dense_heads/deformable_detr_head.py
        (num_decoder_layers, bs, num_queries, cls_out_channels)
        """
        all_cls_scores, all_lanes_preds, hs, outputs_coords_ratios, query_pos, enc_cls_scores, enc_bbox_preds = self(x,batch_data_samples) # [num_layers, bs, num_queries, num_points]
        num_dec_layers = len(all_cls_scores)
        # layer_index = [i for i in range(num_dec_layers)]
        
        batch_gt_instances = []
        batch_gt_ignore_instances = []
        batch_img_metas = []
        for data_sample in batch_data_samples:
            batch_img_metas.append(data_sample.metainfo)
            batch_gt_instances.append(data_sample.gt_line_instances) #gt_line_instances
            batch_gt_ignore_instances.append(data_sample.ignored_line_instances) #gt_line_instances
            
        batch_gt_instances_list = [batch_gt_instances for _ in range(num_dec_layers)] 
        batch_gt_ignore_instances_list = [batch_gt_ignore_instances for _ in range(num_dec_layers)]
        batch_img_metas_list = [batch_img_metas for _ in range(num_dec_layers)]
        
        losses_cls, losses_bbox, assign_results = multi_apply(
            self.loss_by_feat_single, all_cls_scores, all_lanes_preds,
            batch_gt_instances_list, batch_img_metas_list, batch_gt_ignore_instances_list, outputs_coords_ratios)  
        # all_gt_lines_list = [batch_data_samples['line_points'] for _ in range(num_dec_layers)] 
        # all_gt_lines_labels_list = [batch_data_samples['line_labels'] for _ in range(num_dec_layers)]
        # all_img_metas_list = [batch_data_samples['img_metas'] for _ in range(num_dec_layers)]   
        # losses_cls, losses_bbox = multi_apply(
        #     self.loss_by_feat_single, all_cls_scores, all_lanes_preds,
        #     all_gt_lines_list, all_gt_lines_labels_list, layer_index, all_img_metas_list)
        
        loss_dict = dict()

        # loss from the last decoder layer
        loss_dict['loss_lane_cls'] = losses_cls[-1]
        loss_dict['loss_lane_reg'] = losses_bbox[-1]

        # loss from other decoder layers
        num_dec_layer = 0
        for loss_cls_i, loss_bbox_i in zip(
            losses_cls[:-1], losses_bbox[:-1]):
            loss_dict[f'd{num_dec_layer}.loss_lane_cls'] = loss_cls_i
            loss_dict[f'd{num_dec_layer}.loss_lane_reg'] = loss_bbox_i
            num_dec_layer += 1
        #----------zcf self.as_two_stage 240520----------step 3 end
        # loss of proposal generated from encode feature map.
        if enc_cls_scores is not None:
            # NOTE The enc_loss calculation of the DINO is
            # different from that of Deformable DETR.
            #--------------缩放到和all_lanes_preds同样的尺度上
            factors = []
            for img_meta in batch_data_samples:
                img_h, img_w = img_meta.metainfo['img_shape'] 
                factor = enc_bbox_preds.new_tensor([img_w, img_h]).unsqueeze(0)
                factors.append(factor)
            factors = torch.cat(factors, 0)
            expanded_factors = factors.unsqueeze(1).unsqueeze(2)
            enc_bbox_preds = enc_bbox_preds*expanded_factors
            enc_bbox_preds = enc_bbox_preds.flatten(2) #[bs,num_aueries,-1]
            #--------------缩放到和all_lanes_preds同样的尺度上
            enc_loss_cls, enc_losses_bbox, _ = \
                self.loss_by_feat_single(
                    enc_cls_scores, enc_bbox_preds,
                    batch_gt_instances=batch_gt_instances,
                    batch_img_metas=batch_img_metas)
            loss_dict['enc_loss_cls'] = enc_loss_cls
            loss_dict['enc_loss_reg'] = enc_losses_bbox
        #----------zcf self.as_two_stage 240520----------step 3 end
        if return_queries:  
            return loss_dict, hs[-1], query_pos, all_cls_scores[-1], all_lanes_preds[-1][:, :, [0, 1, -2, -1]], assign_results[-1]['pos_inds'] #[bs,num_query,256],[bs,num_query,1],[bs,num_query,4]
        return loss_dict
    # topomlp/models/heads/lane_head.py ; /mmdet/models/dense_heads/detr_head.py
    def predict(self, x: Tuple[Tensor], batch_data_samples: Union[list,dict], rescale: bool = True, return_queries=False) -> InstanceList:
        all_cls_scores_list, all_bbox_preds_list,hs,_,query_pos = self(x,batch_data_samples)
        cls_scores = all_cls_scores_list[-1]
        bbox_preds = all_bbox_preds_list[-1]

        result_list = []
        batch_size = cls_scores.size()[0]
        batch_img_metas = [data_samples.metainfo for data_samples in batch_data_samples]
        for img_id in range(batch_size):
            cls_score = cls_scores[img_id]
            bbox_pred = bbox_preds[img_id]
            img_meta = batch_img_metas[img_id]
            img_shape = img_meta['img_shape']
            # exclude background
            # if self.loss_cls.use_sigmoid:  #---------bg filter 240622
            if (not self.use_bg) and self.loss_cls.use_sigmoid:  #---------bg filter 240622
                cls_score = cls_score.sigmoid()
                scores, det_labels = cls_score.max(-1) #TODO! changed to 1 box vs 1 class, 
                # print('cls_scores',cls_scores)
                # if self.cls_out_channels == 1:
                #     det_labels = [None]*len(det_labels)
            else:
                # scores, det_labels = F.softmax(cls_score, dim=-1)[..., :-1].max(-1)  #num_cls+background
                #---------bg filter 240622
                # print('hhh',cls_score.shape)
                scores, det_labels = F.softmax(cls_score, dim=-1).max(-1)  #num_cls+background
                bg_index = det_labels != (cls_score.shape[-1] - 1)
                bbox_pred, scores, det_labels = bbox_pred[bg_index], scores[bg_index], det_labels[bg_index]
                #---------bg filter 240622
                
            det_bboxes = bbox_pred
            # det_bboxes[:, 0::2] = det_bboxes[:, 0::2] * img_shape[1]
            # det_bboxes[:, 1::2] = det_bboxes[:, 1::2] * img_shape[0]
            det_bboxes[:, 0::2].clamp_(min=0, max=img_shape[1]) #w
            det_bboxes[:, 1::2].clamp_(min=0, max=img_shape[0]) #h
            if rescale:
                assert img_meta.get('scale_factor') is not None
                num_point_here = det_bboxes.shape[-1] //self.pts_dim
                det_bboxes /= det_bboxes.new_tensor(
                    img_meta['scale_factor']).repeat((1, num_point_here)) #w,h,w,h...
#                 batch_data_samples[img_id].gt_line_instances.line_points /= det_bboxes.new_tensor( #---add zhou
#                     img_meta['scale_factor']).repeat((1, self.points_for_lossMetric)).reshape(-1,self.points_for_lossMetric,self.pts_dim) #w,h,w,h...
#                 batch_data_samples[img_id].gt_instances.bboxes /= det_bboxes.new_tensor(
#                     img_meta['scale_factor']).repeat((1, 2)) #TODO! 240129 For visualization

            results = InstanceData()
            results.line_points = det_bboxes #[num_queries,num_points*self.pts_dim]
            results.line_scores = scores #[num_queries,]
            results.line_labels = det_labels #[num_queries,], or None, must same length
            result_list.append(results)
        if return_queries:
            return result_list,batch_data_samples, hs[-1], query_pos, all_cls_scores_list[-1], all_bbox_preds_list[-1]  #[bs,num_query,256],[bs,num_query,1],[bs,num_query,4]
        return result_list,batch_data_samples
    # copy from /mmyolo/models/dense_heads/yolov5_head.py
    def loss_by_feat(
            self,
            cls_scores: Sequence[Tensor],
            bbox_preds: Sequence[Tensor],
            objectnesses: Sequence[Tensor],
            batch_gt_instances: Sequence[InstanceData],
            batch_img_metas: Sequence[dict],
            batch_gt_instances_ignore: OptInstanceList = None) -> dict:
        return 0
    # copy from toponet/models/dense_heads/toponet_head.py
    def loss_by_feat_single(self,
                            cls_scores,
                            lanes_preds,
                            batch_gt_instances,
                            # layer_index,
                            batch_img_metas,
                            gt_bboxes_ignore_list=None,
                            outputs_coords_ratios=None):
        '''
        (bs, num_queries, cls_out_channels)
        '''

        num_imgs = cls_scores.size(0)
        cls_scores_list = [cls_scores[i] for i in range(num_imgs)]
        lanes_preds_list = [lanes_preds[i] for i in range(num_imgs)]

        cls_reg_targets = self.get_targets(cls_scores_list, lanes_preds_list, batch_gt_instances,
                                           gt_bboxes_ignore_list,batch_img_metas,outputs_coords_ratios)
        (labels_list, label_weights_list, bbox_targets_list, bbox_weights_list,
         num_total_pos, num_total_neg, assign_result) = cls_reg_targets
        labels = torch.cat(labels_list, 0)
        label_weights = torch.cat(label_weights_list, 0)
        bbox_targets = torch.cat(bbox_targets_list, 0)
        bbox_weights = torch.cat(bbox_weights_list, 0)

        # classification loss
        cls_scores = cls_scores.reshape(-1, self.cls_out_channels)
        # construct weighted avg_factor to match with the official DETR repo
        cls_avg_factor = num_total_pos * 1.0 + \
            num_total_neg * self.bg_cls_weight #TODO !
        if self.sync_cls_avg_factor:
            cls_avg_factor = reduce_mean(
                cls_scores.new_tensor([cls_avg_factor]))
        cls_avg_factor = max(cls_avg_factor, 1)
        loss_cls = self.loss_cls(
            cls_scores, labels, label_weights, avg_factor=cls_avg_factor)

        # Compute the average number of gt boxes accross all gpus, for
        # normalization purposes
        num_total_pos = loss_cls.new_tensor([num_total_pos])
        num_total_pos = torch.clamp(reduce_mean(num_total_pos), min=1).item()
        '''
        是否需要缩放, toponet的两个计算cost和loss的时候，全部在真实尺寸上
        因此，line_points需要在图像尺寸上，模型预测的line points也默认在图像尺寸上
        
        '''
        # construct factors used for rescale bboxes
        # factors = []
        # for img_meta, bbox_pred in zip(batch_img_metas, lanes_preds):
        #     img_h, img_w, = img_meta['img_shape'] 
        #     factor = bbox_pred.new_tensor([img_w, img_h]).unsqueeze(0).repeat(
        #                                        bbox_pred.size(0), bbox_pred.size(-1)//2)
        #     factors.append(factor)
        # factors = torch.cat(factors, 0)
        lanes_preds = lanes_preds.reshape(-1, lanes_preds.size(-1))
        # lanes_preds = lanes_preds * factors
        # regression L1 loss
        isnotnan = torch.isfinite(bbox_targets).all(dim=-1)
        bbox_weights = bbox_weights * self.code_weights       
        # 统一预测和gt的点数为指定个数
        #------------zcf 240312
        # if lanes_preds[isnotnan].shape[-1] != self.points_for_lossMetric*self.pts_dim:
        #     lanes_preds = prepare_line_points(lanes_preds[isnotnan].reshape(-1,self.lineRefPts,self.pts_dim),self.line_points_inter_method,self.points_for_lossMetric).flatten(1)  #[num_lanes,num_points,pts_dim]
        # if bbox_targets[isnotnan].shape[-1] != self.points_for_lossMetric*self.pts_dim:
        #     gt_points = bbox_targets[isnotnan].shape[-1] // self.pts_dim
        #     bbox_targets = prepare_line_points(bbox_targets[isnotnan].reshape(-1,gt_points,self.pts_dim),self.line_points_inter_method,self.points_for_lossMetric).flatten(1) #[num_lanes,num_points,pts_dim]
        #------------zcf 240312
        # bbox_weights = bbox_weights[isnotnan, :self.code_size].mean(-1).unsqueeze(-1).repeat(1, lanes_preds.shape[-1])
        loss_bbox = self.loss_bbox(
            lanes_preds[isnotnan],   #-----------------240305
            bbox_targets[isnotnan],  #-----------------240305
            bbox_weights[isnotnan], #-----------------240305
            avg_factor=num_total_pos)

        if digit_version(torch.__version__) >= digit_version('1.8'):
            loss_cls = torch.nan_to_num(loss_cls)
            loss_bbox = torch.nan_to_num(loss_bbox)
        return loss_cls, loss_bbox, assign_result

    def get_targets(self,
                    cls_scores_list,
                    lanes_preds_list,
                    # lclc_preds_list,
                    batch_gt_instances,
                    # gt_lane_adj_list,
                    gt_bboxes_ignore_list=None,
                    batch_img_metas=None,
                    outputs_coords_ratios=None):
        '''
        (bs, num_queries, cls_out_channels)
        '''
        if gt_bboxes_ignore_list is not None:
            # print('Only supports for gt_bboxes_ignore setting to None.')
            gt_bboxes_ignore_list = None
        num_imgs = len(cls_scores_list)
        gt_bboxes_ignore_list = [
            gt_bboxes_ignore_list for _ in range(num_imgs)
        ]
        if outputs_coords_ratios is None:
            outputs_coords_ratios = [outputs_coords_ratios for _ in range(num_imgs)]   

        (labels_list, label_weights_list, lanes_targets_list, lanes_weights_list,
            pos_inds_list, neg_inds_list, pos_assigned_gt_inds_list) = multi_apply(
            self._get_targets_single, cls_scores_list, lanes_preds_list,
            batch_gt_instances, gt_bboxes_ignore_list,batch_img_metas,outputs_coords_ratios)
        num_total_pos = sum((inds.numel() for inds in pos_inds_list))
        num_total_neg = sum((inds.numel() for inds in neg_inds_list))
        assign_result = dict(
            pos_inds=pos_inds_list, neg_inds=neg_inds_list, pos_assigned_gt_inds=pos_assigned_gt_inds_list
        )
        return (labels_list, label_weights_list, lanes_targets_list, lanes_weights_list,
                num_total_pos, num_total_neg, assign_result)  
    
    def _get_targets_single(self,
                           cls_score,
                           lanes_pred,
                           # lclc_pred,
                           gt_instances,
                          # gt_lane_adj,
                           gt_bboxes_ignore=None,
                           img_meta=None,
                           outputs_coords_ratios=None):
        '''
        (num_queries, cls_out_channels) gt_line_instances.line_points gt_line_instances.line_labels
        '''
        # img_h, img_w = img_meta['img_shape']
        # lanes_pred = lanes_pred_in.clone()
        # lanes_pred[:, 0::2] = lanes_pred_in[:, 0::2] * img_w
        # lanes_pred[:, 1::2] = lanes_pred_in[:, 1::2] * img_h
        num_bboxes = lanes_pred.size(0)
        gt_lanes = gt_instances.line_points #.flatten(1)#LineStringsOnImage-->points(mmyolo/models/data_preprocessors/line_data_preprocessor.py)
        gt_labels = gt_instances.line_labels
        #------------zcf 240312
        # # 统一预测和gt的点数为指定个数
        # if lanes_pred.shape[-1] != self.points_for_lossMetric*self.pts_dim:
        #     # print('self.code_size',self.code_size)
        #     # print('lanes_pred',lanes_pred.shape)
        #     lanes_pred = prepare_line_points(lanes_pred.reshape(-1,self.lineRefPts,self.pts_dim),self.line_points_inter_method,self.points_for_lossMetric).flatten(1) #[num_lanes,num_points,pts_dim]
        # if gt_lanes.shape[-2]*gt_lanes.shape[-1] != self.points_for_lossMetric*self.pts_dim:
        #     gt_lanes = prepare_line_points(gt_lanes,self.line_points_inter_method,self.points_for_lossMetric)
        #------------zcf 240312
        # 统一预测和gt的点数为指定个数
        if gt_lanes.shape[-2]!= self.points_for_lossMetric:
            if outputs_coords_ratios is not None and outputs_coords_ratios!=[None]:
                #[num_gt,2,2]*[num_queries,num_points]
                #[num_gt,2]*[num_queries,num_points](num_queries,1,num_points,1)-->[num_queries,num_gt,num_points,2]
                start_point = gt_lanes[:,0].unsqueeze(0).unsqueeze(-2) # [1,num_gt,1,2]
                end_point = gt_lanes[:,1].unsqueeze(0).unsqueeze(-2)  # [1,num_gt,1,2]
                gt_lanes = start_point + (end_point - start_point) * outputs_coords_ratios.unsqueeze(-1).unsqueeze(1) # ratios, [num_queries,num_points]
                gt_lanes = gt_lanes.flatten(-2).permute(1,0,2) 
            else:
                gt_lanes = prepare_line_points(gt_lanes,self.line_points_inter_method,self.points_for_lossMetric)
                gt_lanes = gt_lanes.flatten(-2)
        else:
            gt_lanes = gt_lanes.flatten(-2)
        # assigner and sampler
        if len(gt_lanes.shape) ==2:
            pred_instances = InstanceData(scores=cls_score, lines=lanes_pred, bboxes=lanes_pred)
        elif len(gt_lanes.shape) ==3: #==3是什么情况？怎么没有说明
            pred_instances = InstanceData(scores=cls_score, bboxes=lanes_pred.unsqueeze(1))
        gt_instances = InstanceData(labels=gt_labels, lines=gt_lanes, bboxes=gt_lanes) 
        #-----------找到bug
        has_nan = torch.isnan(pred_instances.scores).any() or torch.isnan(pred_instances.bboxes).any()
        if has_nan:
            print('pred_instances.scores: ',pred_instances.scores)
            print('pred_instances.bboxes: ',pred_instances.bboxes)
            print('img_meta: ',img_meta)
        assign_result = self.assigner.assign(pred_instances,gt_instances,img_meta)
        if self.sampler:
            pred_instances.priors = gt_labels.new_zeros(0) #TODO! for PseudoSampler
            sampling_result = self.sampler.sample(assign_result, pred_instances, gt_instances) 
            pos_inds = sampling_result.pos_inds
            neg_inds = sampling_result.neg_inds
            pos_assigned_gt_inds = sampling_result.pos_assigned_gt_inds
        else:
            pos_inds = torch.nonzero(assign_result.gt_inds > 0, as_tuple=False).squeeze(-1).unique()
            neg_inds = torch.nonzero(assign_result.gt_inds == 0, as_tuple=False).squeeze(-1).unique()
            pos_assigned_gt_inds = assign_result.gt_inds[pos_inds] - 1
        
        # label targets
        labels = gt_lanes.new_full((num_bboxes, ), self.num_classes, dtype=torch.long)
        labels[pos_inds] = gt_labels[pos_assigned_gt_inds]
        # label_weights = gt_lanes.new_ones(num_bboxes)
        label_weights = gt_lanes.new_ones(num_bboxes)

        # bbox targets
        gt_c = gt_lanes.shape[-1]
        # if gt_c == 0:
        #     gt_c = self.code_size
        # else:
        #     self.code_size = gt_c
        bbox_targets = torch.zeros_like(lanes_pred)[..., :gt_c]
        bbox_weights = torch.zeros_like(lanes_pred)
        # bbox_weights[pos_inds] = 1.0
        bbox_weights[pos_inds,:] = torch.full_like(bbox_weights[pos_inds, :], 1.0) #for 4090 deterministic
        if len(gt_lanes.shape) ==2:
            bbox_targets[pos_inds] = gt_lanes[pos_assigned_gt_inds.long(), :][..., :gt_c]
        elif len(gt_lanes.shape) ==3:
            bbox_targets[pos_inds] = gt_lanes[pos_assigned_gt_inds.long(),pos_inds, :][..., :gt_c]

        return (labels, label_weights, bbox_targets, bbox_weights,
                pos_inds, neg_inds, pos_assigned_gt_inds)
    
    def forward(self,x,batch_data_samples):
        # 统一多尺度特征的通道数目
        if self.input_proj:
            x = tuple([model(feature) for model, feature in zip(self.input_proj, x)])
        head_inputs_dict = self.forward_transformer(x, batch_data_samples)
        # 1.2. None, None
        # 3.decoder输出的query特征(num_decoder_layers, num_queries, bs, embed_dims)
        # 4.decoder输入的reference_points，(bs, num_queries, 2) ,x,y
        # 5.decoder修正过的reference_points，(bs, num_queries, 2) , 因为不使用with_box_refine，目前来说，所有的reference_points都一样
        '''
        ==copy from TopoNet/projects/toponet/models/dense_heads/toponet_head.py, 每个query对应一个点，这个点存在code_size//2种偏移，因此每个query可以得到code_size//2个点
        toponet: 初始的references是query的位置编码+linear
        TopoMLP-master/projects/topomlp/models/heads/lane_head.py: references是可学习嵌入，6层都只优化这个，但是query的位置编码=references+余弦编码+Linear
        反正TopoNet：query的位置编码（简单）--推--》初始的references（复杂），中间层不修正的references，CustomMSDeformableAttention使用，看一下，decoder的交叉注意力CustomMSDeformableAttention使用reference_points和普通的MultiScaleDeformableAttention基本是一样的。
        TopoMLP：初始的references（简单）--推--》query的位置编码（复杂），中间层不修正的references，decoder不使用reference_points
        '''
        hs, init_reference, inter_references = head_inputs_dict['hidden_states'], head_inputs_dict['references'][0], head_inputs_dict['references'][1:]
        #hs = hs.permute(0, 2, 1, 3)  #历史遗留问题，mmdet 3.x目前输入和输出,bs都在num_queries之前
        #hs=(num_decoder_layers, bs, num_queries, embed_dims)
        
        factors = []
        for img_meta in batch_data_samples:
            img_h, img_w = img_meta.metainfo['img_shape'] 
            factor = hs.new_tensor([img_w, img_h]).unsqueeze(0)
            factors.append(factor)
        factors = torch.cat(factors, 0)
        expanded_factors = factors.unsqueeze(1).unsqueeze(2)
        
        outputs_classes = []
        outputs_coords,outputs_coords_ratios = [], []
        for lvl in range(hs.shape[0]):
            if lvl == 0:
                reference = init_reference
            else:
                reference = inter_references[lvl - 1]
            reference = inverse_sigmoid(reference) #reference=(bs,num_queries, lineRefPts, 2)
            outputs_class = self.cls_branches[lvl](hs[lvl])
            tmp_ori = self.reg_branches[lvl](hs[lvl])
            assert reference.shape[-1] == self.pts_dim #2

            bs, num_query, _ = tmp_ori.shape
            lineRefPts, pts_dim = reference.shape[-2], reference.shape[-1]
            tmp = tmp_ori[...,:lineRefPts*pts_dim].view(bs, num_query, lineRefPts, -1)  #2
            tmp = tmp + reference
            if self.fineLength:
                tmp = correct_reference_points_by_length_scaling(tmp,F.relu(tmp_ori[...,-1]))
            if self.correctAngle:
                if self.correctAngleSigFirst:
                    tmp = tmp.sigmoid()
                tmp = correct_lines_batch_origin(tmp, tmp_ori[...,self.correctAngleIndex].unsqueeze(-1))  #[bs,num_query,lineRefPts,2],[bs,num_query,1]
                if self.correctAngleSigFirst:
                    tmp = tmp.clamp_(min=0, max=1)
                else:
                    tmp = tmp.sigmoid()
            else:
                tmp = tmp.sigmoid()
            # 统一预测和gt的点数为指定个数
            if self.training:
                if tmp.shape[-2] != self.points_for_lossMetric:
                    if getattr(self.decoder, 'APGN_layers', None) is not None:
                        tmp,ratios = self.decoder.APGN_layers[lvl+1](hs[lvl],tmp)
                        outputs_coords_ratios.append(ratios)
                    else:
                        bs, num_queries = tmp.shape[0], tmp.shape[1]
                        tmp = prepare_line_points(tmp.view(-1, *tmp.shape[-2:]),self.line_points_inter_method,self.points_for_lossMetric) #[bs, num_query,num_points,pts_dim]
                        tmp = tmp.view(bs, num_queries, *tmp.shape[-2:])
                        outputs_coords_ratios.append([[None]]*bs)
                else:
                    outputs_coords_ratios.append([[None]]*bs)
            coord = tmp.clone()
            coord = coord*expanded_factors
            # coord[..., 0] = coord[..., 0] * (self.pc_range[3] - self.pc_range[0]) + self.pc_range[0]
            # coord[..., 1] = coord[..., 1] * (self.pc_range[4] - self.pc_range[1]) + self.pc_range[1]
            # if self.pts_dim == 3:
            #     coord[..., 2] = coord[..., 2] * (self.pc_range[5] - self.pc_range[2]) + self.pc_range[2]
            outputs_coord = coord.contiguous().view(bs, num_query, -1)

            outputs_classes.append(outputs_class)
            outputs_coords.append(outputs_coord)

        outputs_classes = torch.stack(outputs_classes)
        outputs_coords = torch.stack(outputs_coords)
        if self.training and getattr(self.decoder, 'APGN_layers', None) is not None:
            outputs_coords_ratios = torch.stack(outputs_coords_ratios)
        # print('outputs_coords',outputs_coords)
        if 'enc_outputs_class' in head_inputs_dict.keys():
            return outputs_classes, outputs_coords, hs, outputs_coords_ratios, head_inputs_dict['query_pos'], head_inputs_dict['enc_outputs_class'], head_inputs_dict['enc_outputs_coord']
        else:
            return outputs_classes, outputs_coords, hs, outputs_coords_ratios, head_inputs_dict['query_pos']
        #(num_decoder_layers,bs,num_queries,num_cls),#(num_decoder_layers,bs,num_queries,2)#(num_decoder_layers,bs,num_queries,embed_dims)
    #   copy from mmdet/models/detectors/base_detr.py
    def forward_transformer(self,
                            img_feats: Tuple[Tensor],
                            batch_data_samples: OptSampleList = None) -> Dict:
        """
        Args:
            img_feats (tuple[Tensor]): Tuple of feature maps from neck. Each
                    feature map has shape (bs, dim, H, W).
            batch_data_samples (list[:obj:`DetDataSample`], optional): The
                batch data samples. It usually includes information such
                as `gt_instance` or `gt_panoptic_seg` or `gt_sem_seg`.
                Defaults to None.

        Returns:
            dict: The dictionary of bbox_head function inputs, which always
            includes the `hidden_states` of the decoder output and may contain
            `references` including the initial and intermediate references.
        """
        encoder_inputs_dict, decoder_inputs_dict = self.pre_transformer(
            img_feats, batch_data_samples)
        if self.encoder: #-----zhou 也许不需要图像编码
            encoder_outputs_dict = self.forward_encoder(**encoder_inputs_dict)
        else:
            encoder_outputs_dict = dict(memory=encoder_inputs_dict['feat'], 
                                        memory_mask=encoder_inputs_dict['feat_mask'], spatial_shapes=encoder_inputs_dict['spatial_shapes'])

        tmp_dec_in, head_inputs_dict = self.pre_decoder(**encoder_outputs_dict)
        decoder_inputs_dict.update(tmp_dec_in)

        decoder_outputs_dict = self.forward_decoder(**decoder_inputs_dict) 
        head_inputs_dict.update(decoder_outputs_dict) 
        return head_inputs_dict
        # 1.2. None, None
        # 3.decoder输出的query特征(num_decoder_layers, num_queries, bs, embed_dims)
        # 4.decoder输入的reference_points，(bs, num_queries, 2)
        # 5.decoder修正过的reference_points，(bs, num_queries, 2) , 因为不使用with_box_refine，目前来说，所有的reference_points都一样

    # copy from mmdet/models/detectors/deformable_detr.py
    def pre_transformer(
            self,
            mlvl_feats: Tuple[Tensor],
            batch_data_samples: OptSampleList = None) -> Tuple[Dict]:
        """Process image features before feeding them to the transformer.

        The forward procedure of the transformer is defined as:
        'pre_transformer' -> 'encoder' -> 'pre_decoder' -> 'decoder'
        More details can be found at `TransformerDetector.forward_transformer`
        in `mmdet/detector/base_detr.py`.

        Args:
            mlvl_feats (tuple[Tensor]): Multi-level features that may have
                different resolutions, output from neck. Each feature has
                shape (bs, dim, h_lvl, w_lvl), where 'lvl' means 'layer'.
            batch_data_samples (list[:obj:`DetDataSample`], optional): The
                batch data samples. It usually includes information such
                as `gt_instance` or `gt_panoptic_seg` or `gt_sem_seg`.
                Defaults to None.

        Returns:
            tuple[dict]: The first dict contains the inputs of encoder and the
            second dict contains the inputs of decoder.

            - encoder_inputs_dict (dict): The keyword args dictionary of
              `self.forward_encoder()`, which includes 'feat', 'feat_mask',
              and 'feat_pos'.
            - decoder_inputs_dict (dict): The keyword args dictionary of
              `self.forward_decoder()`, which includes 'memory_mask'.
        """
        batch_size = mlvl_feats[0].size(0)

        # construct binary masks for the transformer.
        assert batch_data_samples is not None
        batch_input_shape = batch_data_samples[0].batch_input_shape
        img_shape_list = [sample.img_shape for sample in batch_data_samples]
        input_img_h, input_img_w = batch_input_shape
        masks = mlvl_feats[0].new_ones((batch_size, input_img_h, input_img_w))
        for img_id in range(batch_size):
            img_h, img_w = img_shape_list[img_id]
            masks[img_id, :img_h, :img_w] = 0
        # NOTE following the official DETR repo, non-zero values representing
        # ignored positions, while zero values means valid positions.

        mlvl_masks = []
        mlvl_pos_embeds = []
        for feat in mlvl_feats:
            mlvl_masks.append(
                F.interpolate(masks[None],
                              size=feat.shape[-2:]).to(torch.bool).squeeze(0))
            mlvl_pos_embeds.append(self.positional_encoding(mlvl_masks[-1]))

        feat_flatten = []
        lvl_pos_embed_flatten = []
        mask_flatten = []
        spatial_shapes = []
        for lvl, (feat, mask, pos_embed) in enumerate(
                zip(mlvl_feats, mlvl_masks, mlvl_pos_embeds)):
            batch_size, c, h, w = feat.shape
            # [bs, c, h_lvl, w_lvl] -> [bs, h_lvl*w_lvl, c]
            feat = feat.view(batch_size, c, -1).permute(0, 2, 1)
            pos_embed = pos_embed.view(batch_size, c, -1).permute(0, 2, 1)
            if self.encoder is not None:
                lvl_pos_embed = pos_embed + self.level_embed[lvl].view(1, 1, -1)
            # [bs, h_lvl, w_lvl] -> [bs, h_lvl*w_lvl]
            mask = mask.flatten(1)
            spatial_shape = (h, w)

            feat_flatten.append(feat)
            if self.encoder is not None:
                lvl_pos_embed_flatten.append(lvl_pos_embed)
            mask_flatten.append(mask)
            spatial_shapes.append(spatial_shape)

        # (bs, num_feat_points, dim)
        feat_flatten = torch.cat(feat_flatten, 1)
        if self.encoder is not None:
            lvl_pos_embed_flatten = torch.cat(lvl_pos_embed_flatten, 1)
        # (bs, num_feat_points), where num_feat_points = sum_lvl(h_lvl*w_lvl)
        mask_flatten = torch.cat(mask_flatten, 1)

        spatial_shapes = torch.as_tensor(  # (num_level, 2)
            spatial_shapes,
            dtype=torch.long,
            device=feat_flatten.device)
        level_start_index = torch.cat((
            spatial_shapes.new_zeros((1, )),  # (num_level)
            spatial_shapes.prod(1).cumsum(0)[:-1]))
        valid_ratios = torch.stack(  # (bs, num_level, 2)
            [self.get_valid_ratio(m) for m in mlvl_masks], 1)

        encoder_inputs_dict = dict(
            feat=feat_flatten,
            feat_mask=mask_flatten,
            feat_pos=lvl_pos_embed_flatten,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
            valid_ratios=valid_ratios)
        decoder_inputs_dict = dict(
            memory_mask=mask_flatten,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
            valid_ratios=valid_ratios)
        return encoder_inputs_dict, decoder_inputs_dict
    
    # copy from mmdet/models/detectors/deformable_detr.py
    def forward_encoder(self, feat: Tensor, feat_mask: Tensor,
                        feat_pos: Tensor, spatial_shapes: Tensor,
                        level_start_index: Tensor,
                        valid_ratios: Tensor) -> Dict:
        """Forward with Transformer encoder.

        The forward procedure of the transformer is defined as:
        'pre_transformer' -> 'encoder' -> 'pre_decoder' -> 'decoder'
        More details can be found at `TransformerDetector.forward_transformer`
        in `mmdet/detector/base_detr.py`.

        Args:
            feat (Tensor): Sequential features, has shape (bs, num_feat_points,
                dim).
            feat_mask (Tensor): ByteTensor, the padding mask of the features,
                has shape (bs, num_feat_points).
            feat_pos (Tensor): The positional embeddings of the features, has
                shape (bs, num_feat_points, dim).
            spatial_shapes (Tensor): Spatial shapes of features in all levels,
                has shape (num_levels, 2), last dimension represents (h, w).
            level_start_index (Tensor): The start index of each level.
                A tensor has shape (num_levels, ) and can be represented
                as [0, h_0*w_0, h_0*w_0+h_1*w_1, ...].
            valid_ratios (Tensor): The ratios of the valid width and the valid
                height relative to the width and the height of features in all
                levels, has shape (bs, num_levels, 2).

        Returns:
            dict: The dictionary of encoder outputs, which includes the
            `memory` of the encoder output.
        """
        memory = self.encoder(
            query=feat,
            query_pos=feat_pos,
            key_padding_mask=feat_mask,  # for self_attn
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
            valid_ratios=valid_ratios)
        encoder_outputs_dict = dict(
            memory=memory,
            memory_mask=feat_mask,
            spatial_shapes=spatial_shapes)
        return encoder_outputs_dict
    
    # copy from mmdet/models/detectors/deformable_detr.py
    def forward_decoder(self, query: Tensor, query_pos: Tensor, memory: Tensor,
                        memory_mask: Tensor, reference_points: Tensor,
                        spatial_shapes: Tensor, level_start_index: Tensor,
                        valid_ratios: Tensor) -> Dict:
        """Forward with Transformer decoder.

        The forward procedure of the transformer is defined as:
        'pre_transformer' -> 'encoder' -> 'pre_decoder' -> 'decoder'
        More details can be found at `TransformerDetector.forward_transformer`
        in `mmdet/detector/base_detr.py`.

        Args:
            query (Tensor): The queries of decoder inputs, has shape
                (bs, num_queries, dim).
            query_pos (Tensor): The positional queries of decoder inputs,
                has shape (bs, num_queries, dim).
            memory (Tensor): The output embeddings of the Transformer encoder,
                has shape (bs, num_feat_points, dim).
            memory_mask (Tensor): ByteTensor, the padding mask of the memory,
                has shape (bs, num_feat_points).
            reference_points (Tensor): The initial reference, has shape
                (bs, num_queries, 4) with the last dimension arranged as
                (cx, cy, w, h) when `as_two_stage` is `True`, otherwise has
                shape (bs, num_queries, 2) with the last dimension arranged as
                (cx, cy).
            spatial_shapes (Tensor): Spatial shapes of features in all levels,
                has shape (num_levels, 2), last dimension represents (h, w).
            level_start_index (Tensor): The start index of each level.
                A tensor has shape (num_levels, ) and can be represented
                as [0, h_0*w_0, h_0*w_0+h_1*w_1, ...].
            valid_ratios (Tensor): The ratios of the valid width and the valid
                height relative to the width and the height of features in all
                levels, has shape (bs, num_levels, 2).

        Returns:
            dict: The dictionary of decoder outputs, which includes the
            `hidden_states` of the decoder output and `references` including
            the initial and intermediate reference_points.
        """
        bs, num_query, _ = reference_points.shape
        reference_points = reference_points.view(bs, num_query, self.lineRefPts, -1) 
        inter_states, inter_references, query_pos = self.decoder(  #-------------240607
            query=query,
            value=memory,
            query_pos=query_pos,
            key_padding_mask=memory_mask,  # for cross_attn
            reference_points=reference_points,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
            valid_ratios=valid_ratios,
            reg_branches=self.reg_branches # zhou self.bbox_head.reg_branches-->self.reg_branches
            if self.with_box_refine else None,
            fineLength = self.fineLength)
        references = [reference_points, *inter_references]
        decoder_outputs_dict = dict(
            hidden_states=inter_states, references=references, query_pos=query_pos) #-------------240607
        return decoder_outputs_dict
    # copy from mmdet/models/detectors/deformable_detr.py
    @staticmethod
    def get_valid_ratio(mask: Tensor) -> Tensor:
        """Get the valid radios of feature map in a level.

        .. code:: text

                    |---> valid_W <---|
                 ---+-----------------+-----+---
                  A |                 |     | A
                  | |                 |     | |
                  | |                 |     | |
            valid_H |                 |     | |
                  | |                 |     | H
                  | |                 |     | |
                  V |                 |     | |
                 ---+-----------------+     | |
                    |                       | V
                    +-----------------------+---
                    |---------> W <---------|

          The valid_ratios are defined as:
                r_h = valid_H / H,  r_w = valid_W / W
          They are the factors to re-normalize the relative coordinates of the
          image to the relative coordinates of the current level feature map.

        Args:
            mask (Tensor): Binary mask of a feature map, has shape (bs, H, W).

        Returns:
            Tensor: valid ratios [r_w, r_h] of a feature map, has shape (1, 2).
        """
        _, H, W = mask.shape
        valid_H = torch.sum(~mask[:, :, 0], 1)
        valid_W = torch.sum(~mask[:, 0, :], 1)
        valid_ratio_h = valid_H.float() / H
        valid_ratio_w = valid_W.float() / W
        valid_ratio = torch.stack([valid_ratio_w, valid_ratio_h], -1)
        return valid_ratio
    # copy from mmdet/models/detectors/deformable_detr.py
    def pre_decoder(self, memory: Tensor, memory_mask: Tensor,
                    spatial_shapes: Tensor) -> Tuple[Dict, Dict]:
        batch_size, _, c = memory.shape
        #----------zcf self.as_two_stage 240520----------step 1 
        if self.as_two_stage:
            cls_out_features = self.cls_branches[
                self.decoder.num_layers][-1].out_features #---------change for lineHead Linear.out_features
    
            #---------0529----------step 1 
            output_memory, output_proposals = self.gen_encoder_output_proposals(
                memory, memory_mask, spatial_shapes) #output_proposals [batch_size, sum(wh), 2]---[cx,cy]
            # output_memory, output_proposals = self.gen_encoder_output_proposals_v2(
            #     memory, memory_mask, spatial_shapes) #output_proposals [batch_size, sum(wh), 2]---[cx,cy]
            
            enc_outputs_class = self.cls_branches[
                self.decoder.num_layers](
                    output_memory)
            #---------lineRefPts 250322----------
            # reg_out_dims = self.reg_branches[self.decoder.num_layers][-1].out_features
            # if reg_out_dims != output_proposals.shape[-1]:
            #     output_proposals = output_proposals.repeat(1, 1, reg_out_dims//output_proposals.shape[-1])
            #---------lineRefPts 250322----------
            #---------cxcywh2xyxy 0530----------step 2 
            # output_proposals = bbox_cxcywh_to_xyxy(output_proposals) #[batch_size, sum(wh), 4]
            #---------0529----------step 2
            enc_outputs_coord_unact = self.reg_branches[
                self.decoder.num_layers](output_memory) + output_proposals
            # # 获取decoder的最后一层输出
            # reg_output = self.reg_branches[self.decoder.num_layers](output_memory)
            # # 直接进行索引与操作以生成结果
            # enc_outputs_coord_unact = torch.cat([output_proposals[..., 2:]-reg_output[...,  2:], reg_output[..., :2]+output_proposals[..., :2]], dim=-1)
            #---------0529----------step 3
            # 使用clamp函数确保所有输出值保持在[0, 1]区间内
            # enc_outputs_coord_unact = enc_outputs_coord_unact.clamp(0, 1)
            
            # NOTE The DINO selects top-k proposals according to scores of
            # multi-class classification, while DeformDETR, where the input
            # is `enc_outputs_class[..., 0]` selects according to scores of
            # binary classification.
            topk_indices = torch.topk(
                enc_outputs_class.max(-1)[0], k=self.num_queries, dim=1)[1]
            topk_score = torch.gather(
                enc_outputs_class, 1,
                topk_indices.unsqueeze(-1).repeat(1, 1, cls_out_features))
            topk_coords_unact = torch.gather(
                enc_outputs_coord_unact, 1,
                topk_indices.unsqueeze(-1).repeat(1, 1, enc_outputs_coord_unact.shape[-1])) #---------lineRefPts 250322----------#xyxyxy...
            #---------0529----------step 3
            topk_coords = topk_coords_unact.sigmoid() 
            # topk_coords = topk_coords_unact
            
            topk_coords_unact = topk_coords_unact.detach()
            # opt 1: dino
            query = self.query_embedding.weight[:, None, :] 
            query = query.repeat(1, batch_size, 1).transpose(0, 1) #[bs,num_queries,dim]
            # opt 1: rtdetr
            # query = torch.gather(output_memory, 1,
            #                   topk_indices.unsqueeze(-1).repeat(1, 1, c))
            # query = query.detach()  # detach() is not used in DINO
            
            reference_points = topk_coords_unact
            reference_points = reference_points.sigmoid() #---------0529----------step 3
            #----------zcf self.as_two_stage 240520----------step 2, query_pos is move to self.decoder
            query_pos = None
            bs, num_queries, num_pts = topk_coords.shape[0], topk_coords.shape[1], topk_coords.shape[2]//self.pts_dim
            if topk_coords.shape[-1] != self.points_for_lossMetric*self.pts_dim:
                topk_coords = prepare_line_points(topk_coords.view(-1, num_pts, self.pts_dim),self.line_points_inter_method,self.points_for_lossMetric) #[bs, num_query,num_points,pts_dim]
            topk_coords = topk_coords.view(bs, num_queries, -1, self.pts_dim)
        else:
            topk_score, topk_coords = None, None
            query_embed = self.query_embedding.weight
            query_pos, query = torch.split(query_embed, c, dim=1)
            query_pos = query_pos.unsqueeze(0).expand(batch_size, -1, -1)
            query = query.unsqueeze(0).expand(batch_size, -1, -1)
            reference_points = self.reference_points_fc(query_pos).sigmoid()

        decoder_inputs_dict = dict(
            query=query,
            query_pos=query_pos,
            memory=memory,
            reference_points=reference_points)
        head_inputs_dict = dict(
            enc_outputs_class=topk_score,
            enc_outputs_coord=topk_coords) if self.training else dict()
        return decoder_inputs_dict, head_inputs_dict
    # copy from mmdet/models/detectors/deformable_detr.py
    def gen_encoder_output_proposals(
            self, memory: Tensor, memory_mask: Tensor,
            spatial_shapes: Tensor) -> Tuple[Tensor, Tensor]:
        
        output_memory = memory
        if memory_mask is not None:
            output_memory = output_memory.masked_fill(
                memory_mask.unsqueeze(-1), float(0))
        # output_memory = output_memory.masked_fill(~output_proposals_valid,
        #                                           float(0))
        output_memory = self.memory_trans_fc(output_memory)
        output_memory = self.memory_trans_norm(output_memory)
        bs = memory.size(0)
        
        line_directions_all = self.Fea_LineClassifier(output_memory).sigmoid().view(bs, -1)  #[bs,sum(wh)]
        
        proposals = []
        _cur = 0  # start index in the sequence of the current level
        for lvl, HW in enumerate(spatial_shapes):
            H, W = HW

            if memory_mask is not None:
                mask_flatten_ = memory_mask[:, _cur:(_cur + H * W)].view(
                    bs, H, W, 1)
                valid_H = torch.sum(~mask_flatten_[:, :, 0, 0],
                                    1).unsqueeze(-1)
                valid_W = torch.sum(~mask_flatten_[:, 0, :, 0],
                                    1).unsqueeze(-1)
                scale = torch.cat([valid_W, valid_H], 1).view(bs, 1, 1, 2)
            else:
                if not isinstance(HW, torch.Tensor):
                    HW = memory.new_tensor(HW)
                scale = HW.unsqueeze(0).flip(dims=[0, 1]).view(1, 1, 1, 2)
            grid_y, grid_x = torch.meshgrid(
                torch.linspace(
                    0, H - 1, H, dtype=torch.float32, device=memory.device),
                torch.linspace(
                    0, W - 1, W, dtype=torch.float32, device=memory.device))
            grid = torch.cat([grid_x.unsqueeze(-1), grid_y.unsqueeze(-1)], -1)
            grid = (grid.unsqueeze(0).expand(bs, -1, -1, -1) + 0.5) / scale
            # wh = torch.ones_like(grid) * 0.05 * (2.0**lvl)
            grid = grid.view(bs, -1, 2)  # [bs, H*W, 2]
            wh = torch.ones((bs, grid.shape[1]), device=grid.device) * 0.25 * (2.0**lvl) #TODO, 0.05--->0.25
            # print('wh',wh)
            # Generate random line directions (0 for horizontal, 1 for vertical)
            #----------------------------240619
            # line_directions = torch.randint(0, 2, (bs, grid.shape[1]), device=grid.device)   
            #----------------------------240619, 可以解决训练过程中的推理 and 加载模型单独推理 两者结果的不同
            # line_directions = torch.zeros((bs, grid.shape[1]), device=grid.device, dtype=torch.long)
            # line_directions[:, 1::2] = 1  # 奇数索引为水平方向=1,改变y, 偶数索引为垂直方向
            #----------------------------240619
            line_directions = line_directions_all[:, _cur:(_cur + H * W)]
            # Initialize proposals tensor
            proposal = grid.repeat(1, 1, 2)  # duplicate grid along last dimension
            # Compute proposals for horizontal (0) and vertical lines (1)
            # proposal[..., 0] = torch.where(line_directions == 0, grid[..., 0] - wh / 2, grid[..., 0])
            # proposal[..., 1] = torch.where(line_directions == 1, grid[..., 1] - wh / 2, grid[..., 1])
            # # Adjust the other ends of the lines
            # proposal[..., 2] = torch.where(line_directions == 0, grid[..., 0] + wh / 2, grid[..., 0])
            # proposal[..., 3] = torch.where(line_directions == 1, grid[..., 1] + wh / 2, grid[..., 1])
            proposal[..., 0] = grid[..., 0] - wh / 2 * line_directions
            proposal[..., 1] = grid[..., 1] - wh / 2 * (1 - line_directions)
            # Adjust the other ends of the lines
            proposal[..., 2] = grid[..., 0] + wh / 2 * line_directions
            proposal[..., 3] = grid[..., 1] + wh / 2 * (1 - line_directions)
            # proposal = torch.cat((grid, wh), -1).view(bs, -1, 4)
            proposals.append(proposal)
            _cur += (H * W)
        output_proposals = torch.cat(proposals, 1) #[bs, (H*W), 4]
        # do not use `all` to make it exportable to onnx
        # output_proposals_valid = (
        #     (output_proposals > 0.01) & (output_proposals < 0.99)).sum(
        #         -1, keepdim=True) == output_proposals.shape[-1]
        #---------lineRefPts 250322----------same as reference_points 边框分支xyxyxy....在sigmoid之后采样，产生多个点; 在sigmoid之前相加，进行修正
        if output_proposals.shape[-1] !=self.lineRefPts*self.pts_dim:
            bs,num_pro,num_pointsDim = output_proposals.shape
            output_proposals = prepare_line_points(output_proposals.view(-1, num_pointsDim//self.pts_dim, self.pts_dim),self.line_points_inter_method,self.lineRefPts) #[bs, num_query,num_points,pts_dim]
            output_proposals = output_proposals.reshape(bs, num_pro, -1)
        #---------lineRefPts 250322----------
        # inverse_sigmoid
        output_proposals = inverse_sigmoid(output_proposals)
        # output_proposals = torch.log(output_proposals / (1 - output_proposals))
        if memory_mask is not None:
            output_proposals = output_proposals.masked_fill(
                memory_mask.unsqueeze(-1), float('inf'))
        # output_proposals = output_proposals.masked_fill(
        #     ~output_proposals_valid, float('inf'))

        # [bs, sum(hw), 2]
        return output_memory, output_proposals  


