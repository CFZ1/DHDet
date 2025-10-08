
"""
Latest version 2025-04-26-17:00:00
Created on Sun Jan 14 20:52:30 2024

@author: zcf

change based on Ours_e60_65d5a87d4
"""
import torch
from mmdet.models.detectors.single_stage import SingleStageDetector
from mmdet.utils import ConfigType, OptConfigType, OptMultiConfig
from mmyolo.registry import MODELS
from typing import List, Tuple, Union
from torch import Tensor
from mmdet.structures import SampleList, DetDataSample
from mmdet.utils import InstanceList
from mmdet.models.utils import samplelist_boxtype2tensor
from mmdet.structures import OptSampleList
from typing import Dict
from mmengine.dist import get_world_size
from mmengine.logging import print_log 
import torch.nn as nn
from mmcv.cnn import Linear
from mmengine.model import xavier_init
import copy
from mmdet.models import DeformableDetrTransformerEncoder
from mmdet.models.layers import SinePositionalEncoding
import torch.nn.functional as F
from mmengine.model import BaseModule, ModuleList
from mmcv.cnn import build_norm_layer
from mmcv.cnn.bricks.transformer import FFN, MultiheadAttention
from mmcv.cnn import ConvModule
from mmdet.models import MLP
from mmdet.models.dense_heads.tood_head import TaskDecomposition
from mmcv.ops import MultiScaleDeformableAttention
ForwardResults = Union[Dict[str, torch.Tensor], List[DetDataSample],
                       Tuple[torch.Tensor], torch.Tensor]
import numpy as np
from torch.nn.init import xavier_uniform_
from mmengine.dist import all_gather, get_dist_info
import torch.distributed as dist
from ..layers import RepVGGBlock
# from mmyolo.evaluation.metrics.line_coco_metric import line2box_torch #------------------------------zcf use_tta 20240916
from mmengine.structures import InstanceData#------------------------------zcf use_tta 20240916
from mmcv.ops import batched_nms
from torchvision.ops import box_iou
lineMin = 30
def line2box_torch(line_bboxes, box_min=lineMin,img_height=None,img_width=None):
    box_min=lineMin #------------for: test.py change lineMin
    # 使用torch.min和torch.max替换np.min和np.max
    min_x = torch.min(line_bboxes[:, 0::2], dim=-1)[0]
    max_x = torch.max(line_bboxes[:, 0::2], dim=-1)[0]
    min_y = torch.min(line_bboxes[:, 1::2], dim=-1)[0]
    max_y = torch.max(line_bboxes[:, 1::2], dim=-1)[0]  
    # Calculate center (仍然是使用Torch操作)
    center_x = (max_x + min_x) / 2
    center_y = (max_y + min_y) / 2 
    # Adjust width to be at least `box_min`
    # 使用torch.where代替np.where
    min_x_update = torch.where(max_x - min_x < box_min, center_x - box_min / 2, min_x)
    max_x_update = torch.where(max_x - min_x < box_min, center_x + box_min / 2, max_x)  
    # Adjust height to be at least `box_min`
    min_y_update = torch.where(max_y - min_y < box_min, center_y - box_min / 2, min_y)
    max_y_update = torch.where(max_y - min_y < box_min, center_y + box_min / 2, max_y) 
    #----------------adjust_range, 处理超出图像边界的情况，同时保持box_min
    # # 如果左边界超出，先调整左边界，再相应调整右边界
    # min_x_update = torch.clamp(min_x_update, min=0)
    # max_x_update = torch.maximum(max_x_update, min_x_update + box_min)
    # if img_width is not None:
    #     # 如果右边界超出，先调整右边界，再相应调整左边界
    #     max_x_update = torch.clamp(max_x_update, max=img_width)
    #     min_x_update = torch.minimum(min_x_update, max_x_update - box_min)
    # # 如果上边界超出，先调整上边界，再相应调整下边界
    # min_y_update = torch.clamp(min_y_update, min=0)
    # max_y_update = torch.maximum(max_y_update, min_y_update + box_min)
    # if img_height is not None:
    #     # 如果下边界超出，先调整下边界，再相应调整上边界
    #     max_y_update = torch.clamp(max_y_update, max=img_height)
    #     min_y_update = torch.minimum(min_y_update, max_y_update - box_min)
    #----------------adjust_range
    # Concatenate results [minX, minY, maxX, maxY] 使用torch.stack
    results = torch.stack((min_x_update, min_y_update, max_x_update, max_y_update), dim=-1)    
    return results
#--------copy from https://github.com/icoz69/CEC-CVPR2021 and change
# clear 240815
class MultiHeadAttentionModify(nn.Module):
    ''' Multi-Head Attention module '''

    def __init__(self, d_model, NumLinePoints=2): #-----240815_NumLinePoints
        super().__init__()
        self.embed_dims = d_model
        self.w_vs = nn.Linear( self.embed_dims,  self.embed_dims, bias=False)
        xavier_uniform_(self.w_vs.weight)
        self.ref_point_head = MLP(self.embed_dims*2, self.embed_dims,self.embed_dims, 2) #为了共享参数
        self.fuse_conv3d3SiLU1d1 = nn.Sequential(
            ConvModule(
                self.embed_dims,
                self.embed_dims,
                3,
                padding=1,
                norm_cfg=dict(type='BN', momentum=0.03, eps=0.001),
                act_cfg=dict(type='SiLU', inplace=True)),
            ConvModule(
                self.embed_dims,
                self.embed_dims,
                1,
                norm_cfg=None,
                act_cfg=None))
        self.NumLinePoints = NumLinePoints #-----240815_NumLinePoints
    def forward(self, query, v, k=None, query_pos=None, key_pos=None, other_W=None): #other_W=[bs,num_k]
        v = self.w_vs(v)
        
        query_pos = query_pos.repeat(1,1,self.NumLinePoints) #为了共享参数 #-----240815_NumLinePoints #[bs,num_query,points_dim*c]--[bs,num_query,(points_dim*c)*num_points]
        query_pos = self.ref_point_head(query_pos)
        query_norm = query_pos / query_pos.norm(dim=-1, keepdim=True)
        key_norm = key_pos / key_pos.norm(dim=-1, keepdim=True)
        relative_pos = torch.bmm(query_norm, key_norm.transpose(1, 2))
        relative_pos = F.relu(relative_pos)
        if other_W is not None:
            relative_pos = relative_pos* other_W.unsqueeze(1) #[bs,num_q,num_k]    
        output = torch.bmm(relative_pos, v) #[bs,num_q,dim]
        
        output = output.permute(0, 2, 1).view(query.shape).contiguous() #[bs, sum(H,W), 256]--->[bs, 256, H, W]
        output = self.fuse_conv3d3SiLU1d1(output)
        output = query + output
        return output  
# copy from mmdet/models/layers/transformer/detr_layers.py: DetrTransformerDecoderLayer,只保留交叉注意力
class DetrTransformerDecoderRemoveSelfLayer(BaseModule):
    """Implements decoder layer in DETR transformer.

    Args:
        cross_attn_cfg (:obj:`ConfigDict` or dict, optional): Config for cross
            attention.
        ffn_cfg (:obj:`ConfigDict` or dict, optional): Config for FFN.
        norm_cfg (:obj:`ConfigDict` or dict, optional): Config for
            normalization layers. All the layers will share the same
            config. Defaults to `LN`.
        init_cfg (:obj:`ConfigDict` or dict, optional): Config to control
            the initialization. Defaults to None.
    """

    def __init__(self,
                 cross_attn_cfg: OptConfigType = dict(
                     embed_dims=256,
                     num_heads=8,
                     dropout=0.0,
                     batch_first=True),
                 ffn_cfg: OptConfigType = dict(
                     embed_dims=256,
                     feedforward_channels=1024,
                     num_fcs=2,
                     ffn_drop=0.,
                     act_cfg=dict(type='ReLU', inplace=True),
                 ),
                 norm_cfg: OptConfigType = dict(type='LN'),
                 init_cfg: OptConfigType = None) -> None:

        super().__init__(init_cfg=init_cfg)
        
        self.cross_attn_cfg = cross_attn_cfg

        if 'batch_first' not in self.cross_attn_cfg:
            self.cross_attn_cfg['batch_first'] = True
        else:
            assert self.cross_attn_cfg['batch_first'] is True, 'First \
            dimension of all DETRs in mmdet is `batch`, \
            please set `batch_first` flag.'

        self.ffn_cfg = ffn_cfg
        self.norm_cfg = norm_cfg
        self._init_layers()

    def _init_layers(self) -> None:
        """Initialize self-attention, FFN, and normalization."""
        # self.cross_attn = MODELS.build(self.cross_attn_cfg) #-----------
        self.cross_attn = MultiheadAttention(**self.cross_attn_cfg)
        # self.cross_attn = MultiScaleDeformableAttention(**self.cross_attn_cfg) #因为query_features是一维的，level_start_index，reference_points，spatial_shapes代表是在query_features上的位置，而且他们代表的是二维的位置
        self.embed_dims = self.cross_attn.embed_dims
        self.ffn = FFN(**self.ffn_cfg)
        norms_list = [
            build_norm_layer(self.norm_cfg, self.embed_dims)[1]
            for _ in range(2)
        ]
        self.norms = ModuleList(norms_list)

    def forward(self,
                query: Tensor,
                key: Tensor = None,
                value: Tensor = None,
                query_pos: Tensor = None,
                key_pos: Tensor = None,
                # self_attn_mask: Tensor = None,
                cross_attn_mask: Tensor = None,
                key_padding_mask: Tensor = None,
                **kwargs) -> Tensor:
        """
        Args:
            query (Tensor): The input query, has shape (bs, num_queries, dim).
            key (Tensor, optional): The input key, has shape (bs, num_keys,
                dim). If `None`, the `query` will be used. Defaults to `None`.
            value (Tensor, optional): The input value, has the same shape as
                `key`, as in `nn.MultiheadAttention.forward`. If `None`, the
                `key` will be used. Defaults to `None`.
            query_pos (Tensor, optional): The positional encoding for `query`,
                has the same shape as `query`. If not `None`, it will be added
                to `query` before forward function. Defaults to `None`.
            key_pos (Tensor, optional): The positional encoding for `key`, has
                the same shape as `key`. If not `None`, it will be added to
                `key` before forward function. If None, and `query_pos` has the
                same shape as `key`, then `query_pos` will be used for
                `key_pos`. Defaults to None.
            cross_attn_mask (Tensor, optional): ByteTensor mask, has shape
                (num_queries, num_keys), as in `nn.MultiheadAttention.forward`.
                Defaults to None.
            key_padding_mask (Tensor, optional): The `key_padding_mask` of
                `self_attn` input. ByteTensor, has shape (bs, num_value).
                Defaults to None.

        Returns:
            Tensor: forwarded results, has shape (bs, num_queries, dim).
        """
        query = self.cross_attn(
            query=query,
            key=key,
            value=value,
            query_pos=query_pos,
            key_pos=key_pos,
            attn_mask=cross_attn_mask,
            key_padding_mask=key_padding_mask,
            **kwargs)
        query = self.norms[0](query)
        query = self.ffn(query)
        query = self.norms[1](query)

        return query


# copy from mmyolo/models/detectors/yolo_detector.py
@MODELS.register_module()
class coLineDetectorV4(SingleStageDetector):
    '''
    HighFeaRefine and line_head.return_queries
    '''
    def __init__(self,
                 backbone: ConfigType,
                 neck: ConfigType,
                 bbox_head: ConfigType,
                 train_cfg: OptConfigType = None,
                 test_cfg: OptConfigType = None,
                 data_preprocessor: OptConfigType = None,
                 init_cfg: OptMultiConfig = None,
                 use_syncbn: bool = True,
                 HighFeaRefine: bool = False, 
                 line_head: ConfigType = None):
        # if 'train_cfg' in bbox_head.keys():
        #     train_cfg = bbox_head.train_cfg
        # if 'test_cfg' in bbox_head.keys():
        #     test_cfg = bbox_head.test_cfg
        # SingleStageDetector.__init__, 希望直接跳到SingleStageDetector的上级
        super(SingleStageDetector, self).__init__(
            data_preprocessor=data_preprocessor, init_cfg=init_cfg)
        self.use_tta = False #------------------------------zcf use_tta 20240916, 默认不使用
        self.backbone = MODELS.build(backbone)
        if neck is not None:
            self.neck = MODELS.build(neck)
        # bbox_head.update(train_cfg=train_cfg)
        # bbox_head.update(test_cfg=test_cfg)
        self.bbox_head_type = bbox_head.type 
        #---------------only for BboxDeformableDETRHead
        if self.bbox_head_type in ['BboxDeformableDETRHead','BboxDINOHead']:
            share_decoder_withLineHead = bbox_head.get('share_decoder_withLineHead', {'shared': False, 'deepCopy': False})
            # bbox_head.pop('share_decoder_withLineHead')
            if share_decoder_withLineHead['shared']:
                bbox_head.decoder_num_layers = line_head.transformer.decoder.num_layers
        #---------------only for BboxDeformableDETRHead
        self.bbox_head = MODELS.build(bbox_head)
        # self.train_cfg = train_cfg
        # self.test_cfg = test_cfg
        # SingleStageDetector.__init__
        if isinstance(line_head['in_channels'], list):
            line_head['in_channels'] = [int(x) for x in line_head['in_channels']]
        self.line_head = MODELS.build(line_head) #self.neck.out_channels
        
        #---------------only for BboxDeformableDETRHead
        if self.bbox_head_type in ['BboxDeformableDETRHead','BboxDINOHead'] and share_decoder_withLineHead['shared']:
            if not share_decoder_withLineHead['deepCopy']:
                self.bbox_head.input_proj = self.line_head.input_proj
                self.bbox_head.decoder = self.line_head.decoder
            else:
                self.bbox_head.input_proj = copy.deepcopy(self.line_head.input_proj)
                self.bbox_head.decoder = copy.deepcopy(self.line_head.decoder)
        #---------------only for BboxDeformableDETRHead
        '''
        '''
        self.highFeasFuseMode = 5 #[1,2,3,4] enhance_highFeas_with_queries_v
        self.embed_dims = 256 
        
        # modify HighFeaRefine and line_head.return_queries
        self.HighFeaRefine = HighFeaRefine
        if self.HighFeaRefine:
            if self.highFeasFuseMode==5: 
                print('*************************==================self.highFeasFuseMode ', self.highFeasFuseMode)
                self.alpha = nn.Parameter(torch.ones(1))
                self.beta = nn.Parameter(torch.zeros(1))
                # self.register_buffer('running_lineBg_weight', torch.tensor([float('inf')]))
                self.register_buffer('running_lineBg_weight', torch.tensor([0.0]))
                # self.register_buffer('count', torch.tensor([0]))
                # self.current_epoch = -1  # 直接使用整数
                self.momentum = 0.1
                # self.register_buffer('current_epoch', torch.tensor([-1]))
                self.highFeas_encoder = MultiHeadAttentionModify(d_model=self.embed_dims,
                                                                 NumLinePoints=line_head['lineRefPts']) #---------240815_NumLinePoints
                # positional_encoding_cfg=dict(num_feats=128, normalize=True, offset=-0.5)
                positional_encoding_cfg=line_head.positional_encoding
                positional_encoding_cfg.pop('type')
                self.positional_encoding = SinePositionalEncoding(**positional_encoding_cfg)
                if self.line_head.as_two_stage: #--------------fix bug 240913
                    self.highFeas_encoder.ref_point_head = self.line_head.decoder.ref_point_head  #共享模型参数
                # self.highFeas_encoder.ref_point_head = copy.deepcopy(self.line_head.decoder.ref_point_head) #只共享初始化
                # self.fuse_conv = ConvModule(self.embed_dims, self.embed_dims,1,padding=1,act_cfg=None,inplace=False)
        # TODO： Waiting for mmengine support
        if use_syncbn and get_world_size() > 1:
            torch.nn.SyncBatchNorm.convert_sync_batchnorm(self)
            print_log('Using SyncBatchNorm()', 'current') 
    def forward(self,
                inputs: torch.Tensor,
                data_samples: OptSampleList = None,
                data_samples_yolov5: OptSampleList = None,
                mode: str = 'tensor') -> ForwardResults:
        """The unified entry for a forward process in both training and test.

        The method should accept three modes: "tensor", "predict" and "loss":

        - "tensor": Forward the whole network and return tensor or tuple of
        tensor without any post-processing, same as a common nn.Module.
        - "predict": Forward and return the predictions, which are fully
        processed to a list of :obj:`DetDataSample`.
        - "loss": Forward and return a dict of losses according to the given
        inputs and data samples.

        Note that this method doesn't handle either back propagation or
        parameter update, which are supposed to be done in :meth:`train_step`.

        Args:
            inputs (torch.Tensor): The input tensor with shape
                (N, C, ...) in general.
            data_samples (list[:obj:`DetDataSample`], optional): A batch of
                data samples that contain annotations and predictions.
                Defaults to None.
            mode (str): Return what kind of value. Defaults to 'tensor'.

        Returns:
            The return type depends on ``mode``.
            - If ``mode="tensor"``, return a tensor or a tuple of tensor.
            - If ``mode="predict"``, return a list of :obj:`DetDataSample`.
            - If ``mode="loss"``, return a dict of tensor.
        pseudo_and_yolov5_collate+Det_YOLOv5DetDataPreprocessor: {'inputs': inputs, 'data_samples': data_samples, 'data_samples_yolov5': data_samples_output}
        """
        # for i in data_samples:
        #     if i.img_id==355:
        #         print(i.gt_line_instances.line_points)
        if mode == 'loss':
            return self.loss(inputs, data_samples, data_samples_yolov5)
        elif mode == 'predict':
            return self.predict(inputs, data_samples)
        elif mode == 'tensor':
            return self._forward(inputs, data_samples)
        else:
            raise RuntimeError(f'Invalid mode "{mode}". '
                               'Only supports loss, predict and tensor mode')
    # copy from mmdet/models/detectors/single_stage.py
    def loss(self, batch_inputs: Tensor,
             batch_data_samples: SampleList,
             data_samples_yolov5: SampleList = None) -> Union[dict, list]:
        """Calculate losses from a batch of inputs and data samples.

        Args:
            batch_inputs (Tensor): Input images of shape (N, C, H, W).
                These should usually be mean centered and std scaled.
            batch_data_samples (list[:obj:`DetDataSample`]): The batch
                data samples. It usually includes information such
                as `gt_instance` or `gt_panoptic_seg` or `gt_sem_seg`.
        Returns:
            dict: A dictionary of loss components.
        """
        x = self.extract_feat(batch_inputs)
        if self.HighFeaRefine:
            line_losses, query_feas, query_pos, query_scores, query_preds, pos_inds = self.line_head.loss(x[-self.line_head.num_feature_levels:], batch_data_samples, return_queries=True) #zcf 240920
            # x_last = self.enhance_highFeas_with_queries(x[-1], query_feas, query_scores, query_preds, batch_data_samples[0].metainfo['img_shape'],pos_inds)  #h,w
            method_name = f"enhance_highFeas_with_queries_v{self.highFeasFuseMode}"
            # 获取实例的方法并调用
            # epoch = batch_data_samples[0].metainfo['epoch'] #.get('epoch', 0) # 获取 epoch 的值
            # if epoch!=self.current_epoch: #-------for debug
            #     print('epoch***********',epoch)
            x_last = getattr(self, method_name, lambda: "Invalid version.")(x[-1],batch_data_samples,query_feas, query_pos, 
                                    query_scores, pos_inds=pos_inds) #epoch, modify in mmengine/runner/loops.py:128
            x = list(x[:-1]) + [x_last]
            x = tuple(x)
        else:
            line_losses = self.line_head.loss(x, batch_data_samples)

        if data_samples_yolov5 and (self.bbox_head_type in ['YOLOv8Head','YOLOv5Head','YOLOv8HeadFlexibleLoss','YOLOv8MedLarHead']):
            losses = self.bbox_head.loss(x, data_samples_yolov5)
        else:
            losses = self.bbox_head.loss(x, batch_data_samples)  
            # if data_samples_yolov5:
            #     print('=======collate and DetDataPreprocessor: unnecessary to choose the collate and DetDataPreprocessor of YOLOv5')
        #{'loss_cls': tensor(41.8035, device='cuda:0', grad_fn=<MulBackward0>), 'loss_bbox': tensor(0., device='cuda:0', grad_fn=<MulBackward0>), 'loss_dfl': tensor(0., device='cuda:0', grad_fn=<MulBackward0>)}
        losses.update(line_losses)
        return losses
    # copy from mmdet/models/detectors/single_stage.py
    def predict(self,
                batch_inputs: Tensor,
                batch_data_samples: SampleList,
                rescale: bool = True) -> SampleList:
        """Predict results from a batch of inputs and data samples with post-
        processing.

        Args:
            batch_inputs (Tensor): Inputs with shape (N, C, H, W).
            batch_data_samples (List[:obj:`DetDataSample`]): The Data
                Samples. It usually includes information such as
                `gt_instance`, `gt_panoptic_seg` and `gt_sem_seg`.
            rescale (bool): Whether to rescale the results.
                Defaults to True.

        Returns:
            list[:obj:`DetDataSample`]: Detection results of the
            input images. Each DetDataSample usually contain
            'pred_instances'. And the ``pred_instances`` usually
            contains following keys.

                - scores (Tensor): Classification scores, has a shape
                    (num_instance, )
                - labels (Tensor): Labels of bboxes, has a shape
                    (num_instances, ).
                - bboxes (Tensor): Has a shape (num_instances, 4),
                    the last dimension 4 arrange as (x1, y1, x2, y2).
        """
        x = self.extract_feat(batch_inputs)
        if self.HighFeaRefine:
            line_results_list,batch_data_samples, query_feas, query_pos, query_scores, query_preds = self.line_head.predict(
                x[-self.line_head.num_feature_levels:], batch_data_samples, rescale=rescale, return_queries=True)  #zcf 240920
            #[InstanceData]*bs, results.bboxes,results.scores,results.labels
            # x_last = self.enhance_highFeas_with_queries(x[-1], query_feas, query_scores, query_preds, batch_data_samples[0].metainfo['img_shape'])  #h,w
            method_name = f"enhance_highFeas_with_queries_v{self.highFeasFuseMode}"
            # 获取实例的方法并调用
            x_last = getattr(self, method_name, lambda: "Invalid version.")(x[-1],batch_data_samples,query_feas, query_pos, query_scores)
            x = list(x[:-1]) + [x_last]
            x = tuple(x)
        else:
            line_results_list,batch_data_samples = self.line_head.predict(
                x, batch_data_samples, rescale=rescale) 
        results_list = self.bbox_head.predict(
                x, batch_data_samples, rescale=rescale) #[InstanceData]*bs, results.bboxes,results.scores,results.labels
        batch_data_samples = self.add_pred_to_datasample(
            batch_data_samples, results_list, line_results_list)
        return batch_data_samples
    # same as def predict, 在/media/zcf/extra/zcf/code/231108_FoundationModels/mmyolo_forv8/mmyolo/mmyolo/models/layers/line_head_v3_layers.py中增加一个.item()
    def _forward(self,
                batch_inputs: Tensor,
                batch_data_samples: SampleList,
                rescale: bool = True) -> SampleList:
        x = self.extract_feat(batch_inputs)
        if self.HighFeaRefine:
            line_results_list,batch_data_samples, query_feas, query_pos, query_scores, query_preds = self.line_head.predict(
                x[-self.line_head.num_feature_levels:], batch_data_samples, rescale=rescale, return_queries=True)  #zcf 240920
            #[InstanceData]*bs, results.bboxes,results.scores,results.labels
            # x_last = self.enhance_highFeas_with_queries(x[-1], query_feas, query_scores, query_preds, batch_data_samples[0].metainfo['img_shape'])  #h,w
            method_name = f"enhance_highFeas_with_queries_v{self.highFeasFuseMode}"
            # 获取实例的方法并调用
            x_last = getattr(self, method_name, lambda: "Invalid version.")(x[-1],batch_data_samples,query_feas, query_pos, query_scores)
            x = list(x[:-1]) + [x_last]
            x = tuple(x)
        else:
            line_results_list,batch_data_samples = self.line_head.predict(
                x, batch_data_samples, rescale=rescale) 
        # results_list = self.bbox_head.predict(
        #         x, batch_data_samples, rescale=rescale) #[InstanceData]*bs, results.bboxes,results.scores,results.labels
        results = self.bbox_head.forward(x)
        # batch_data_samples = self.add_pred_to_datasample(
        #     batch_data_samples, results_list, line_results_list)
        return results
    def enhance_highFeas_with_queries_v5(self, box_feas, batch_data_samples, query_feas, query_pos, query_scores, pos_inds=None):
        """
        使用Query特征和边框信息增强特征图
        Parameters:
        - box_feas: 原始特征图，维度为 [bs, 256, H, W] (H1/32, W1/32)
        - query_feas: Query特征，维度为 [bs, num_query, 256]
        - query_pos: Query特征，维度为 [bs, num_query, 256]
        - query_scores: 分类分数信息，维度为 [bs, num_query, 1]
        - query_preds: 边框信息，维度为 [bs, num_query, 2*2], on img_shape
        - pos_inds: [tensor]*bs 
        Returns:
        - updated_feature_map: 更新后的特征图，维度为 [bs, 256, H, W]
        """
        query_pos, query_scores= torch.clone(query_pos).detach(), torch.clone(query_scores).detach()
        query_feas = torch.clone(query_feas).detach()
        query_scores = query_scores.sigmoid()
        query_scores, _ = query_scores.max(-1) 
        #--------------get 
        if self.training:
            bs = query_pos.shape[0]
            # if pos_inds is not None:
            lineBg_thr = torch.stack([
                torch.min(query_scores[i, pos_inds[i]]) if len(pos_inds[i]) > 0 else torch.max(query_scores[i])
                for i in range(bs)]).unsqueeze(1) # # #[bs,1]

            if dist.is_available() and dist.is_initialized():
                # rank, world_size = get_dist_info()
                lineBg_thr_here = torch.cat(all_gather(lineBg_thr.squeeze(1))) #torch.Size([num_gpus*bs])
                # 更新运行时统计信息
                with torch.no_grad():
                    self.running_lineBg_weight = self.momentum * lineBg_thr_here.mean() + (1 - self.momentum) * self.running_lineBg_weight
            else:
                # 更新运行时统计信息
                with torch.no_grad():
                    self.running_lineBg_weight = self.momentum * lineBg_thr.squeeze(1).mean() + (1 - self.momentum) * self.running_lineBg_weight
        else:
            lineBg_thr = self.running_lineBg_weight
        #--------------
        # lineBg_weight = self.alpha * (lineBg_thr - self.beta - query_scores)
        lineBg_weight = self.alpha * F.relu(lineBg_thr - self.beta - query_scores,inplace=True)
        # lineBg_weight = self.alpha * F.relu(lineBg_thr - 0.5 * torch.tanh(self.beta) - query_scores,inplace=True)
        # lineBg_weight = self.alpha * F.relu(lineBg_thr - query_scores,inplace=True)
        # if not self.training:
        #     print('self.beta', self.beta)
        #     print('lineBg_weight',lineBg_weight)
        
        encoder_inputs_dict = self.pre_transformer_v2(box_feas, batch_data_samples)
        
        feat_update = self.highFeas_encoder(query=box_feas,k=query_feas,v=query_feas,key_pos=query_pos,other_W=lineBg_weight,**encoder_inputs_dict) #identity=None, attn_mask=None,
        # feat_update = feat_update.permute(0, 2, 1).view(box_feas.shape).contiguous() #[bs, sum(H,W), 256]--->[bs, 256, H, W]
        # feat_update = self.fuse_conv(feat_update)
        return feat_update 
 
    # copy from mmdetection_v3d3d0/mmdet/models/detectors/detr.py: pre_transformer
    def pre_transformer_v2(
            self,
            feat: Tuple[Tensor],
            batch_data_samples: OptSampleList = None) -> Tuple[Dict, Dict]:

        # NOTE feat contains only one feature.
        batch_size, feat_dim, _, _ = feat.shape
        # construct binary masks which for the transformer.
        assert batch_data_samples is not None
        batch_input_shape = batch_data_samples[0].batch_input_shape
        input_img_h, input_img_w = batch_input_shape
        img_shape_list = [sample.img_shape for sample in batch_data_samples]
        same_shape_flag = all([
            s[0] == input_img_h and s[1] == input_img_w for s in img_shape_list
        ])
        if torch.onnx.is_in_onnx_export() or same_shape_flag:
            masks = None
            # [batch_size, embed_dim, h, w]
            pos_embed = self.positional_encoding(masks, input=feat)
        else:
            masks = feat.new_ones((batch_size, input_img_h, input_img_w))
            for img_id in range(batch_size):
                img_h, img_w = img_shape_list[img_id]
                masks[img_id, :img_h, :img_w] = 0
            # NOTE following the official DETR repo, non-zero values represent
            # ignored positions, while zero values mean valid positions.

            masks = F.interpolate(
                masks.unsqueeze(1),
                size=feat.shape[-2:]).to(torch.bool).squeeze(1)
            # [batch_size, embed_dim, h, w]
            pos_embed = self.positional_encoding(masks)

        # use `view` instead of `flatten` for dynamically exporting to ONNX
        # [bs, c, h, w] -> [bs, h*w, c]
        # feat = feat.view(batch_size, feat_dim, -1).permute(0, 2, 1).contiguous()
        pos_embed = pos_embed.view(batch_size, feat_dim, -1).permute(0, 2, 1).contiguous()
        # [bs, h, w] -> [bs, h*w]
        if masks is not None:
            masks = masks.view(batch_size, -1)

        # prepare transformer_inputs_dict
        # decoder_inputs_dict = dict(query=feat,query_pos=pos_embed, key_padding_mask=masks)
        # decoder_inputs_dict = dict(query=feat,query_pos=pos_embed)
        decoder_inputs_dict = dict(query_pos=pos_embed)
        return decoder_inputs_dict
    
    #copy from mmdet/models/detectors/base.py
    def add_pred_to_datasample(self, data_samples: SampleList,
                               results_list: InstanceList, line_results_list = None) -> SampleList:
        """Add predictions to `DetDataSample`.

        Args:
            data_samples (list[:obj:`DetDataSample`], optional): A batch of
                data samples that contain annotations and predictions.
            results_list (list[:obj:`InstanceData`]): Detection results of
                each image.

        Returns:
            list[:obj:`DetDataSample`]: Detection results of the
            input images. Each DetDataSample usually contain
            'pred_instances'. And the ``pred_instances`` usually
            contains following keys.

                - scores (Tensor): Classification scores, has a shape
                    (num_instance, )
                - labels (Tensor): Labels of bboxes, has a shape
                    (num_instances, ).
                - bboxes (Tensor): Has a shape (num_instances, 4),
                    the last dimension 4 arrange as (x1, y1, x2, y2).
        """
        if line_results_list is None:
            line_results_list = [line_results_list for _ in range(len(data_samples))]
        for data_sample, pred_instances, line_pred_instances in zip(data_samples, results_list, line_results_list):
            #------------------------------zcf use_tta 20240916
            if self.use_tta:
                bboxes2 = line2box_torch(line_pred_instances['line_points'],img_height=data_sample.ori_shape[0],img_width=data_sample.ori_shape[1])
                results = InstanceData()
                results.bboxes = torch.cat((pred_instances['bboxes'], bboxes2), dim=0) #[num_queries,num_points*self.pts_dim]
                results.scores = torch.cat((pred_instances['scores'], line_pred_instances['line_scores']), dim=0)  #[num_queries,]
                results.labels = torch.cat((pred_instances['labels'], line_pred_instances['line_labels']+self.box_num_cls), dim=0) #[num_queries,], or None, must same length
                pred_instances = results 
            else:
                data_sample.line_pred_instances = line_pred_instances
            #------------------------------zcf use_tta 20240916
            data_sample.pred_instances = pred_instances
        samplelist_boxtype2tensor(data_samples)
        return data_samples
    
if __name__ == '__main__':
    print('hello')


