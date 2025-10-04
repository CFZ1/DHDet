# ****************Latest version 2025-03-22 lineRefPts 250322
from typing import Optional, Tuple, Union

import torch
from mmcv.cnn import build_norm_layer
from mmcv.cnn.bricks.transformer import FFN, MultiheadAttention
from mmengine.model import ModuleList
from torch import Tensor, nn

from mmdet.models.layers.transformer import (DetrTransformerDecoder, DetrTransformerDecoderLayer,
                          DetrTransformerEncoder, DetrTransformerEncoderLayer,DeformableDetrTransformerDecoderLayer)
from mmdet.models.layers.transformer import inverse_sigmoid
from mmyolo.registry import MODELS
from mmengine.model import BaseModule
import math
import warnings
from typing import Optional, no_type_check

import mmengine
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmengine.model import BaseModule, constant_init, xavier_init
from mmengine.registry import MODELS
from mmengine.utils import deprecated_api_warning
from torch.autograd.function import Function, once_differentiable

from mmcv.utils import IS_CUDA_AVAILABLE, IS_MLU_AVAILABLE
from mmcv.ops.multi_scale_deform_attn import MultiScaleDeformableAttnFunction, multi_scale_deformable_attn_pytorch
from mmcv.ops import MultiScaleDeformableAttention
from mmyolo.models.data_preprocessors import prepare_line_points
from mmcv.cnn import Linear
from mmdet.utils import ConfigType, OptConfigType
from mmengine import ConfigDict
import copy
from mmdet.models import MLP
def coordinate_to_encoding(coord_tensor: Tensor,
                           num_feats: int = 128,
                           temperature: int = 10000,
                           scale: float = 2 * math.pi):
    """Convert coordinate tensor to positional encoding.

    Args:
        coord_tensor (Tensor): Coordinate tensor to be converted to
            positional encoding. With the last dimension as 2 or 4.
        num_feats (int, optional): The feature dimension for each position
            along x-axis or y-axis. Note the final returned dimension
            for each position is 2 times of this value. Defaults to 128.
        temperature (int, optional): The temperature used for scaling
            the position embedding. Defaults to 10000.
        scale (float, optional): A scale factor that scales the position
            embedding. The scale will be used only when `normalize` is True.
            Defaults to 2*pi.
    Returns:
        Tensor: Returned encoded positional tensor.
    """
    dim_t = torch.arange(
        num_feats, dtype=torch.float32, device=coord_tensor.device)
    # dim_t = temperature**(2 * (dim_t // 2) / num_feats)
    if torch.__version__ <= '1.8.0': #change by zhou, 因为4090上老UserWarning，实际上对negative values才有影响
        dim_t = temperature**(2 * (dim_t // 2) / num_feats)
    else:
        dim_t = temperature**(2 * torch.div(dim_t, 2, rounding_mode='floor') / num_feats)
    x_embed = coord_tensor[..., 0] * scale
    y_embed = coord_tensor[..., 1] * scale
    pos_x = x_embed[..., None] / dim_t
    pos_y = y_embed[..., None] / dim_t
    pos_x = torch.stack((pos_x[..., 0::2].sin(), pos_x[..., 1::2].cos()),
                        dim=-1).flatten(-2) #-------------2-->-2, change by zhou
    pos_y = torch.stack((pos_y[..., 0::2].sin(), pos_y[..., 1::2].cos()),
                        dim=-1).flatten(-2) #-------------2-->-2, change by zhou
    if coord_tensor.size(-1) == 2:
        pos = torch.cat((pos_y, pos_x), dim=-1)
    elif coord_tensor.size(-1) == 4:
        w_embed = coord_tensor[..., 2] * scale
        pos_w = w_embed[..., None] / dim_t
        pos_w = torch.stack((pos_w[..., 0::2].sin(), pos_w[..., 1::2].cos()),
                            dim=-1).flatten(2)

        h_embed = coord_tensor[..., 3] * scale
        pos_h = h_embed[..., None] / dim_t
        pos_h = torch.stack((pos_h[..., 0::2].sin(), pos_h[..., 1::2].cos()),
                            dim=-1).flatten(2)

        pos = torch.cat((pos_y, pos_x, pos_w, pos_h), dim=-1)
    else:
        raise ValueError('Unknown pos_tensor shape(-1):{}'.format(
            coord_tensor.size(-1)))
    return pos

def correct_reference_points_by_length_scaling(new_reference_points,tmp_reg_preds_length): 
    # 计算中心点
    center_point = new_reference_points.mean(dim=2, keepdim=True)
    # 计算每个点相对于中心点的偏移量
    offsets = new_reference_points - center_point  
    # 根据 tmp_reg_preds_length 缩放每个点相对于中心点的偏移量
    scaled_offsets = offsets * tmp_reg_preds_length.unsqueeze(-1).unsqueeze(-1)  
    # 将修正后的偏移量加回到中心点上得到修正后的新参考点
    corrected_reference_points = center_point + scaled_offsets
    return corrected_reference_points

def correct_lines_batch_origin(tmp, correctAngleVal):
    # tmp的形状为[bs,num_query,lineRefPts,2], correctAngleVal的形状为[bs,num_query,1]
    # 将角度修正值转换为弧度,correctAngleVal*180=角度
    angles = correctAngleVal * math.pi
    cos_vals = torch.cos(angles).unsqueeze(-1)
    sin_vals = torch.sin(angles).unsqueeze(-1)
    # 构造旋转矩阵，适应新的数据结构
    rotation_matrix = torch.cat((cos_vals, -sin_vals, sin_vals, cos_vals), dim=-1).view(*angles.shape[:-1], 2, 2)
    origins = tmp[..., 0, :] # 提取第一个点, 作为旋转点
    # 对每条线段的第二个点进行旋转操作
    rotated_points = torch.einsum('bnik,bnk->bni', rotation_matrix, tmp[..., 1, :]-origins) + origins
    # 重建旋转后的线段
    corrected_lines = torch.cat((origins.unsqueeze(-2), rotated_points.unsqueeze(-2)), dim=-2)
    return corrected_lines
# tmp = torch.tensor([[[[5.0, 5.0], [0.0, 0.0]], 
#                     [[1.0, 1.0], [2.0, 4.0]], 
#                     [[2.0, 2.0], [6.0, 6.0]]]], dtype=torch.float32)  # 形状 [3, 2, 2]
# correctAngleVal = torch.tensor([[[0.5], [1/4], [-0.5]]], dtype=torch.float32)  # 形状 [3, 1]

# copy from mmdet.models.layers.transformer/deformable_detr_layers.py: DeformableDetrTransformerDecoder
# change reference_points
@MODELS.register_module()
class DeformableDetrTransformerMultiRefDecoder(DetrTransformerDecoder):
    """Transformer Decoder of Deformable DETR."""
    def __init__(self,
                 num_layers: int,
                 layer_cfg: ConfigType,
                 post_norm_cfg: OptConfigType = dict(type='LN'),
                 return_intermediate: bool = True,
                 init_cfg: Union[dict, ConfigDict] = None,
                 **kwargs) -> None:
        self.as_two_stage = kwargs['as_two_stage']
        self.ref_numPts = kwargs['ref_numPts']
        super().__init__(num_layers=num_layers,layer_cfg=layer_cfg,post_norm_cfg=post_norm_cfg,
                         return_intermediate=return_intermediate,init_cfg=init_cfg)
        
    def _init_layers(self) -> None:
        """Initialize decoder layers."""
        if self.layer_cfg.type =='DeformableDetrTransformerDecoderLayer':
            self.layer_cfg.pop('type')
            self.layers = ModuleList([
                DeformableDetrTransformerDecoderLayer(**self.layer_cfg)
                for _ in range(self.num_layers)
            ])
        else:
            self.layers = ModuleList([
                MODELS.build(self.layer_cfg) for _ in range(self.num_layers)
            ])
        self.embed_dims = self.layers[0].embed_dims
        if self.post_norm_cfg is not None:
            raise ValueError('There is not post_norm in '
                             f'{self._get_name()}')
        if self.as_two_stage:
            # opt 1: dino
            self.ref_point_head = MLP(self.embed_dims * self.ref_numPts, self.embed_dims,self.embed_dims, 2) #---------lineRefPts 250322----------
            # opt 2: rtdetr
            # self.ref_point_head = MLP(self.ref_len, self.embed_dims * 2, self.embed_dims, 2)

    def forward(self,
                query: Tensor,
                query_pos: Tensor,
                value: Tensor,
                key_padding_mask: Tensor,
                reference_points: Tensor,
                spatial_shapes: Tensor,
                level_start_index: Tensor,
                valid_ratios: Tensor,
                self_attn_mask: Tensor = None, #----------------CDN----240601----step 6 
                reg_branches: Optional[nn.Module] = None,
                fineLength: bool = False,
                **kwargs) -> Tuple[Tensor]:
        """Forward function of Transformer decoder.

        Args:
            query (Tensor): The input queries, has shape (bs, num_queries,
                dim).
            query_pos (Tensor): The input positional query, has shape
                (bs, num_queries, dim). It will be added to `query` before
                forward function.
            value (Tensor): The input values, has shape (bs, num_value, dim).
            key_padding_mask (Tensor): The `key_padding_mask` of `cross_attn`
                input. ByteTensor, has shape (bs, num_value).
            reference_points (Tensor): The initial reference, has shape
                (bs, num_queries, 4) with the last dimension arranged as
                (cx, cy, w, h) when `as_two_stage` is `True`, otherwise has
                shape (bs, num_queries, 2) with the last dimension arranged
                as (cx, cy).
                [bs, num_queries, line_num_points, 2] #-----------------------------------------------zhou
            spatial_shapes (Tensor): Spatial shapes of features in all levels,
                has shape (num_levels, 2), last dimension represents (h, w).
            level_start_index (Tensor): The start index of each level.
                A tensor has shape (num_levels, ) and can be represented
                as [0, h_0*w_0, h_0*w_0+h_1*w_1, ...].
            valid_ratios (Tensor): The ratios of the valid width and the valid
                height relative to the width and the height of features in all
                levels, has shape (bs, num_levels, 2).
            reg_branches: (obj:`nn.ModuleList`, optional): Used for refining
                the regression results. Only would be passed when
                `with_box_refine` is `True`, otherwise would be `None`.

        Returns:
            tuple[Tensor]: Outputs of Deformable Transformer Decoder.

            - output (Tensor): Output embeddings of the last decoder, has
              shape (num_queries, bs, embed_dims) when `return_intermediate`
              is `False`. Otherwise, Intermediate output embeddings of all
              decoder layers, has shape (num_decoder_layers, num_queries, bs,
              embed_dims).
            - reference_points (Tensor): The reference of the last decoder
              layer, has shape (bs, num_queries, 4)  when `return_intermediate`
              is `False`. Otherwise, Intermediate references of all decoder
              layers, has shape (num_decoder_layers, bs, num_queries, 4). The
              coordinates are arranged as (cx, cy, w, h)
        """
        #TODO!
        query_pos_Flag = False
        if query_pos is None:
            query_pos_Flag = True
        output = query
        intermediate = []
        intermediate_reference_points = []
        for layer_id, layer in enumerate(self.layers):
            if reference_points.shape[-1] == 4:
                if len(reference_points.shape) ==4: # for line head   
                    reference_points_input = \
                        reference_points[:, :, None] * \
                        torch.cat([valid_ratios, valid_ratios], -1)[:, None, :, None]
                elif len(reference_points.shape) ==3:
                    reference_points_input = \
                        reference_points[:, :, None] * \
                        torch.cat([valid_ratios, valid_ratios], -1)[:, None]
            else:
                assert reference_points.shape[-1] == 2
                if len(reference_points.shape) ==4: # for line head    
                    # [bs, num_queries, lineRefPts, 2]---> [bs, num_queries, lineRefPts, 2]
                    if reference_points.shape[-2] != self.points_for_lossMetric:
                        bs, num_queries = reference_points.shape[0], reference_points.shape[1]
                        reference_points_input = prepare_line_points(reference_points.view(-1, *reference_points.shape[-2:]),self.line_points_inter_method,self.points_for_lossMetric)
                        reference_points_input = reference_points_input.view(bs, num_queries, *reference_points_input.shape[-2:])
                    else:
                        reference_points_input = reference_points
                    reference_points_input = \
                        reference_points_input[:, :, None] * \
                        valid_ratios[:, None, :, None] #[bs, num_queries, 1, lineRefPts, 2]* [bs, 1, num_levels, 1, 2]
                elif len(reference_points.shape) ==3:
                    reference_points_input = \
                        reference_points[:, :, None] * \
                        valid_ratios[:, None]
            #----------zcf self.as_two_stage 240520----------step 2, query_pos is move to self.decoder
            if query_pos_Flag:
                if reference_points.shape[-1] == 2 and len(reference_points.shape) ==4:
                    reference_points_here = \
                        reference_points[:, :, None] * \
                        valid_ratios[:, None, :, None] 
                    reference_points_here = reference_points_here[:, :, 0, :, :] #[bs, num_queries,lineRefPts,2]
                else:
                    print('not implement')
                #opt 1: dino
                # -------[bs, num_queries,lineRefPts,2]-->[bs, num_queries,lineRefPts,num_feats*2]
                query_sine_embed = coordinate_to_encoding(reference_points_here,num_feats=128).flatten(-2) # 理论上应该在差值之前，在乘以valid_ratios之后
                # -------[bs, num_queries,lineRefPts*num_feats*2]--> 
                query_pos = self.ref_point_head(query_sine_embed)
                #opt 2: rtdetr
                # bs, num_queries = reference_points_here.shape[0], reference_points_here.shape[1]
                # query_pos = self.ref_point_head(reference_points_here.view(bs, num_queries,-1)) #[bs, num_queries, 4]-->[bs, num_queries,lineRefPts,2]
            output = layer(
                output,
                query_pos=query_pos,
                value=value,
                key_padding_mask=key_padding_mask,
                self_attn_mask=self_attn_mask, #----------------CDN----240601----step 6 送入到decoder.自注意力中
                spatial_shapes=spatial_shapes,
                level_start_index=level_start_index,
                valid_ratios=valid_ratios,
                reference_points=reference_points_input,
                **kwargs)
            if reg_branches is not None:
                if len(reference_points.shape) ==4: # for line head    
                    tmp_reg_preds_ori = reg_branches[layer_id](output) #output= (bs, num_queries, embed_dims),reference_points=[bs, num_queries, line_num_points, 2]
                    lineRefPts, pts_dim = reference_points.shape[-2], reference_points.shape[-1]
                    size_first_dims =tmp_reg_preds_ori.shape[:-1]
                    tmp_reg_preds = tmp_reg_preds_ori[...,:lineRefPts*pts_dim].view(*size_first_dims,lineRefPts, -1)
                    if fineLength:
                        tmp_reg_preds_length = F.relu(tmp_reg_preds_ori[...,-1])
                    if reference_points.shape[-1] == 4:
                        new_reference_points = tmp_reg_preds + inverse_sigmoid(
                            reference_points)
                        new_reference_points = new_reference_points.sigmoid()
                    else:
                        assert reference_points.shape[-1] == 2
                        new_reference_points = tmp_reg_preds
                        new_reference_points[..., :2] = tmp_reg_preds[
                            ..., :2] + inverse_sigmoid(reference_points)
                        if fineLength:
                            new_reference_points = correct_reference_points_by_length_scaling(new_reference_points,tmp_reg_preds_length)
                        if self.correctAngle:
                            if self.correctAngleSigFirst:
                                new_reference_points = new_reference_points.sigmoid()
                            new_reference_points = correct_lines_batch_origin(new_reference_points, tmp_reg_preds_ori[...,self.correctAngleIndex].unsqueeze(-1))  #[bs,num_query,lineRefPts,2],[bs,num_query,1]
                            if self.correctAngleSigFirst: 
                                new_reference_points = new_reference_points.clamp_(min=0, max=1)
                            else:
                                new_reference_points = new_reference_points.sigmoid()
                        else:
                            new_reference_points = new_reference_points.sigmoid()
                        
                    reference_points = new_reference_points.detach()
                elif len(reference_points.shape) ==3:
                    tmp_reg_preds = reg_branches[layer_id](output)
                    if reference_points.shape[-1] == 4:
                        new_reference_points = tmp_reg_preds + inverse_sigmoid(
                            reference_points)
                        new_reference_points = new_reference_points.sigmoid()
                    else:
                        assert reference_points.shape[-1] == 2
                        new_reference_points = tmp_reg_preds
                        new_reference_points[..., :2] = tmp_reg_preds[
                            ..., :2] + inverse_sigmoid(reference_points)
                        new_reference_points = new_reference_points.sigmoid()
                    reference_points = new_reference_points.detach()

            if self.return_intermediate:
                intermediate.append(output)
                # intermediate_reference_points.append(reference_points)
                intermediate_reference_points.append(new_reference_points)
                # NOTE this is for the "Look Forward Twice" module,
                # in the DeformDETR, reference_points was appended.

        if self.return_intermediate:
            return torch.stack(intermediate), torch.stack(
                intermediate_reference_points), query_pos #-------------240607

        return output, reference_points, query_pos#-------------240607
    
# copy from mmdet.models.layers.transformer/deformable_detr_layers.py: DeformableDetrTransformerDecoderLayer
@MODELS.register_module()    
class DeformableDetrTransformerDecoderLayerForRegist(DetrTransformerDecoderLayer):
    """Decoder layer of Deformable DETR."""

    def _init_layers(self) -> None:
        """Initialize self_attn, cross-attn, ffn, and norms."""
        self.self_attn = MultiheadAttention(**self.self_attn_cfg)
        self.cross_attn = MODELS.build(self.cross_attn_cfg) #-----------
        self.embed_dims = self.self_attn.embed_dims
        self.ffn = FFN(**self.ffn_cfg)
        norms_list = [
            build_norm_layer(self.norm_cfg, self.embed_dims)[1]
            for _ in range(3)
        ]
        self.norms = ModuleList(norms_list)  
# copy from mmcv/ops/multi_scale_deform_attn.py: MultiScaleDeformableAttention    
@MODELS.register_module()
class MultiScaleDeformableMultiRefAttention(MultiScaleDeformableAttention):
    """An attention module used in Deformable-Detr.

    `Deformable DETR: Deformable Transformers for End-to-End Object Detection.
    <https://arxiv.org/pdf/2010.04159.pdf>`_.

    Args:
        embed_dims (int): The embedding dimension of Attention.
            Default: 256.
        num_heads (int): Parallel attention heads. Default: 64.
        num_levels (int): The number of feature map used in
            Attention. Default: 4.
        num_points (int): The number of sampling points for
            each query in each head. Default: 4.
        im2col_step (int): The step used in image_to_column.
            Default: 64.
        dropout (float): A Dropout layer on `inp_identity`.
            Default: 0.1.
        batch_first (bool): Key, Query and Value are shape of
            (batch, n, embed_dim)
            or (n, batch, embed_dim). Default to False.
        norm_cfg (dict): Config dict for normalization layer.
            Default: None.
        init_cfg (obj:`mmcv.ConfigDict`): The Config for initialization.
            Default: None.
        value_proj_ratio (float): The expansion ratio of value_proj.
            Default: 1.0.
    """

    @no_type_check
    @deprecated_api_warning({'residual': 'identity'},
                            cls_name='MultiScaleDeformableMultiRefAttention')
    def forward(self,
                query: torch.Tensor,
                key: Optional[torch.Tensor] = None,
                value: Optional[torch.Tensor] = None,
                identity: Optional[torch.Tensor] = None,
                query_pos: Optional[torch.Tensor] = None,
                key_padding_mask: Optional[torch.Tensor] = None,
                reference_points: Optional[torch.Tensor] = None,
                spatial_shapes: Optional[torch.Tensor] = None,
                level_start_index: Optional[torch.Tensor] = None,
                **kwargs) -> torch.Tensor:
        """Forward Function of MultiScaleDeformAttention.

        Args:
            query (torch.Tensor): Query of Transformer with shape
                (num_query, bs, embed_dims).
            key (torch.Tensor): The key tensor with shape
                `(num_key, bs, embed_dims)`.
            value (torch.Tensor): The value tensor with shape
                `(num_key, bs, embed_dims)`.
            identity (torch.Tensor): The tensor used for addition, with the
                same shape as `query`. Default None. If None,
                `query` will be used.
            query_pos (torch.Tensor): The positional encoding for `query`.
                Default: None.
            key_padding_mask (torch.Tensor): ByteTensor for `query`, with
                shape [bs, num_key].
            reference_points (torch.Tensor):  The normalized reference
                points with shape (bs, num_query, num_levels, 2),
                all elements is range in [0, 1], top-left (0,0),
                bottom-right (1, 1), including padding area.
                or (N, Length_{query}, num_levels, 4), add
                additional two dimensions is (w, h) to
                form reference boxes.
            spatial_shapes (torch.Tensor): Spatial shape of features in
                different levels. With shape (num_levels, 2),
                last dimension represents (h, w).
            level_start_index (torch.Tensor): The start index of each level.
                A tensor has shape ``(num_levels, )`` and can be represented
                as [0, h_0*w_0, h_0*w_0+h_1*w_1, ...].

        Returns:
            torch.Tensor: forwarded results with shape
            [num_query, bs, embed_dims].
        """

        if value is None:
            value = query

        if identity is None:
            identity = query
        if query_pos is not None:
            query = query + query_pos
        if not self.batch_first:
            # change to (bs, num_query ,embed_dims)
            query = query.permute(1, 0, 2)
            value = value.permute(1, 0, 2)

        bs, num_query, _ = query.shape
        bs, num_value, _ = value.shape
        assert (spatial_shapes[:, 0] * spatial_shapes[:, 1]).sum() == num_value

        value = self.value_proj(value)
        if key_padding_mask is not None:
            value = value.masked_fill(key_padding_mask[..., None], 0.0)
        value = value.view(bs, num_value, self.num_heads, -1)
        sampling_offsets = self.sampling_offsets(query).view(
            bs, num_query, self.num_heads, self.num_levels, self.num_points, 2) #(bs, num_query, num_heads, num_levels, num_points, 2)
        attention_weights = self.attention_weights(query).view(
            bs, num_query, self.num_heads, self.num_levels * self.num_points)
        attention_weights = attention_weights.softmax(-1)

        attention_weights = attention_weights.view(bs, num_query,
                                                   self.num_heads,
                                                   self.num_levels,
                                                   self.num_points)
        if len(reference_points.shape)==5:
            lineRefPts, pt_dim = reference_points.shape[-2], reference_points.shape[-1]  # .item()这会是一个整数,240722, Strange, when mode = 'tensor', Not an integer
            reference_points = reference_points.repeat_interleave(self.num_points // lineRefPts, dim=-2)
            if reference_points.shape[-1] == 2:
                offset_normalizer = torch.stack(
                    [spatial_shapes[..., 1], spatial_shapes[..., 0]], -1)
                sampling_locations = reference_points.view(bs, num_query, 1, self.num_levels,self.num_points, pt_dim) \
                    + sampling_offsets \
                    / offset_normalizer[None, None, None, :, None, :] #(bs, num_query, num_heads, self.num_levels,self.num_points, pt_dim
            elif reference_points.shape[-1] == 4:
                sampling_locations = reference_points.view(bs, num_query, 1, self.num_levels, self.num_points, pt_dim)[...,:2] \
                    + sampling_offsets / self.num_points \
                    * reference_points.view(bs, num_query, 1, self.num_levels, self.num_points, pt_dim)[...,2:] \
                    * 0.5 # reference_points[:, :, None, :, None, :2]
            else:
                raise ValueError(
                    f'Last dim of reference_points must be'
                    f' 2 or 4, but get {reference_points.shape[-1]} instead.')
        elif len(reference_points.shape)==4:
            if reference_points.shape[-1] == 2:
                offset_normalizer = torch.stack(
                    [spatial_shapes[..., 1], spatial_shapes[..., 0]], -1)
                sampling_locations = reference_points[:, :, None, :, None, :] \
                    + sampling_offsets \
                    / offset_normalizer[None, None, None, :, None, :]
            elif reference_points.shape[-1] == 4:
                sampling_locations = reference_points[:, :, None, :, None, :2] \
                    + sampling_offsets / self.num_points \
                    * reference_points[:, :, None, :, None, 2:] \
                    * 0.5
            else:
                raise ValueError(
                    f'Last dim of reference_points must be'
                    f' 2 or 4, but get {reference_points.shape[-1]} instead.')
        if ((IS_CUDA_AVAILABLE and value.is_cuda)
                or (IS_MLU_AVAILABLE and value.is_mlu)):
            output = MultiScaleDeformableAttnFunction.apply(
                value, spatial_shapes, level_start_index, sampling_locations,
                attention_weights, self.im2col_step)
        else:
            output = multi_scale_deformable_attn_pytorch(
                value, spatial_shapes, sampling_locations, attention_weights)

        output = self.output_proj(output)

        if not self.batch_first:
            # (num_query, bs ,embed_dims)
            output = output.permute(1, 0, 2)

        return self.dropout(output) + identity
    
class APGN(nn.Module):
    '''自适应采样点生成网络" (Adaptive Point Generation Network, APGN):'''
    def __init__(self, embed_dims, num_points, design_type=1):
        super().__init__()
        self.embed_dims = embed_dims
        self.num_points = num_points
        self.design_type = design_type
        if design_type==1:
            self.mlp = nn.Sequential(
                Linear(embed_dims, embed_dims),
                nn.ReLU(inplace=True),
                Linear(embed_dims, num_points),# 减去起点和终点
                nn.Sigmoid()  # 输出在 [0, 1] 范围内
            )
        else:
            print('APGN.design_type', design_type)
            self.mlp = nn.Sequential(
                Linear(embed_dims, embed_dims),
                nn.ReLU(inplace=True),
                Linear(embed_dims, num_points+1),  # 减去起点
            )
    def forward(self, query, points):
        # query: [bs, num_queries, dim]
        # points = start_point, end_point: [bs, num_queries, line_num_points, 2]
        # return sample_points=[bs, num_queries, num_points+2, 2], ratios[bs, num_queries, num_points+2]
        bs,num_query,_ = query.shape
        # 预测每个采样点的位置
        ratios = self.mlp(query)  # [bs, num_queries, num_points]
        if self.design_type==1:
            ratios = torch.cat([torch.zeros((bs,num_query,1),device=ratios.device), ratios, torch.ones((bs,num_query,1),device=ratios.device)], dim=-1)  # 添加起点的位置
        else:
            ratios = torch.softmax(ratios, dim=-1)  # 归一化,使总和为1
            ratios = torch.cat([torch.zeros((bs,num_query,1),device=ratios.device), ratios], dim=-1)  # 添加起点的位置
            ratios = ratios.cumsum(dim=-1)  
        # 根据预测的位置生成采样点
        start_point = points[:,:,0].unsqueeze(2)  # [bs, num_queries, 1, 2]
        end_point = points[:,:,-1].unsqueeze(2)  # [bs, num_queries, 1, 2]
        sample_points = start_point + (end_point - start_point) * ratios.unsqueeze(-1)  # ratios, [bs, num_queries, num_points, 1]
        return sample_points,ratios