"""
Created on Mon Apr 15 19:37:13 2024

@author: zcf
"""
from typing import Dict, Tuple, Union, List

import torch
import torch.nn as nn
from torch import Tensor

from mmdet.models import DeformableDETRHead,DeformableDetrTransformerDecoder
from mmdet.structures import OptSampleList
from mmdet.utils import OptConfigType
from mmdet.models import SinePositionalEncoding
from torch.nn.init import normal_
from mmengine.model import xavier_init

from mmyolo.registry import MODELS
import torch.nn.functional as F
import math
from mmcv.cnn import Linear
import copy
from mmdet.models import inverse_sigmoid
from mmengine.model import bias_init_with_prob, constant_init
from mmcv.cnn.bricks.transformer import MultiScaleDeformableAttention

# 如果继承DeformableDETR, 无法在不调用DeformableDETR.__init__的情况下，调用DeformableDETR的其他函数
@MODELS.register_module()
class BboxDeformableDETRHead(DeformableDETRHead): 
    r"""
    3个最重要的函数：loss\ predict \forward
    """
    def __init__(self,
                 *args,
                 share_decoder_withLineHead: dict = {'shared': False, 'deepCopy': False},
                 decoder=None,
                 query_num_ref: int = 1, # 目前不使用
                 positional_encoding: OptConfigType = None,
                 embed_dims: int = 256, 
                 num_queries: int = 300,
                 num_feature_levels: int = 3,
                 num_pred_layer: int = 3,
                 decoder_num_layers: int = 3,
                 with_box_refine: bool = False,
                 as_two_stage: bool = False,
                 **kwargs):
        self.as_two_stage = as_two_stage
        self.query_num_ref = 1 # 感觉不行，因为边框分支的维度是4，预测了xywh, reference_points是2维，修正xy，4维度修正xywh; 如果想要使用，需要考虑很复杂的东西
        kwargs['share_pred_layer'] = not with_box_refine
        kwargs['num_pred_layer'] = (decoder_num_layers + 1) if self.as_two_stage else decoder_num_layers
        kwargs['as_two_stage'] = as_two_stage
        #1.---------DeformableDETRHead
        super().__init__(*args,**kwargs) #确保只是用DeformableDETRHead的
        self.encoder, self.input_proj = None, None
        self.share_decoder_withLineHead = share_decoder_withLineHead
        if share_decoder_withLineHead['shared'] or decoder is None:
            self.decoder = None
        else:
            self.decoder = DeformableDetrTransformerDecoder(**decoder)
        
        #2.---------DeformableDETR
        self.embed_dims = embed_dims
        self.num_queries = num_queries 
        self.num_feature_levels = num_feature_levels
        self.with_box_refine = with_box_refine
        
        self.positional_encoding = SinePositionalEncoding(
            **positional_encoding)
        # self.encoder = DeformableDetrTransformerEncoder(**self.encoder)
        # self.decoder = DeformableDetrTransformerDecoder(**self.decoder)
        if not self.as_two_stage:
            self.query_embedding = nn.Embedding(self.num_queries,
                                                self.embed_dims * 2)
            # NOTE The query_embedding will be split into query and query_pos
            # in self.pre_decoder, hence, the embed_dims are doubled.

        num_feats = self.positional_encoding.num_feats
        assert num_feats * 2 == self.embed_dims, \
            'embed_dims should be exactly 2 times of num_feats. ' \
            f'Found {self.embed_dims} and {num_feats}.'
        if self.encoder is not None:
            self.level_embed = nn.Parameter(
                torch.Tensor(self.num_feature_levels, self.embed_dims))

        if self.as_two_stage:
            self.memory_trans_fc = nn.Linear(self.embed_dims, self.embed_dims)
            self.memory_trans_norm = nn.LayerNorm(self.embed_dims)
            self.pos_trans_fc = nn.Linear(self.embed_dims * 2,
                                          self.embed_dims * 2)
            self.pos_trans_norm = nn.LayerNorm(self.embed_dims * 2)
        else:
            self.reference_points_fc = nn.Linear(self.embed_dims, self.query_num_ref*2) #-----------1.query_num_ref
        #2.---------DeformableDETR
    def _init_layers(self) -> None:
        """Initialize classification branch and regression branch of head."""
        fc_cls = Linear(self.embed_dims, self.cls_out_channels)
        reg_branch = []
        for _ in range(self.num_reg_fcs):
            reg_branch.append(Linear(self.embed_dims, self.embed_dims))
            reg_branch.append(nn.ReLU())
        reg_branch.append(Linear(self.embed_dims, self.query_num_ref*4)) #-----------2.query_num_ref
        reg_branch = nn.Sequential(*reg_branch)

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
        
    def init_weights(self) -> None:
        #1.---------DeformableDETRHead
        super(DeformableDETRHead, self).init_weights()  #希望调用BaseModule.init_weights
        """Initialize weights of the Deformable DETR head."""
        if self.loss_cls.use_sigmoid:
            bias_init = bias_init_with_prob(0.01)
            for m in self.cls_branches:
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.constant_(m.bias, bias_init)
        for m in self.reg_branches:
            constant_init(m[-1], 0, bias=0)
        nn.init.constant_(self.reg_branches[0][-1].bias.data[2:], -2.0)
        if self.as_two_stage:
            for m in self.reg_branches:
                nn.init.constant_(m[-1].bias.data[2:], 0.0)
        #2.---------DeformableDETR
        if not self.share_decoder_withLineHead['shared']:
            for coder in self.encoder, self.decoder:
                for p in coder.parameters():
                    if p.dim() > 1:
                        nn.init.xavier_uniform_(p)
            for m in self.modules():
                if isinstance(m, MultiScaleDeformableAttention):
                    m.init_weights()
        if self.as_two_stage:
            nn.init.xavier_uniform_(self.memory_trans_fc.weight)
            nn.init.xavier_uniform_(self.pos_trans_fc.weight)
        else:
            xavier_init(
                self.reference_points_fc, distribution='uniform', bias=0.)
        if self.encoder is not None:
            normal_(self.level_embed)
    #2.---------copy from /models/dense_heads/deformable_detr_head.py: 
    # def forward(self, hidden_states: Tensor,
    #             references: List[Tensor]) -> Tuple[Tensor, Tensor]:
    #     all_layers_outputs_classes = []
    #     all_layers_outputs_coords = []

    #     for layer_id in range(hidden_states.shape[0]):
    #         reference = inverse_sigmoid(references[layer_id])
    #         # NOTE The last reference will not be used.
    #         hidden_state = hidden_states[layer_id]
    #         outputs_class = self.cls_branches[layer_id](hidden_state)
    #         tmp_reg_preds = self.reg_branches[layer_id](hidden_state)
    #         if reference.shape[-1] == 4:
    #             # When `layer` is 0 and `as_two_stage` of the detector
    #             # is `True`, or when `layer` is greater than 0 and
    #             # `with_box_refine` of the detector is `True`.
    #             tmp_reg_preds += reference
    #         else:
    #             # When `layer` is 0 and `as_two_stage` of the detector
    #             # is `False`, or when `layer` is greater than 0 and
    #             # `with_box_refine` of the detector is `False`.
    #             assert reference.shape[-1] == 2
    #             tmp_reg_preds[..., :2] += reference
    #         outputs_coord = tmp_reg_preds.sigmoid()
    #         all_layers_outputs_classes.append(outputs_class)
    #         all_layers_outputs_coords.append(outputs_coord)

    #     all_layers_outputs_classes = torch.stack(all_layers_outputs_classes)
    #     all_layers_outputs_coords = torch.stack(all_layers_outputs_coords)

    #     return all_layers_outputs_classes, all_layers_outputs_coords
        #2.---------DeformableDETR
    def loss(self, x: Tuple[Tensor], batch_data_samples: Union[list,dict]):
        # 统一多尺度特征的通道数目
        if self.input_proj:
            x = tuple([model(feature) for model, feature in zip(self.input_proj, x)])
        head_inputs_dict = self.forward_transformer(x, batch_data_samples) #hidden_states, references
        hidden_states,references,enc_outputs_class,enc_outputs_coord = head_inputs_dict['hidden_states'],head_inputs_dict['references'],head_inputs_dict['enc_outputs_class'],head_inputs_dict['enc_outputs_coord']
        batch_gt_instances = []
        batch_img_metas = []
        for data_sample in batch_data_samples:
            batch_img_metas.append(data_sample.metainfo)
            batch_gt_instances.append(data_sample.gt_instances)
        outs = self(hidden_states, references)
        loss_inputs = outs + (enc_outputs_class, enc_outputs_coord,
                              batch_gt_instances, batch_img_metas)
        losses = self.loss_by_feat(*loss_inputs)
        return losses
    def predict(self, x: Tuple[Tensor], batch_data_samples: Union[list,dict], rescale: bool = True):
        # 统一多尺度特征的通道数目
        if self.input_proj:
            x = tuple([model(feature) for model, feature in zip(self.input_proj, x)])
        head_inputs_dict = self.forward_transformer(x, batch_data_samples) #hidden_states, references
        hidden_states,references = head_inputs_dict['hidden_states'],head_inputs_dict['references']
        
        batch_img_metas = [
            data_samples.metainfo for data_samples in batch_data_samples
        ]

        outs = self(hidden_states, references)

        results_list = self.predict_by_feat(
            *outs, batch_img_metas=batch_img_metas, rescale=rescale)
        
        return results_list
    
        
    #  same as mmyolo/models/dense_heads/line_head_v4.py
    def forward_transformer(self,
                            img_feats: Tuple[Tensor],
                            batch_data_samples: OptSampleList = None) -> Dict:
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
    # ---------DeformableDETR: no bbox_head
    def pre_decoder(self, memory: Tensor, memory_mask: Tensor,
                    spatial_shapes: Tensor) -> Tuple[Dict, Dict]:
        batch_size, _, c = memory.shape
        if self.as_two_stage:
            output_memory, output_proposals = \
                self.gen_encoder_output_proposals(
                    memory, memory_mask, spatial_shapes)
            enc_outputs_class = self.cls_branches[ # 1.zhou bbox_head.reg_branches-->reg_branches
                self.decoder.num_layers](
                    output_memory)
            if self.query_num_ref > 1:
                repeat_dims = (1,) * (output_proposals.ndim-1)
                enc_outputs_coord_unact = self.reg_branches[  # 2.zhou bbox_head.reg_branches-->reg_branches
                    self.decoder.num_layers](output_memory) + output_proposals.repeat(*repeat_dims, self.query_num_ref)
            else:
                enc_outputs_coord_unact = self.reg_branches[  # 2.zhou bbox_head.reg_branches-->reg_branches
                    self.decoder.num_layers](output_memory) + output_proposals
            if self.query_num_ref > 1:
                enc_outputs_coord = enc_outputs_coord_unact.reshape(*enc_outputs_coord_unact.shape[:-1],self.query_num_ref,-1)
                enc_outputs_coord = torch.mean(enc_outputs_coord,dim=-2).sigmoid()   
            else:
                enc_outputs_coord = enc_outputs_coord_unact.sigmoid()
            # We only use the first channel in enc_outputs_class as foreground,
            # the other (num_classes - 1) channels are actually not used.
            # Its targets are set to be 0s, which indicates the first
            # class (foreground) because we use [0, num_classes - 1] to
            # indicate class labels, background class is indicated by
            # num_classes (similar convention in RPN).
            # See https://github.com/open-mmlab/mmdetection/blob/master/mmdet/models/dense_heads/deformable_detr_head.py#L241 # noqa
            # This follows the official implementation of Deformable DETR.
            topk_proposals = torch.topk(
                enc_outputs_class[..., 0], self.num_queries, dim=1)[1]
            if self.query_num_ref > 1:
                topk_coords_unact = torch.gather(
                    enc_outputs_coord_unact, 1,
                    topk_proposals.unsqueeze(-1).repeat(1, 1, enc_outputs_coord_unact.shape[-1]))
            else:
                topk_coords_unact = torch.gather(
                    enc_outputs_coord_unact, 1,
                    topk_proposals.unsqueeze(-1).repeat(1, 1, 4))
            topk_coords_unact = topk_coords_unact.detach()
            reference_points = topk_coords_unact.sigmoid()
            if self.query_num_ref > 1:
                topk_coords_unact = topk_coords_unact.reshape(*topk_coords_unact.shape[:-1],self.query_num_ref,-1)
                topk_coords_unact = torch.mean(topk_coords_unact,dim=-2)
            pos_trans_out = self.pos_trans_fc(
                self.get_proposal_pos_embed(topk_coords_unact))
            pos_trans_out = self.pos_trans_norm(pos_trans_out)
            query_pos, query = torch.split(pos_trans_out, c, dim=2)
        else:
            enc_outputs_class, enc_outputs_coord = None, None
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
            enc_outputs_class=enc_outputs_class,
            enc_outputs_coord=enc_outputs_coord) if self.training else dict()
        return decoder_inputs_dict, head_inputs_dict
    
    # ---------DeformableDETR: no bbox_head
    def forward_decoder(self, query: Tensor, query_pos: Tensor, memory: Tensor,
                        memory_mask: Tensor, reference_points: Tensor,
                        spatial_shapes: Tensor, level_start_index: Tensor,
                        valid_ratios: Tensor) -> Dict:
        if self.query_num_ref > 1:
            bs, num_query, _ = reference_points.shape
            reference_points = reference_points.view(bs, num_query, self.query_num_ref, -1) 
        inter_states, inter_references = self.decoder(
            query=query,
            value=memory,
            query_pos=query_pos,
            key_padding_mask=memory_mask,  # for cross_attn
            reference_points=reference_points,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
            valid_ratios=valid_ratios,
            reg_branches=self.reg_branches  # 3.zhou bbox_head.reg_branches-->reg_branches
            if self.with_box_refine else None)
        references = [reference_points, *inter_references]
        decoder_outputs_dict = dict(
            hidden_states=inter_states, references=references)
        return decoder_outputs_dict
    
    # ---------DeformableDETR:
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
        input_img_h, input_img_w = batch_input_shape
        img_shape_list = [sample.img_shape for sample in batch_data_samples]
        same_shape_flag = all([
            s[0] == input_img_h and s[1] == input_img_w for s in img_shape_list
        ])
        # support torch2onnx without feeding masks
        if torch.onnx.is_in_onnx_export() or same_shape_flag:
            mlvl_masks = []
            mlvl_pos_embeds = []
            for feat in mlvl_feats:
                mlvl_masks.append(None)
                mlvl_pos_embeds.append(
                    self.positional_encoding(None, input=feat))
        else:
            masks = mlvl_feats[0].new_ones(
                (batch_size, input_img_h, input_img_w))
            for img_id in range(batch_size):
                img_h, img_w = img_shape_list[img_id]
                masks[img_id, :img_h, :img_w] = 0
            # NOTE following the official DETR repo, non-zero
            # values representing ignored positions, while
            # zero values means valid positions.

            mlvl_masks = []
            mlvl_pos_embeds = []
            for feat in mlvl_feats:
                mlvl_masks.append(
                    F.interpolate(masks[None], size=feat.shape[-2:]).to(
                        torch.bool).squeeze(0))
                mlvl_pos_embeds.append(
                    self.positional_encoding(mlvl_masks[-1]))

        feat_flatten = []
        lvl_pos_embed_flatten = []
        mask_flatten = []
        spatial_shapes = []
        for lvl, (feat, mask, pos_embed) in enumerate(
                zip(mlvl_feats, mlvl_masks, mlvl_pos_embeds)):
            batch_size, c, h, w = feat.shape
            spatial_shape = torch._shape_as_tensor(feat)[2:].to(feat.device)
            # [bs, c, h_lvl, w_lvl] -> [bs, h_lvl*w_lvl, c]
            feat = feat.view(batch_size, c, -1).permute(0, 2, 1)
            pos_embed = pos_embed.view(batch_size, c, -1).permute(0, 2, 1)
            if self.encoder is not None:
                lvl_pos_embed = pos_embed + self.level_embed[lvl].view(1, 1, -1)
            # [bs, h_lvl, w_lvl] -> [bs, h_lvl*w_lvl]
            if mask is not None:
                mask = mask.flatten(1)

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
        if mask_flatten[0] is not None:
            mask_flatten = torch.cat(mask_flatten, 1)
        else:
            mask_flatten = None

        # (num_level, 2)
        spatial_shapes = torch.cat(spatial_shapes).view(-1, 2)
        level_start_index = torch.cat((
            spatial_shapes.new_zeros((1, )),  # (num_level)
            spatial_shapes.prod(1).cumsum(0)[:-1]))
        if mlvl_masks[0] is not None:
            valid_ratios = torch.stack(  # (bs, num_level, 2)
                [self.get_valid_ratio(m) for m in mlvl_masks], 1)
        else:
            valid_ratios = mlvl_feats[0].new_ones(batch_size, len(mlvl_feats),
                                                  2)

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
    def gen_encoder_output_proposals(
            self, memory: Tensor, memory_mask: Tensor,
            spatial_shapes: Tensor) -> Tuple[Tensor, Tensor]:
        """Generate proposals from encoded memory. The function will only be
        used when `as_two_stage` is `True`.

        Args:
            memory (Tensor): The output embeddings of the Transformer encoder,
                has shape (bs, num_feat_points, dim).
            memory_mask (Tensor): ByteTensor, the padding mask of the memory,
                has shape (bs, num_feat_points).
            spatial_shapes (Tensor): Spatial shapes of features in all levels,
                has shape (num_levels, 2), last dimension represents (h, w).

        Returns:
            tuple: A tuple of transformed memory and proposals.

            - output_memory (Tensor): The transformed memory for obtaining
              top-k proposals, has shape (bs, num_feat_points, dim).
            - output_proposals (Tensor): The inverse-normalized proposal, has
              shape (batch_size, num_keys, 4) with the last dimension arranged
              as (cx, cy, w, h).
        """

        bs = memory.size(0)
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
            wh = torch.ones_like(grid) * 0.05 * (2.0**lvl)
            proposal = torch.cat((grid, wh), -1).view(bs, -1, 4)
            proposals.append(proposal)
            _cur += (H * W)
        output_proposals = torch.cat(proposals, 1)
        # do not use `all` to make it exportable to onnx
        output_proposals_valid = (
            (output_proposals > 0.01) & (output_proposals < 0.99)).sum(
                -1, keepdim=True) == output_proposals.shape[-1]
        # inverse_sigmoid
        output_proposals = torch.log(output_proposals / (1 - output_proposals))
        if memory_mask is not None:
            output_proposals = output_proposals.masked_fill(
                memory_mask.unsqueeze(-1), float('inf'))
        output_proposals = output_proposals.masked_fill(
            ~output_proposals_valid, float('inf'))

        output_memory = memory
        if memory_mask is not None:
            output_memory = output_memory.masked_fill(
                memory_mask.unsqueeze(-1), float(0))
        output_memory = output_memory.masked_fill(~output_proposals_valid,
                                                  float(0))
        output_memory = self.memory_trans_fc(output_memory)
        output_memory = self.memory_trans_norm(output_memory)
        # [bs, sum(hw), 2]
        return output_memory, output_proposals
    
    @staticmethod
    def get_proposal_pos_embed(proposals: Tensor,
                               num_pos_feats: int = 128,
                               temperature: int = 10000) -> Tensor:
        """Get the position embedding of the proposal.

        Args:
            proposals (Tensor): Not normalized proposals, has shape
                (bs, num_queries, 4) with the last dimension arranged as
                (cx, cy, w, h).
            num_pos_feats (int, optional): The feature dimension for each
                position along x, y, w, and h-axis. Note the final returned
                dimension for each position is 4 times of num_pos_feats.
                Default to 128.
            temperature (int, optional): The temperature used for scaling the
                position embedding. Defaults to 10000.

        Returns:
            Tensor: The position embedding of proposal, has shape
            (bs, num_queries, num_pos_feats * 4), with the last dimension
            arranged as (cx, cy, w, h)
        """
        scale = 2 * math.pi
        dim_t = torch.arange(
            num_pos_feats, dtype=torch.float32, device=proposals.device)
        if torch.__version__ <= '1.8.0': #change by zhou, 因为4090上老UserWarning，实际上对negative values才有影响
            dim_t = temperature**(2 * (dim_t // 2) / num_pos_feats)
        else:
            dim_t = temperature**(2 * torch.div(dim_t, 2, rounding_mode='floor') / num_pos_feats)
        # N, L, 4
        proposals = proposals.sigmoid() * scale
        # N, L, 4, 128
        pos = proposals[:, :, :, None] / dim_t
        # N, L, 4, 64, 2
        pos = torch.stack((pos[:, :, :, 0::2].sin(), pos[:, :, :, 1::2].cos()),
                          dim=4).flatten(2)
        return pos
