"""
Latest version 2025-04-01-19:00:00, add onlyPos in 
Created on Sat Jun  1 21:49:21 2024

@author: zcf
"""
from typing import Dict, Tuple
import torch
from mmengine.structures import InstanceData
from torch import Tensor
from mmdet.structures import SampleList
from mmdet.structures.bbox import (bbox_cxcywh_to_xyxy, bbox_overlaps,
                                   bbox_xyxy_to_cxcywh)
from mmdet.utils import InstanceList
from mmdet.models.utils import multi_apply
from mmdet.models import inverse_sigmoid
from mmdet.models import CdnQueryGenerator
from mmdet.utils import OptConfigType
from mmyolo.models.data_preprocessors.line_data_preprocessor import prepare_line_points
from typing import Union
# 还有不存在line的情况
class LineCdnQueryGenerator(CdnQueryGenerator):
    def __init__(self,
                 num_classes: int,
                 embed_dims: int,
                 num_matching_queries: int,
                 label_noise_scale: float = 0.5,
                 box_noise_scale: float = 1.0,
                 group_cfg: OptConfigType = None,
                 **kwargs) -> None:
        self.num_classes_ori = num_classes# TODO [-1:] the embedding of `unknown` class is used, 
        if kwargs.get('useUnkCls',None) and label_noise_scale > 0:
            num_classes = num_classes+1
            print('num_classes',num_classes)
        super().__init__(num_classes=num_classes, embed_dims=embed_dims, num_matching_queries=num_matching_queries,
                         label_noise_scale=label_noise_scale, box_noise_scale=box_noise_scale, group_cfg=group_cfg)
        self.lineMin = kwargs.get('lineMin',None) #w,h
        self.onlyPos = kwargs.get('onlyPos',False) 
        self.num_times = 1 if self.onlyPos else 2 # 只生成正样本 # 生成正负样本
    # def init_weights(self) -> None:
    #     torch.nn.init.xavier_uniform_(self.label_embedding.weight)
        
        
    def __call__(self, batch_data_samples: SampleList) -> tuple:
        # normalize bbox and collate ground truth (gt)
        gt_labels_list = []
        gt_bboxes_list = []
        for sample in batch_data_samples:
            img_h, img_w = sample.img_shape
            #-------------change by 240601
            # bboxes = sample.gt_instances.bboxes
            bboxes = sample.gt_line_instances.line_points
            bs, LineNumPts = bboxes.shape[0], bboxes.shape[-2]
            if bs ==0:
                bboxes = bboxes[:, [0, -1], :].view(bs,4) #xyxy
            else:
                if LineNumPts > 2:
                    bboxes = bboxes[:, [0, -1], :].view(bs,-1) #xyxy
                
            factor = bboxes.new_tensor([img_w, img_h, img_w,
                                        img_h]).unsqueeze(0)
            bboxes_normalized = bboxes / factor
            gt_bboxes_list.append(bboxes_normalized)
            # gt_labels_list.append(sample.gt_instances.labels)
            gt_labels_list.append(sample.gt_line_instances.line_labels) #-------------change by 240601
        gt_labels = torch.cat(gt_labels_list)  # (num_target_total, 4)
        gt_bboxes = torch.cat(gt_bboxes_list)
        num_target_list = [len(bboxes) for bboxes in gt_bboxes_list]
        max_num_target = max(num_target_list)
        num_groups = self.get_num_groups(max_num_target)
        
        dn_label_query = self.generate_dn_label_query(gt_labels, num_groups)
        dn_bbox_query = self.generate_dn_bbox_query(gt_bboxes, num_groups) # 不知道最后为什么又转xywh, 去掉，因为要和two_stage的reference_points匹配

        # The `batch_idx` saves the batch index of the corresponding sample
        # for each target, has shape (num_target_total).
        batch_idx = torch.cat([
            torch.full_like(t.long(), i) for i, t in enumerate(gt_labels_list)
        ])
        dn_label_query, dn_bbox_query = self.collate_dn_queries(
            dn_label_query, dn_bbox_query, batch_idx, len(batch_data_samples),
            num_groups)

        attn_mask = self.generate_dn_mask(
            max_num_target, num_groups, device=dn_label_query.device) #,为什么最后的都是0, 因为是CDN以外的query

        dn_meta = dict(
            num_denoising_queries=int(max_num_target * self.num_times * num_groups),
            num_denoising_groups=num_groups)

        return dn_label_query, dn_bbox_query, attn_mask, dn_meta
    
    #----------------change only for self.onlyPos (self.num_times) and self.label_noise_scale == 0
    def generate_dn_label_query(self, gt_labels: Tensor,
                                num_groups: int) -> Tensor:
        
        assert self.label_noise_scale >= 0
        gt_labels_expand = gt_labels.repeat(self.num_times * num_groups,
                                            1).view(-1)  # zcf-------------2->self.num_times
        if self.label_noise_scale > 0:
            p = torch.rand_like(gt_labels_expand.float())
            chosen_indice = torch.nonzero(p < (self.label_noise_scale * 0.5)).view(
                -1)  # Note `* 0.5`
            new_labels = torch.randint_like(chosen_indice, 0, self.num_classes)
            noisy_labels_expand = gt_labels_expand.scatter(0, chosen_indice,
                                                           new_labels)
        else:
            noisy_labels_expand = gt_labels_expand
        dn_label_query = self.label_embedding(noisy_labels_expand)
        return dn_label_query
    #----------------change only for self.onlyPos (self.num_times)
    def collate_dn_queries(self, input_label_query: Tensor,
                           input_bbox_query: Tensor, batch_idx: Tensor,
                           batch_size: int, num_groups: int) -> Tuple[Tensor]:
        device = input_label_query.device
        num_target_list = [
            torch.sum(batch_idx == idx) for idx in range(batch_size)
        ]
        max_num_target = max(num_target_list)
        num_denoising_queries = int(max_num_target * self.num_times * num_groups)

        map_query_index = torch.cat([
            torch.arange(num_target, device=device)
            for num_target in num_target_list
        ])
        map_query_index = torch.cat([
            map_query_index + max_num_target * i for i in range(self.num_times * num_groups)
        ]).long()
        batch_idx_expand = batch_idx.repeat(self.num_times * num_groups, 1).view(-1)
        mapper = (batch_idx_expand, map_query_index)

        batched_label_query = torch.zeros(
            batch_size, num_denoising_queries, self.embed_dims, device=device)
        batched_bbox_query = torch.zeros(
            batch_size, num_denoising_queries, 4, device=device)

        batched_label_query[mapper] = input_label_query
        batched_bbox_query[mapper] = input_bbox_query
        return batched_label_query, batched_bbox_query
    #----------------change only for self.onlyPos (self.num_times)
    def generate_dn_mask(self, max_num_target: int, num_groups: int,
                         device: Union[torch.device, str]) -> Tensor:
        """
        .. code:: text

                        0 0 0 0 1 1 1 1 0 0 0 0 0
                        0 0 0 0 1 1 1 1 0 0 0 0 0
                        0 0 0 0 1 1 1 1 0 0 0 0 0
                        0 0 0 0 1 1 1 1 0 0 0 0 0
                        1 1 1 1 0 0 0 0 0 0 0 0 0
                        1 1 1 1 0 0 0 0 0 0 0 0 0
                        1 1 1 1 0 0 0 0 0 0 0 0 0
                        1 1 1 1 0 0 0 0 0 0 0 0 0
                        1 1 1 1 1 1 1 1 0 0 0 0 0
                        1 1 1 1 1 1 1 1 0 0 0 0 0
                        1 1 1 1 1 1 1 1 0 0 0 0 0
                        1 1 1 1 1 1 1 1 0 0 0 0 0
                        1 1 1 1 1 1 1 1 0 0 0 0 0
         max_num_target |_|           |_________| num_matching_queries
                        |_____________| num_denoising_queries

               1 -> True  (Masked), means 'can not see'.
               0 -> False (UnMasked), means 'can see'.
        """
        num_denoising_queries = int(max_num_target * self.num_times * num_groups)
        num_queries_total = num_denoising_queries + self.num_matching_queries
        attn_mask = torch.zeros(
            num_queries_total,
            num_queries_total,
            device=device,
            dtype=torch.bool)
        # Make the matching part cannot see the denoising groups
        attn_mask[num_denoising_queries:, :num_denoising_queries] = True
        # Make the denoising groups cannot see each other
        for i in range(num_groups):
            # Mask rows of one group per step.
            row_scope = slice(max_num_target * self.num_times * i,
                              max_num_target * self.num_times * (i + 1))
            left_scope = slice(max_num_target * self.num_times * i)
            right_scope = slice(max_num_target * self.num_times * (i + 1),
                                num_denoising_queries)
            attn_mask[row_scope, right_scope] = True
            attn_mask[row_scope, left_scope] = True
        return attn_mask

    
    def generate_dn_bbox_query(self, gt_bboxes: Tensor,
                               num_groups: int) -> Tensor:
        assert self.box_noise_scale >= 0
        device = gt_bboxes.device
        
        # expand gt_bboxes as groups
        gt_bboxes_expand = gt_bboxes.repeat(self.num_times * num_groups, 1)  # xyxy

        # obtain index of negative queries in gt_bboxes_expand
        positive_idx = torch.arange(
            len(gt_bboxes), dtype=torch.long, device=device)
        positive_idx = positive_idx.unsqueeze(0).repeat(num_groups, 1)
        positive_idx += self.num_times * len(gt_bboxes) * torch.arange(
            num_groups, dtype=torch.long, device=device)[:, None]
        positive_idx = positive_idx.flatten()
        if not self.onlyPos:
            negative_idx = positive_idx + len(gt_bboxes)

        # determine the sign of each element in the random part of the added
        # noise to be positive or negative randomly.
        rand_sign = torch.randint_like(
            gt_bboxes_expand, low=0, high=2,
            dtype=torch.float32) * 2.0 - 1.0  # [low, high), 1 or -1, randomly

        # calculate the random part of the added noise
        rand_part = torch.rand_like(gt_bboxes_expand)  # [0, 1)
        if not self.onlyPos:
            rand_part[negative_idx] += 1.0  # pos: [0, 1); neg: [1, 2)
        rand_part *= rand_sign  # pos: (-1, 1); neg: (-2, -1] U [1, 2)

        # add noise to the bboxes
        bboxes_whwh = bbox_xyxy_to_cxcywh(gt_bboxes_expand)[:, 2:].repeat(1, 2)
        # 主要是两个点跳来跳去，跳动的范围和wh和box_noise_scale有关, 如果和wh有关，则水平线和垂直线的跳动还在原来的线上，只不过长短变了
        if self.lineMin is not None:
            bboxes_whwh = torch.maximum(bboxes_whwh, bboxes_whwh.new_tensor(self.lineMin*2).unsqueeze(0))
        noisy_bboxes_expand = gt_bboxes_expand + torch.mul(
            rand_part, bboxes_whwh) * self.box_noise_scale / 2  # xyxy, 
        noisy_bboxes_expand = noisy_bboxes_expand.clamp(min=0.0, max=1.0)
        # noisy_bboxes_expand = bbox_xyxy_to_cxcywh(noisy_bboxes_expand) #-------------change by 240601

        dn_bbox_query = inverse_sigmoid(noisy_bboxes_expand, eps=1e-3)
        return dn_bbox_query
    # copy from dino_head.py
    @staticmethod
    def split_outputs(all_layers_cls_scores: Tensor,
                      all_layers_bbox_preds: Tensor,
                      dn_meta: Dict[str, int]) -> Tuple[Tensor]:
        """Split outputs of the denoising part and the matching part.

        For the total outputs of `num_queries_total` length, the former
        `num_denoising_queries` outputs are from denoising queries, and
        the rest `num_matching_queries` ones are from matching queries,
        where `num_queries_total` is the sum of `num_denoising_queries` and
        `num_matching_queries`.

        Args:
            all_layers_cls_scores (Tensor): Classification scores of all
                decoder layers, has shape (num_decoder_layers, bs,
                num_queries_total, cls_out_channels).
            all_layers_bbox_preds (Tensor): Regression outputs of all decoder
                layers. Each is a 4D-tensor with normalized coordinate format
                (cx, cy, w, h) and has shape (num_decoder_layers, bs,
                num_queries_total, 4).
            dn_meta (Dict[str, int]): The dictionary saves information about
              group collation, including 'num_denoising_queries' and
              'num_denoising_groups'.

        Returns:
            Tuple[Tensor]: a tuple containing the following outputs.

            - all_layers_matching_cls_scores (Tensor): Classification scores
              of all decoder layers in matching part, has shape
              (num_decoder_layers, bs, num_matching_queries, cls_out_channels).
            - all_layers_matching_bbox_preds (Tensor): Regression outputs of
              all decoder layers in matching part. Each is a 4D-tensor with
              normalized coordinate format (cx, cy, w, h) and has shape
              (num_decoder_layers, bs, num_matching_queries, 4).
            - all_layers_denoising_cls_scores (Tensor): Classification scores
              of all decoder layers in denoising part, has shape
              (num_decoder_layers, bs, num_denoising_queries,
              cls_out_channels).
            - all_layers_denoising_bbox_preds (Tensor): Regression outputs of
              all decoder layers in denoising part. Each is a 4D-tensor with
              normalized coordinate format (cx, cy, w, h) and has shape
              (num_decoder_layers, bs, num_denoising_queries, 4).
        """
        num_denoising_queries = dn_meta['num_denoising_queries']
        if dn_meta is not None:
            all_layers_denoising_cls_scores = \
                all_layers_cls_scores[:, :, : num_denoising_queries, :]
            all_layers_denoising_bbox_preds = \
                all_layers_bbox_preds[:, :, : num_denoising_queries, :]
            all_layers_matching_cls_scores = \
                all_layers_cls_scores[:, :, num_denoising_queries:, :]
            all_layers_matching_bbox_preds = \
                all_layers_bbox_preds[:, :, num_denoising_queries:, :]
        else:
            all_layers_denoising_cls_scores = None
            all_layers_denoising_bbox_preds = None
            all_layers_matching_cls_scores = all_layers_cls_scores
            all_layers_matching_bbox_preds = all_layers_bbox_preds
        return (all_layers_matching_cls_scores, all_layers_matching_bbox_preds,
                all_layers_denoising_cls_scores,
                all_layers_denoising_bbox_preds)
    # copy from dino_head.py
    def get_dn_targets(self, batch_gt_instances: InstanceList,
                       batch_img_metas: dict, dn_meta: Dict[str,
                                                            int]) -> tuple:
        """Get targets in denoising part for a batch of images.

        Args:
            batch_gt_instances (list[:obj:`InstanceData`]): Batch of
                gt_instance. It usually includes ``bboxes`` and ``labels``
                attributes.
            batch_img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            dn_meta (Dict[str, int]): The dictionary saves information about
              group collation, including 'num_denoising_queries' and
              'num_denoising_groups'. It will be used for split outputs of
              denoising and matching parts and loss calculation.

        Returns:
            tuple: a tuple containing the following targets.

            - labels_list (list[Tensor]): Labels for all images.
            - label_weights_list (list[Tensor]): Label weights for all images.
            - bbox_targets_list (list[Tensor]): BBox targets for all images.
            - bbox_weights_list (list[Tensor]): BBox weights for all images.
            - num_total_pos (int): Number of positive samples in all images.
            - num_total_neg (int): Number of negative samples in all images.
        """
        (labels_list, label_weights_list, bbox_targets_list, bbox_weights_list,
         pos_inds_list, neg_inds_list) = multi_apply(
             self._get_dn_targets_single,
             batch_gt_instances,
             batch_img_metas,
             dn_meta=dn_meta)
        num_total_pos = sum((inds.numel() for inds in pos_inds_list))
        num_total_neg = sum((inds.numel() for inds in neg_inds_list))
        return (labels_list, label_weights_list, bbox_targets_list,
                bbox_weights_list, num_total_pos, num_total_neg)
    # copy from dino_head.py
    def _get_dn_targets_single(self, gt_instances: InstanceData,
                               img_meta: dict, dn_meta: Dict[str,
                                                             int]) -> tuple:
        """Get targets in denoising part for one image.

        Args:
            gt_instances (:obj:`InstanceData`): Ground truth of instance
                annotations. It should includes ``bboxes`` and ``labels``
                attributes.
            img_meta (dict): Meta information for one image.
            dn_meta (Dict[str, int]): The dictionary saves information about
              group collation, including 'num_denoising_queries' and
              'num_denoising_groups'. It will be used for split outputs of
              denoising and matching parts and loss calculation.

        Returns:
            tuple[Tensor]: a tuple containing the following for one image.

            - labels (Tensor): Labels of each image.
            - label_weights (Tensor]): Label weights of each image.
            - bbox_targets (Tensor): BBox targets of each image.
            - bbox_weights (Tensor): BBox weights of each image.
            - pos_inds (Tensor): Sampled positive indices for each image.
            - neg_inds (Tensor): Sampled negative indices for each image.
        """
        gt_bboxes = gt_instances.line_points
        # 统一预测和gt的点数为指定个数
        if gt_bboxes.shape[-2]!= self.points_for_lossMetric:
            # if outputs_coords_ratios is not None and outputs_coords_ratios!=[None]: #TODO
            #     #[num_gt,2,2]*[num_queries,num_points]
            #     #[num_gt,2]*[num_queries,num_points](num_queries,1,num_points,1)-->[num_queries,num_gt,num_points,2]
            #     start_point = gt_bboxes[:,0].unsqueeze(0).unsqueeze(-2) # [1,num_gt,1,2]
            #     end_point = gt_bboxes[:,1].unsqueeze(0).unsqueeze(-2)  # [1,num_gt,1,2]
            #     gt_bboxes = start_point + (end_point - start_point) * outputs_coords_ratios.unsqueeze(-1).unsqueeze(1) # ratios, [num_queries,num_points]
            #     gt_bboxes = gt_bboxes.flatten(-2).permute(1,0,2) 
            # else:
            # gt_bboxes = (-1, num_pts, self.pts_dim) -->#[-1,line_points_inter_method,pts_dim]
            gt_bboxes = prepare_line_points(gt_bboxes,self.line_points_inter_method,self.points_for_lossMetric)
            gt_bboxes = gt_bboxes.flatten(-2)
        else:
            gt_bboxes = gt_bboxes.flatten(-2)
        gt_labels = gt_instances.line_labels
        # gt_bboxes = gt_instances.bboxes
        # gt_labels = gt_instances.labels


        num_groups = dn_meta['num_denoising_groups']
        num_denoising_queries = dn_meta['num_denoising_queries']
        num_queries_each_group = int(num_denoising_queries / num_groups)
        device = gt_bboxes.device

        if len(gt_labels) > 0:
            t = torch.arange(len(gt_labels), dtype=torch.long, device=device)
            t = t.unsqueeze(0).repeat(num_groups, 1)
            pos_assigned_gt_inds = t.flatten()
            pos_inds = torch.arange(
                num_groups, dtype=torch.long, device=device)
            pos_inds = pos_inds.unsqueeze(1) * num_queries_each_group + t
            pos_inds = pos_inds.flatten()
        else:
            pos_inds = pos_assigned_gt_inds = \
                gt_bboxes.new_tensor([], dtype=torch.long)
        if self.onlyPos:
            neg_inds = gt_bboxes.new_tensor([], dtype=torch.long)
        else:  
            neg_inds = pos_inds + num_queries_each_group // 2

        # label targets
        labels = gt_bboxes.new_full((num_denoising_queries, ),
                                    self.num_classes_ori, #TODO [-1:] the embedding of `unknown` class is used, 
                                    dtype=torch.long)
        labels[pos_inds] = gt_labels[pos_assigned_gt_inds]
        label_weights = gt_bboxes.new_ones(num_denoising_queries)

        # bbox targets
        # bbox_targets = torch.zeros(num_denoising_queries, 4, device=device)
        # bbox_weights = torch.zeros(num_denoising_queries, 4, device=device)
        bbox_targets = torch.zeros(num_denoising_queries, gt_bboxes.shape[-1], device=device)
        bbox_weights = torch.zeros(num_denoising_queries, gt_bboxes.shape[-1], device=device)
        # bbox_weights[pos_inds] = 1.0
        bbox_weights[pos_inds,:] = torch.full_like(bbox_weights[pos_inds, :], 1.0) #for 4090 deterministic
        # img_h, img_w = img_meta['img_shape']

        # DETR regress the relative position of boxes (cxcywh) in the image.
        # Thus the learning target should be normalized by the image size, also
        # the box format should be converted from defaultly x1y1x2y2 to cxcywh.
        # factor = gt_bboxes.new_tensor([img_w, img_h, img_w,
        #                                img_h]).unsqueeze(0)
        # gt_bboxes_normalized = gt_bboxes / factor
        # gt_bboxes_targets = bbox_xyxy_to_cxcywh(gt_bboxes_normalized)
        # bbox_targets[pos_inds] = gt_bboxes_targets.repeat([num_groups, 1])
        bbox_targets[pos_inds] = gt_bboxes.repeat([num_groups, 1]) #TODO, 不除以factor，保持统一

        return (labels, label_weights, bbox_targets, bbox_weights, pos_inds,
                neg_inds)