"""
Latest version 2025-04-26-17:00:00
Created on Tue Jan 16 16:42:29 2024

@author: zcf
20250405: 比LineCocoMetric多：cls_infos
"""
import itertools
import os.path as osp
import tempfile
from collections import OrderedDict
from typing import Dict, List, Optional, Sequence, Union

import numpy as np
import torch
from mmengine.fileio import dump, load
from mmengine.logging import MMLogger
from terminaltables import AsciiTable
from mmdet.datasets.api_wrappers import COCO, COCOeval
from mmyolo.registry import METRICS
from mmdet.structures.mask import encode_mask_results
from mmdet.evaluation import CocoMetric
from tqdm import tqdm
import copy
from .distance import pairwise, chamfer_distance, frechet_distance # , iou_distance
from mmyolo.models import prepare_line_points
import sys
from mmcv.ops import nms
from mmengine.fileio import get_local_path
from mmdet.evaluation import eval_map
import networkx as nx
import os
from .line_metric_utils import _mAP_over_threshold

right_start = 6450 - 3280
left_end = 3280
from mmcv.ops import batched_nms

THRESHOLDS_FRECHET = [float(x) for x in range(10, 201, 5)] 
lineMin = 10
def bbox_overimg_iou(box1, box2, right_start, left_end, merge_iou_thr=0.5, debug_label=False):
    """
    计算一组bbox和一组bbox在交叠区域(交叠区域x方向起点和终点为right_start,left_end)的IoU
    box1: [N, 4]
    box2: [M, 4]
    """
    x1_internal = torch.maximum(box1[:, None, 0], box2[:, 0]) # (N, M)
    y1_internal = torch.maximum(box1[:, None, 1], box2[:, 1]) # (N, M)
    x2_internal = torch.minimum(box1[:, None, 2], box2[:, 2]) # (N, M)
    y2_internal = torch.minimum(box1[:, None, 3], box2[:, 3]) # (N, M)
    # 计算交叉区域的面积
    intersection_area = torch.maximum(torch.tensor(0), x2_internal - x1_internal) * torch.maximum(torch.tensor(0), y2_internal - y1_internal)  # (N, M)
    x1 = torch.minimum(box1[:, None, 0], box2[:, 0]) # (N, M)
    y1 = torch.minimum(box1[:, None, 1], box2[:, 1]) # (N, M)
    x2 = torch.maximum(box1[:, None, 2], box2[:, 2]) # (N, M)
    y2 = torch.maximum(box1[:, None, 3], box2[:, 3]) # (N, M)
    x1 = torch.maximum(x1, torch.tensor(right_start)) # (N, M)
    x2 = torch.minimum(x2, torch.tensor(left_end)) # (N, M)
    # 计算并集的面积
    area = (x2 - x1) * (y2 - y1) # (N, M)
    iou = intersection_area / area # (N, M)
    # # 计算并集的面积
    # intersectionx = torch.maximum(torch.tensor(0), (x2_internal-x1_internal)/(x2-x1))
    # intersectiony = torch.maximum(torch.tensor(0), (y2_internal-y1_internal)/(y2-y1))
    # # 找到 iou > 0.5 的位置
    # mask = iou > 0.3  
    # # 使用布尔索引获取满足条件的 intersectionx 和 intersectiony 值
    # x_values = intersectionx[mask]
    # y_values = intersectiony[mask]
    # iou = (intersectionx>merge_iou_thr)*(intersectionx+merge_iou_thr)*intersectiony #merge_iou_thr=0.5
    # iou = (intersectionx>0.1)*intersectiony #merge_iou_thr=0.5
    # if debug_label and x_values.shape[0]>0:
    #     print('x_iou',torch.mean(x_values),'y_iou',torch.mean(y_values))
    return iou
def merge_detections(left_preds,right_preds,merge_score_thr=0.0,merge_iou_thr=0.5):
    """
    合并两张图像的检测结果
    left_preds: 左图像的检测框坐标+分数+类别 [n1, 4+1+1] x1y1x2y2
    right_preds: 右图像的检测框坐标+分数+类别 [n2, 4+1+1]
    merge_score_thr: 合并时使用的分数阈值
    """
    # right_start = 6450 - 3280
    # left_end = 3280
    # from mmcv.ops import batched_nms
    def to_tensor(arr):
        return torch.from_numpy(arr) if isinstance(arr, np.ndarray) else arr
    left_preds = to_tensor(left_preds)
    right_preds = to_tensor(right_preds)
    # 过滤低分数的检测框
    left_keep = left_preds[:,4] >= merge_score_thr
    right_keep = right_preds[:,4] >= merge_score_thr
    # 找到落在交叠区域内部的检测框
    left_in_overlap = torch.logical_and(left_preds[:, 2] > right_start, left_keep)
    right_in_overlap = torch.logical_and(right_preds[:, 0] < left_end, right_keep)
    labels_i = set(left_preds[:,-1][left_in_overlap].tolist()) & set(right_preds[:,-1][right_in_overlap].tolist())
    labels_u = set(left_preds[:,-1][left_in_overlap].tolist()) | set(right_preds[:,-1][right_in_overlap].tolist())
    # 不需要过滤，直接返回
    if sum(left_in_overlap) == 0 or sum(right_in_overlap) == 0 or len(labels_i)==0:
        all_preds = torch.cat([left_preds, right_preds], dim=0)
        # 按照预测分数从大到小排序
        return all_preds[torch.argsort(-all_preds[:,4])]
    # 处理不同类别的检测框
    # 跨越交叠区域的检测框
    left_in_overlap_large = torch.logical_and(left_preds[:, 0] < right_start, left_in_overlap)
    right_in_overlap_large = torch.logical_and(right_preds[:, 2] > left_end, right_in_overlap)
    if sum(left_in_overlap_large) == 0 and sum(right_in_overlap_large) == 0:
        deal_preds = torch.cat([left_preds[left_in_overlap], right_preds[right_in_overlap]], dim=0)
    else:
        deal_preds = [] 
        for label in labels_u:
            left_deal_preds = left_preds[torch.logical_and(left_in_overlap, left_preds[:,-1] == label)]
            right_deal_preds = right_preds[torch.logical_and(right_in_overlap, right_preds[:,-1] == label)]
            if label not in labels_i:
                deal_preds.append(torch.cat([left_deal_preds, right_deal_preds], dim=0))
                continue
            # debug_label = False
            # if label ==3:
            #     debug_label= True
                # print('debug')
            left_overBor = left_in_overlap_large[torch.logical_and(left_in_overlap, left_preds[:,-1] == label)]
            right_overBor = right_in_overlap_large[torch.logical_and(right_in_overlap, right_preds[:,-1] == label)]     
            if sum(left_overBor) == 0 and sum(right_overBor) == 0:
                deal_preds.append(torch.cat([left_deal_preds, right_deal_preds], dim=0))
                continue
            # 按照预测分数从大到小排序
            left_deal_preds = left_deal_preds[torch.argsort(-left_deal_preds[:, 4])]
            right_deal_preds = right_deal_preds[torch.argsort(-right_deal_preds[:, 4])]
            # 计算交并比 IOU1 = 相交部分 / 在交叠区域的合并部分
            ious = bbox_overimg_iou(left_deal_preds[:,:4],right_deal_preds[:,:4],right_start,left_end)  #,merge_iou_thr,debug_label
            mask = (~left_overBor.unsqueeze(1)) & (~right_overBor.unsqueeze(0)) #-----------------------240620
            ious[mask] = 0  #-----------------------240620
            # 处理被截断的预测框
            def merge_predictions(preds, other_preds, iou_threshold): #-----------------------240620
                """
                根据给定的IOU阈值合并预测。
                :param preds: 主预测张量 [N, 6], 其中前四列是坐标，第五列是分数，第六列是标签。
                :param other_preds: 对比预测张量 [M, 6]。
                :param ious: 预测之间的IOU矩阵 [N, M]。
                :param iou_threshold: IOU合并阈值。
                :return: 合并后的预测张量。
                """
                #-----------------------240620
                valid_merge_mask = ious > iou_threshold
                if not valid_merge_mask.any():
                    return torch.cat([preds, other_preds], dim=0)
                
                # 1)保留未合并的框
                remaining_preds = preds[~valid_merge_mask.any(dim=1)]
                remaining_other_preds = other_preds[~valid_merge_mask.any(dim=0)]
                # 2)合并的框=======扩展维度以进行并集计算
                preds = preds.unsqueeze(1)
                other_preds = other_preds.unsqueeze(0)
                merged_x1 = torch.min(preds[..., 0], other_preds[..., 0])
                merged_y1 = torch.min(preds[..., 1], other_preds[..., 1])
                merged_x2 = torch.max(preds[..., 2], other_preds[..., 2])
                merged_y2 = torch.max(preds[..., 3], other_preds[..., 3])
                score = (preds[..., 4] + other_preds[..., 4]) / 2
                categories = (preds[..., 5] + other_preds[..., 5]) / 2 #因为是相同的类别，这样做，只是为了保持统一的尺寸
                merged_boxes = torch.stack([merged_x1, merged_y1, merged_x2, merged_y2, score, categories], dim=-1)
                # 提取需要的合并框
                merged_boxes = merged_boxes[valid_merge_mask]  
                
                final_merged_preds = torch.cat([merged_boxes, remaining_preds, remaining_other_preds], dim=0)
                #-----------------------240620
                return final_merged_preds
            deal_preds.append(merge_predictions(left_deal_preds, right_deal_preds, merge_iou_thr)) #-----------------------240620
        deal_preds = torch.cat(deal_preds,dim=0)
    # 执行NMS
    nms_cfg = dict(type='nms', iou_threshold=0.7)
    det_bboxes, keep_idxs = batched_nms(deal_preds[:,:4].float(), deal_preds[:,4].float(), deal_preds[:,-1].int(), nms_cfg)
    deal_preds = deal_preds[keep_idxs]
    all_preds = torch.cat([left_preds[~left_in_overlap],right_preds[~right_in_overlap],deal_preds], axis=0)
    return all_preds

def merge_detections_v0(left_preds,right_preds,merge_score_thr=0.0,merge_iou_thr=0.5):
    """
    合并两张图像的检测结果
    left_preds: 左图像的检测框坐标+分数+类别 [n1, 4+1+1] x1y1x2y2
    right_preds: 右图像的检测框坐标+分数+类别 [n2, 4+1+1]
    merge_score_thr: 合并时使用的分数阈值
    """
    # right_start = 6450 - 3280
    # left_end = 3280
    # from mmcv.ops import batched_nms
    def to_tensor(arr):
        return torch.from_numpy(arr) if isinstance(arr, np.ndarray) else arr
    left_preds = to_tensor(left_preds)
    right_preds = to_tensor(right_preds)
    # 过滤低分数的检测框
    left_keep = left_preds[:,4] >= merge_score_thr
    right_keep = right_preds[:,4] >= merge_score_thr
    # 找到落在交叠区域内部的检测框
    left_in_overlap = torch.logical_and(left_preds[:, 2] > right_start, left_keep)
    right_in_overlap = torch.logical_and(right_preds[:, 0] < left_end, right_keep)
    labels_i = set(left_preds[:,-1][left_in_overlap].tolist()) & set(right_preds[:,-1][right_in_overlap].tolist())
    labels_u = set(left_preds[:,-1][left_in_overlap].tolist()) | set(right_preds[:,-1][right_in_overlap].tolist())
    # 不需要过滤，直接返回
    if sum(left_in_overlap) == 0 or sum(right_in_overlap) == 0 or len(labels_i)==0:
        all_preds = torch.cat([left_preds, right_preds], dim=0)
        # 按照预测分数从大到小排序
        return all_preds[torch.argsort(-all_preds[:,4])]
    # 处理不同类别的检测框
    # 跨越交叠区域的检测框
    left_in_overlap_large = torch.logical_and(left_preds[:, 0] < right_start, left_in_overlap)
    right_in_overlap_large = torch.logical_and(right_preds[:, 2] > left_end, right_in_overlap)
    if sum(left_in_overlap_large) == 0 and sum(right_in_overlap_large) == 0:
        deal_preds = torch.cat([left_preds[left_in_overlap], right_preds[right_in_overlap]], dim=0)
    else:
        deal_preds = [] 
        for label in labels_u:
            left_deal_preds = left_preds[torch.logical_and(left_in_overlap, left_preds[:,-1] == label)]
            right_deal_preds = right_preds[torch.logical_and(right_in_overlap, right_preds[:,-1] == label)]
            if label not in labels_i:
                deal_preds.append(torch.cat([left_deal_preds, right_deal_preds], dim=0))
                continue
            # debug_label = False
            # if label ==3:
            #     debug_label= True
                # print('debug')
            left_overBor = left_in_overlap_large[torch.logical_and(left_in_overlap, left_preds[:,-1] == label)]
            right_overBor = right_in_overlap_large[torch.logical_and(right_in_overlap, right_preds[:,-1] == label)]     
            if sum(left_overBor) == 0 and sum(right_overBor) == 0:
                deal_preds.append(torch.cat([left_deal_preds, right_deal_preds], dim=0))
                continue
            # 按照预测分数从大到小排序
            left_deal_preds = left_deal_preds[torch.argsort(-left_deal_preds[:, 4])]
            right_deal_preds = right_deal_preds[torch.argsort(-right_deal_preds[:, 4])]
            # 计算交并比 IOU1 = 相交部分 / 在交叠区域的合并部分
            ious = bbox_overimg_iou(left_deal_preds[:,:4],right_deal_preds[:,:4],right_start,left_end)  #,merge_iou_thr,debug_label
            # 处理被截断的预测框
            def merge_predictions(preds, other_preds, ious, iou_threshold):
                """
                根据给定的IOU阈值合并预测。
                :param preds: 主预测张量 [N, 6], 其中前四列是坐标，第五列是分数，第六列是标签。
                :param other_preds: 对比预测张量 [M, 6]。
                :param ious: 预测之间的IOU矩阵 [N, M]。
                :param iou_threshold: IOU合并阈值。
                :return: 合并后的预测张量。
                """
                # 应用掩码和IOU阈值
                valid_merge_mask = torch.max(ious, dim=1)[0] > iou_threshold
                if not valid_merge_mask.any():
                    return preds  # 如果没有任何有效合并，直接返回原始预测
                # 计算所有可能的合并框坐标
                max_iou_idxs = torch.argmax(ious, dim=1)
                selected_bboxes = other_preds[max_iou_idxs, :4]
                merged_x1 = torch.min(preds[:, 0], selected_bboxes[:, 0])
                merged_x2 = torch.max(preds[:, 2], selected_bboxes[:, 2])
                merged_y1 = torch.min(preds[:, 1], selected_bboxes[:, 1])
                merged_y2 = torch.max(preds[:, 3], selected_bboxes[:, 3])   
                # 构建新的预测张量
                merged_preds = torch.stack([merged_x1, merged_y1, merged_x2, merged_y2, preds[:, 4], preds[:, 5]], dim=1)     
                # 更新预测张量
                non_merged_preds = preds[~valid_merge_mask]
                final_merged_preds = torch.cat([non_merged_preds, merged_preds[valid_merge_mask]], dim=0)  
                return final_merged_preds
            deal_preds.append(merge_predictions(left_deal_preds, right_deal_preds, ious*right_overBor[None,:], merge_iou_thr))
            deal_preds.append(merge_predictions(right_deal_preds, left_deal_preds, ious.T*left_overBor[None,:], merge_iou_thr))
        deal_preds = torch.cat(deal_preds,dim=0)
    # 执行NMS
    nms_cfg = dict(type='nms', iou_threshold=0.7)
    det_bboxes, keep_idxs = batched_nms(deal_preds[:,:4].float(), deal_preds[:,4].float(), deal_preds[:,-1].int(), nms_cfg)
    deal_preds = deal_preds[keep_idxs]
    all_preds = torch.cat([left_preds[~left_in_overlap],right_preds[~right_in_overlap],deal_preds], axis=0)
    return all_preds

# 多对多, 定义一个函数来计算两个边框的交并比
def calculate_iou(boxes1, boxes2):
    # 扩展boxes1形状为[11,1,3], 扩展boxes2形状为[1,11,3]
    boxes1 = boxes1.unsqueeze(1)
    boxes2 = boxes2.unsqueeze(0)
    # 计算交叉区域的坐标
    x1 = torch.maximum(boxes1[:, :, 0], boxes2[:, :, 0])
    y1 = torch.maximum(boxes1[:, :, 1], boxes2[:, :, 1])
    x2 = torch.minimum(boxes1[:, :, 2], boxes2[:, :, 2])
    y2 = torch.minimum(boxes1[:, :, 3], boxes2[:, :, 3])
    # 计算交叉区域的面积
    intersection_area = torch.maximum(x2 - x1, torch.zeros_like(x2)) * torch.maximum(y2 - y1, torch.zeros_like(y2))
    # 计算每对边框的面积
    area_boxes1 = (boxes1[:, :, 2] - boxes1[:, :, 0]) * (boxes1[:, :, 3] - boxes1[:, :, 1])
    area_boxes2 = (boxes2[:, :, 2] - boxes2[:, :, 0]) * (boxes2[:, :, 3] - boxes2[:, :, 1])
    # 计算交并比(返回形状为[11,11]的IoU矩阵)
    iou = intersection_area / (area_boxes1 + area_boxes2 - intersection_area)
    # 将对角线元素置为0
    iou.fill_diagonal_(0)
    # 将对角线and上半对角角元素置为0元素置为0
    # iou.tril_(diagonal=-1)
    return iou
def merge_bboxs(input_lines, line_min=lineMin):
    line_min=lineMin #------------for: test.py change lineMin
    # line_bboxes = [num_line,num_point*2]
    # 寻找最小的x（x1），最小的y（y1），最大的x（x2），最大的y（y2）
    x_min = torch.min(input_lines[:, 0::2])
    x_max = torch.max(input_lines[:, 0::2])
    y_min = torch.min(input_lines[:, 1::2])
    y_max = torch.max(input_lines[:, 1::2])
    if (y_max - y_min) < line_min:
        y_max, y_min = (y_max + y_min) / 2.0 + line_min / 2.0, (y_max + y_min) / 2.0 - line_min / 2.0
    if (x_max - x_min) < line_min:
        x_max, x_min = (x_max + x_min) / 2.0 + line_min / 2.0, (x_max + x_min) / 2.0 - line_min / 2.0
    return torch.tensor([x_min, y_min, x_max, y_max])
def filterLineBoxByIoU(bboxes, scores, lines, iou_threshold=0.5,score_threshold=0.3):
    # bboxes [num,4] xyxy; scores[num]
    # 获取置信度大于阈值的索引
    valid_indices = scores >= score_threshold
    # 如果没有边界框满足要求,直接返回原始向量
    if sum(valid_indices)<2:
        return bboxes, scores
    # step1: 获取满足要求的边界框和置信度
    valid_bboxes = bboxes[valid_indices]
    valid_scores = scores[valid_indices]
    valid_lines = lines[valid_indices]
    # 计算边框1对1的交并比
    iou_torch = calculate_iou(valid_bboxes, valid_bboxes)
    merge_indices = torch.nonzero(iou_torch > iou_threshold, as_tuple=True)
    # 对于没有超过 iou_threshold 的边界框,直接添加到 merged_bboxes 中
    merged_bboxes = valid_bboxes[~torch.any(iou_torch > iou_threshold, dim=1)]
    merged_scores = valid_scores[~torch.any(iou_torch > iou_threshold, dim=1)]
    if merge_indices[0].numel() > 0:
        # 使用networkx找出所有应该合并的边界框组
        edges = list(zip(merge_indices[0].tolist(), merge_indices[1].tolist()))
        G = nx.Graph()
        G.add_edges_from(edges)
        # 找出所有连接的组件
        components = list(nx.connected_components(G))
        # 将IOU大于line_merge_iou_threshold的Line box合并给第一个，其他都置为None，然后移除
        for i in components:
            input_lines = valid_lines[list(i)] #--------line
            merged_bbox = merge_bboxs(input_lines)
            merged_bbox = merged_bbox.to(device=merged_bboxes.device)
            merged_bboxes = torch.cat([merged_bboxes, merged_bbox.unsqueeze(0)])
            merged_scores = torch.cat([merged_scores, torch.max(valid_scores[list(i)]).unsqueeze(0)])
    # step2: 将未处理的边界框与合并后的边界框连接在一起
    all_bboxes = torch.cat([merged_bboxes,bboxes[~valid_indices]],dim=0)
    all_scores = torch.cat([merged_scores,scores[~valid_indices]],dim=0)    
    return all_bboxes, all_scores

def line2box(line_bboxes,box_min=lineMin):
    box_min=lineMin #------------for: test.py change lineMin
    min_x = np.min(line_bboxes[:,0::2], axis=-1)
    max_x = np.max(line_bboxes[:,0::2], axis=-1)
    min_y = np.min(line_bboxes[:,1::2], axis=-1)
    max_y = np.max(line_bboxes[:,1::2], axis=-1)
    # Calculate center
    center_x = (max_x + min_x) / 2
    center_y = (max_y + min_y) / 2
    # Adjust width to be at least 30
    min_x_update = np.where(max_x - min_x < box_min, center_x - box_min / 2, min_x)
    max_x_update = np.where(max_x - min_x < box_min, center_x + box_min / 2, max_x)
    # Adjust height to be at least 30
    min_y_update = np.where(max_y - min_y < box_min, center_y - box_min / 2, min_y)
    max_y_update = np.where(max_y - min_y < box_min, center_y + box_min / 2, max_y)  
    # Concatenate results [minX, minY, maxX, maxY]
    results = np.stack((min_x_update, min_y_update, max_x_update, max_y_update), axis=-1)
    return results   
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

@METRICS.register_module()
class LineCocoMetric_overLapCls(CocoMetric):
    
    default_prefix: Optional[str] = 'coco'

    def __init__(
            self,
            *args,
            ann_file: Optional[str] = None,
            line_pre=None,
            box_num_cls = 3,
            metric: Union[str, List[str]] = 'bbox',
            line_head_merge_cfg=dict(line_nms_thre=None,line_nms_socre_thre=0.3),
            merged_preds_cfg=dict(merged_preds=False,fullImg_ann_file=None), #将左右两张图的结果合并在一张张图上，再计算metric
            clc_voc=False, # voc metric
            recordMetCha=False,
            recordMetCha_lineName='line',
            recordMetCha_OtherName='TP',
            clc_line_metric = True, #计算基于点集的指标, 已经弃用
            clc_twoHeadScoreDiff_onVal=False, #line head和box head的分数也许不在同一尺度上，在验证集上，计算两者需要补偿的差值
            cls_infos = dict(all_classes=('black_core', 'finger', 'thick_line', 'short_circuit', 'crack', 'horizontal_dislocation'), 
                             box_classes=('black_core', 'thick_line', 'short_circuit', 'crack'),
                             line_classes=('finger','horizontal_dislocation',)),
            **kwargs) -> None:
        '''
        for CocoMetric: merged_preds_cfg,clc_voc 
        '''
        self.clc_line_metric = clc_line_metric
        self.clc_twoHeadScoreDiff_onVal = clc_twoHeadScoreDiff_onVal
        self.clc_voc = clc_voc 
        self.recordMetCha = recordMetCha
        self.recordMetCha_lineName = recordMetCha_lineName
        self.recordMetCha_OtherName = recordMetCha_OtherName
        if self.recordMetCha:
            self.recordMetCha_epoch = 1
            self.bbox_map = []
            self.bbox_map_50 = []
            self.tp_changes = []
            self.line_changes = []
            self.previous_tp_value = None
            self.previous_line_value = None
            self.category_indices = {}
            self.max_values = {
                'bbox_mAP': (-1, float('-inf')),
                'bbox_mAP_50': (-1, float('-inf')),
                self.recordMetCha_OtherName or 'TP': (-1, float('-inf')),
                self.recordMetCha_lineName: (-1, float('-inf'))
            }
        #-------------for line_nms_thre-------------
        self.line_nms_iou_thre = line_head_merge_cfg.pop('line_nms_thre', None) # or None, IOU = 0.5
        self.line_nms_socre_thre = line_head_merge_cfg.pop('line_nms_socre_thre', 0.3)
        ann_file_line = ann_file
        #-------------close by zcf 250321
        # if (self.line_nms_iou_thre is not None) and (self.line_nms_iou_thre>0):
        #     if '_lineIoU0d5.json' not in ann_file:
        #         ann_file = ann_file.split('.json')[0]+'_lineIoU0d5.json'
        #     # print(ann_file)
        # if (self.line_nms_iou_thre is None) and ('_lineIoU0d5.json' in ann_file):
        #     print('**************for line head, There may be an error here')
        #     # 因为对非line head没有影响
        #     self.line_nms_iou_thre = 0.5
        #     self.line_nms_socre_thre = 0.3
        #-------------close by zcf 250321
        #-------------for line_nms_thre-------------
        #-------------for merged_preds-------------
        self.merged_preds = merged_preds_cfg.pop('merged_preds', None)
        if self.merged_preds:
            self.merged_count = 0
            dir_name, file_name = os.path.split(ann_file)
            new_file_name = file_name.replace('half', 'full')
            fullImg_ann_file = os.path.join(dir_name, new_file_name)
            self.fullImg_ann_file = merged_preds_cfg.pop('fullImg_ann_file', None)
            if self.fullImg_ann_file is None:
                self.fullImg_ann_file = fullImg_ann_file  
            print('evaluate on ',new_file_name,'..........')
        #-------------for merged_preds-------------
        super().__init__(*args, ann_file=ann_file, metric=metric, **kwargs)
        self.line_pre,self.inter_reg  = line_pre, None
        if line_pre:
            self.points_for_lossMetric = line_pre['points_for_lossMetric'] #计算损失的时候，统一预测和gt的点数，两者都是points_for_lossMetric=11个
            self.line_points_inter_method = line_pre['line_points_inter_method'] 
            self.inter_reg = line_pre['inter_reg'] 
        self.box_num_cls = box_num_cls
        #-------------for line_nms_thre-------------
        if ann_file_line is not None:
            with get_local_path(ann_file_line, backend_args=self.backend_args) as local_path:
                self._coco_api_line = COCO(local_path)
        #-------------for line_nms_thre------------- 
        #--------------------------cls_infos---------------1
        self.cls_infos = cls_infos
        if cls_infos is not None:
            global_index = {cls: idx for idx, cls in enumerate(cls_infos['all_classes'])}
            # Build mappings for box_classes and line_classes
            self.box_index_mapping = {idx: global_index[cls] for idx, cls in enumerate(cls_infos['box_classes'])}
            self.line_index_mapping = {idx: global_index[cls] for idx, cls in enumerate(cls_infos['line_classes'])}
        #-------------------------- 
    
    def process(self, data_batch: dict, data_samples: Sequence[dict]) -> None:
        """Process one batch of data samples and predictions. The processed
        results should be stored in ``self.results``, which will be used to
        compute the metrics when all batches have been processed.

        Args:
            data_batch (dict): A batch of data from the dataloader.
            data_samples (Sequence[dict]): A batch of data samples that
                contain annotations and predictions.
        """
        for data_sample in data_samples:
            result = dict()
            pred = data_sample['pred_instances']
            result['img_id'] = data_sample['img_id']
            result['bboxes'] = pred['bboxes'].cpu().numpy()
            result['scores'] = pred['scores'].cpu().numpy()
	    #--------------------------cls_infos---------------2
            if self.cls_infos is not None:
                pred['labels'] = torch.tensor([self.box_index_mapping[label.item()] for label in pred['labels']])
	    #--------------------------cls_infos---------------2
            result['labels'] = pred['labels'].cpu().numpy()
            # encode mask to RLE
            if 'masks' in pred:
                result['masks'] = encode_mask_results(
                    pred['masks'].detach().cpu().numpy()) if isinstance(
                        pred['masks'], torch.Tensor) else pred['masks']
            # some detectors use different scores for bbox and mask
            if 'mask_scores' in pred:
                result['mask_scores'] = pred['mask_scores'].cpu().numpy()
            #---------------------zhou line_pred_instances
            if 'line_pred_instances' in data_sample:
                pred = data_sample['line_pred_instances']
                result['line_points'] = pred['line_points'].cpu().numpy()
                result['line_scores'] = pred['line_scores'].cpu().numpy()
                result['line_labels'] = pred['line_labels'].cpu().numpy()
                #-----line2box
                # bboxes2 = line2box(pred['line_points'].cpu().numpy()) #[num,11*2]-->(x1,y1,x2,y2), image scale before resize
                #TODO: 不行，合并的方式考虑得太简单
                line2 = pred['line_points']
                bboxes2 = line2box_torch(pred['line_points'],img_height=data_sample['ori_shape'][0],img_width=data_sample['ori_shape'][1])
                scores2 = pred['line_scores']
                labels2 = pred['line_labels']
                scores2, idxs = scores2.sort(descending=True)
                line2 = line2[idxs]
                bboxes2 = bboxes2[idxs]
                labels2 = labels2[idxs]
                #-------------for line_nms_thre-------------
                if self.line_nms_iou_thre is not None:
                    all_bboxes,all_scores,all_labels = [], [], []
                    # print('line_nms_thre')
                    for label_i in labels2.unique():
                        indices_i = labels2 == label_i
                        bboxes2_i, scores2_i = filterLineBoxByIoU(bboxes2[indices_i], scores2[indices_i], line2[indices_i], iou_threshold=self.line_nms_iou_thre,score_threshold=self.line_nms_socre_thre)
                        labels2_i = torch.full_like(scores2_i, fill_value=label_i, dtype=labels2.dtype)
                        all_bboxes.append(bboxes2_i)
                        all_scores.append(scores2_i)
                        all_labels.append(labels2_i)
                    bboxes2,scores2,labels2 = torch.cat(all_bboxes, dim=0),torch.cat(all_scores, dim=0),torch.cat(all_labels, dim=0)
                # if self.line_nms_iou_thre is not None:
                #     dets, keep = nms(bboxes2, pred['line_scores'], iou_threshold=self.line_nms_iou_thre) #torch[num,4] xyxy,torch[num],
                #     bboxes2 = bboxes2[keep]
                #     scores2 = dets[:, -1]
                #     labels2 = labels2[keep]
                #-------------for line_nms_thre-------------
                result['bboxes'] = np.concatenate((result['bboxes'], bboxes2.cpu().numpy()), axis=0)
                result['scores'] = np.concatenate((result['scores'], copy.deepcopy(scores2.cpu().numpy())), axis=0)  
		#--------------------------cls_infos---------------3
                if self.cls_infos is not None:
                    labels2 = torch.tensor([self.line_index_mapping[label.item()] for label in labels2])
                else:
                    labels2 +=self.box_num_cls 
                result['labels'] = np.concatenate((result['labels'], copy.deepcopy(labels2.cpu().numpy())), axis=0) #TODO!
                #---------------------zhou line_pred_instances
		#--------------------------cls_infos---------------3
            # parse gt
            gt = dict()
            gt['width'] = data_sample['ori_shape'][1]
            gt['height'] = data_sample['ori_shape'][0]
            gt['img_id'] = data_sample['img_id']
            if self._coco_api is None:
                # TODO: Need to refactor to support LoadAnnotations
                assert 'instances' in data_sample, \
                    'ground truth is required for evaluation when ' \
                    '`ann_file` is not provided'
                gt['anns'] = data_sample['instances']
            # gt['line_anns'] = data_sample['gt_line_instances'] #TODO!------------zhou, 但是图像的scale不对，没使用scale_factor缩放到原始的图像尺寸上
            # add converted result to the results list
            self.results.append((gt, result))
            
    def compute_metrics_line(self, results: list) -> Dict[str, float]:
        
        # split gt and prediction list
        gts, preds = zip(*results)
        preds = self.line_format_results(preds)
        gts = self.line_format_gt(gts)

        """
            calculate distances between gts and preds    
        """
    
        distance_matrixs = {
            'chamfer': {}, #-zhou Chamfer distance ignores the direction information.
            # 'frechet': {}, # Frechet distance takes into account both the distance and direction information between the predicted and ground truth centerlines 
            # 'iou': {},
        }
        verbose = True
        for token in tqdm(gts.keys(), desc='calculating distances:', ncols=80, disable=not verbose):
            distance_matrixs['chamfer'][token] = pairwise(
                [gt['points'] for gt in gts[token]['lane_centerline']],
                [pred['points'] for pred in preds[token]['lane_centerline']],
                chamfer_distance,
                relax=False,
            ) # < THRESHOLDS_FRECHET[-1]
            # distance_matrixs['frechet'][token] = pairwise(
            #     [gt['points'] for gt in gts[token]['lane_centerline']],
            #     [pred['points'] for pred in preds[token]['lane_centerline']],
            #     frechet_distance,
            #     mask=mask,
            #     relax=True,
            # )
        """
            evaluate
        """
        line_metrics = _mAP_over_threshold(
            gts=gts, 
            preds=preds, 
            distance_matrixs=distance_matrixs['chamfer'], 
            distance_thresholds=THRESHOLDS_FRECHET,
            object_type='lane_centerline',
            filter=lambda _: True,
            inject=False, # True==save tp for eval on graph
        ).mean()
        return line_metrics
    # 
    def get_voc_input(self, preds,coco_api,preds_cls_order=[0,3,1,2]):
        preds_voc = []
        for pred_i in preds:
            dets = []
            pred_labels = pred_i['labels']
            for label in preds_cls_order:
                index = np.where(pred_labels == label)[0]
                pred_bbox_scores = np.hstack([pred_i['bboxes'][index], pred_i['scores'][index].reshape((-1, 1))]) #xyxy,score
                dets.append(pred_bbox_scores)
            preds_voc.append(dets)
        
        ann_voc = []    
        ImgIds = coco_api.getImgIds()
        for ImgId_i in ImgIds:
            Anns_i = coco_api.loadAnns(coco_api.getAnnIds(imgIds=ImgId_i))
            # 提取'category_id'和'bbox'并堆叠
            labels = np.array([item['category_id']-1 for item in Anns_i]) #start from 0
            bboxes = np.array([[item['bbox'][0],item['bbox'][1],item['bbox'][0]+item['bbox'][2],item['bbox'][1]+item['bbox'][3]] for item in Anns_i]).reshape((-1, 4))
            bboxes_ignore = np.array([[item['bbox'][0],item['bbox'][1],item['bbox'][0]+item['bbox'][2],item['bbox'][1]+item['bbox'][3]] for item in Anns_i if item['ignore']==1]).reshape((-1, 4))
            labels_ignore = np.array([item['category_id']-1 for item in Anns_i if item['ignore']==1])
            bboxes_ignore,labels_ignore = None,None
            # print('bboxes_ignore',bboxes_ignore)
            ann = dict(
                labels=labels,
                bboxes=bboxes,
                bboxes_ignore=bboxes_ignore,
                labels_ignore=labels_ignore)
            ann_voc.append(ann)
        return preds_voc,ann_voc
    
    # def compute_metrics(self, results: list) -> Dict[str, float]:
    #     eval_results = self.compute_metrics_singeRun(results)
    #     eval_results2 = self.compute_metrics_singeRun(results, full_mode=True)
    #     return eval_results
        
    def compute_metrics(self, results: list) -> Dict[str, float]:
        """Compute the metrics from processed results.

        Args:
            results (list): The processed results of each batch.

        Returns:
            Dict[str, float]: The computed metrics. The keys are the names of
            the metrics, and the values are corresponding results.
        """
        logger: MMLogger = MMLogger.get_current_instance()
        eval_results = OrderedDict() #---------zhou
        if self.clc_line_metric and 'line_points' in results[0][-1]:
            # 但是目前计算的仍旧半图上的指标，不是整图上的, 已经弃用
            line_metric = self.compute_metrics_line(results) #---------zhou
            # line_metric = 0.0
            logger.info(f'line_mAP_metric: {line_metric:.3f}') #---------zhou
            eval_results['line_mAP_metric'] = round(line_metric, 3)  #---------zhou
        # split gt and prediction list
        gts, preds = zip(*results)
        max_length_line,max_length_count = [], 0 
        for ii in preds:
            if (ii['scores'] > 0.1).any():
                kkkkk_here = torch.from_numpy(ii['bboxes'][ii['scores'] > 0.1])
                max_length_ = torch.maximum(
                                    kkkkk_here[:,3] - kkkkk_here[:,1], 
                                    kkkkk_here[:,2] - kkkkk_here[:,0]
                                )
                if sum(max_length_>4000)>0:
                    max_length_count+=1
                max_length_line.extend(copy.deepcopy(max_length_.tolist()))
        # logger.info(f'max_length_line, count>4000 numimgs ={max_length_count}, mean={np.array(max_length_line).mean()}, max={max(max_length_line)}, min={min(max_length_line)}')
        #-------------for merged_preds-------------
        if self.merged_preds:
            if self.merged_count == 0:
                self._coco_api_back = copy.deepcopy(self._coco_api) #如果不保存，下一次的file_name_indices就会报错
                self.merged_count =1
            #键是file_name,值是对应结果在results列表中的索引。: '_l.jpg':index,'_r.jpg':index,.....
            file_name_indices = {self._coco_api_back.loadImgs(ids=[result['img_id']])[0]['file_name']: idx for idx, result in enumerate(preds)}
            # print(file_name_indices)
            self._coco_api = COCO(self.fullImg_ann_file)
            # print(self.fullImg_ann_file)
            full_preds = []
            full_Imgids = self._coco_api.getImgIds()
            for i in full_Imgids:
                full_ImgInfos = self._coco_api.loadImgs(ids=i)
                left_preds_index = file_name_indices[full_ImgInfos[0]['file_name'].replace('.jpg', '_l.jpg')]
                right_preds_index = file_name_indices[full_ImgInfos[0]['file_name'].replace('.jpg', '_r.jpg')]
                left_preds = preds[left_preds_index]
                right_preds = preds[right_preds_index]
                right_preds['bboxes'][:,0] += right_start
                right_preds['bboxes'][:,2] += right_start
                # print(full_ImgInfos[0]['file_name'])
                all_pr = merge_detections(np.concatenate([left_preds['bboxes'], left_preds['scores'][:,None], left_preds['labels'][:,None]],axis=1),
                                          np.concatenate([right_preds['bboxes'], right_preds['scores'][:,None], right_preds['labels'][:,None]],axis=1))
                result_per = dict()
                result_per['img_id'] = i
                result_per['bboxes'] = all_pr[:,:4].numpy().astype(left_preds['bboxes'].dtype)
                result_per['scores'] = all_pr[:,4].numpy().astype(left_preds['scores'].dtype)
                result_per['labels'] = all_pr[:,5].numpy().astype(left_preds['labels'].dtype)
                full_preds.append(result_per)
            preds = tuple(full_preds)
         #-------------for merged_preds-------------
        tmp_dir = None
        if self.outfile_prefix is None:
            tmp_dir = tempfile.TemporaryDirectory()
            outfile_prefix = osp.join(tmp_dir.name, 'results')
        else:
            outfile_prefix = self.outfile_prefix

        if self._coco_api is None:
            # use converted gt json file to initialize coco api
            logger.info('Converting ground truth to coco format...')
            coco_json_path = self.gt_to_coco_json(
                gt_dicts=gts, outfile_prefix=outfile_prefix)
            self._coco_api = COCO(coco_json_path)
        # handle lazy init
        if self.cat_ids is None:
            self.cat_ids = self._coco_api.get_cat_ids(cat_names=self.dataset_meta['classes'])
        if 'line_points' in results[0][-1] or 'line_classes' in self.dataset_meta:
            if getattr(self, 'cat_ids_line', None) is None:#--zhou
                self.cat_ids_line = self._coco_api.get_cat_ids(cat_names=self.dataset_meta['line_classes']) #--zhou
        else:
            self.cat_ids_line = []
	#--------------------------cls_infos---------------4
        if self.cls_infos is not None:
            cat_ids = self._coco_api.get_cat_ids(cat_names=self.cls_infos['all_classes'])
        else: #--------------------------cls_infos---------------4
            cat_ids = self.cat_ids+self.cat_ids_line #--zhou
        if self.img_ids is None:
            self.img_ids = self._coco_api.get_img_ids()
        # convert predictions to coco format and dump to json file
        result_files = self.results2json(preds, outfile_prefix)
        # eval_results = OrderedDict()
        if self.clc_voc:
            preds_voc,ann_voc = self.get_voc_input(preds,self._coco_api,preds_cls_order=[0,3,1,2]) #TP,line,'mura','sq'
            dataset_names = np.array(self.dataset_meta['classes']+self.dataset_meta['line_classes'])[[0,3,1,2]]
            mean_ap, _ = eval_map(preds_voc,ann_voc,scale_ranges=None,
                                    iou_thr=0.5,
                                    dataset=dataset_names.tolist(),
                                    logger=logger,
                                    eval_mode='area',
                                    use_legacy_coordinate=True)
        if self.format_only:
            logger.info('results are saved in '
                        f'{osp.dirname(outfile_prefix)}')
            return eval_results

        for metric in self.metrics:
            logger.info(f'Evaluating {metric}...')

            # TODO: May refactor fast_eval_recall to an independent metric?
            # fast eval recall
            if metric == 'proposal_fast':
                ar = self.fast_eval_recall(
                    preds, self.proposal_nums, self.iou_thrs, logger=logger)
                log_msg = []
                for i, num in enumerate(self.proposal_nums):
                    eval_results[f'AR@{num}'] = ar[i]
                    log_msg.append(f'\nAR@{num}\t{ar[i]:.4f}')
                log_msg = ''.join(log_msg)
                logger.info(log_msg)
                continue

            # evaluate proposal, bbox and segm
            iou_type = 'bbox' if metric == 'proposal' else metric
            if metric not in result_files:
                raise KeyError(f'{metric} is not in results')
            try:
                predictions = load(result_files[metric])
                if iou_type == 'segm':
                    # Refer to https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocotools/coco.py#L331  # noqa
                    # When evaluating mask AP, if the results contain bbox,
                    # cocoapi will use the box area instead of the mask area
                    # for calculating the instance area. Though the overall AP
                    # is not affected, this leads to different
                    # small/medium/large mask AP results.
                    for x in predictions:
                        x.pop('bbox') 
                coco_dt = self._coco_api.loadRes(predictions)

            except IndexError:
                logger.error(
                    'The testing results of the whole dataset is empty.')
                if 'line_mAP_metric' in eval_results: #当什么都没有检测出来的时候，如果不这样，程序会报错终止KeyError: 'coco/bbox_mAP_50'
                    eval_results.pop('line_mAP_metric')
                break

            coco_eval = COCOeval(self._coco_api, coco_dt, iou_type)

            coco_eval.params.catIds = cat_ids #--zhou
            coco_eval.params.imgIds = self.img_ids
            coco_eval.params.maxDets = list(self.proposal_nums)
            coco_eval.params.iouThrs = self.iou_thrs

            # mapping of cocoEval.stats
            coco_metric_names = {
                'mAP': 0,
                'mAP_50': 1,
                'mAP_75': 2,
                'mAP_s': 3,
                'mAP_m': 4,
                'mAP_l': 5,
                'AR@100': 6,
                'AR@300': 7,
                'AR@1000': 8,
                'AR_s@1000': 9,
                'AR_m@1000': 10,
                'AR_l@1000': 11
            }
            metric_items = self.metric_items
            if metric_items is not None:
                for metric_item in metric_items:
                    if metric_item not in coco_metric_names:
                        raise KeyError(
                            f'metric item "{metric_item}" is not supported')

            if metric == 'proposal':
                coco_eval.params.useCats = 0
                coco_eval.evaluate()
                coco_eval.accumulate()
                coco_eval.summarize()
                if metric_items is None:
                    metric_items = [
                        'AR@100', 'AR@300', 'AR@1000', 'AR_s@1000',
                        'AR_m@1000', 'AR_l@1000'
                    ]

                for item in metric_items:
                    val = float(
                        f'{coco_eval.stats[coco_metric_names[item]]:.3f}')
                    eval_results[item] = val
            else:
                coco_eval.evaluate()
                coco_eval.accumulate()
                coco_eval.summarize()
                #----------------------add
                if self.clc_twoHeadScoreDiff_onVal:
                    score_cls = coco_eval.eval['scores'][0,:,:,0,-1] #[num_recalls,num_cls] iou=0.5, area=all, coco_eval.params.maxDets=-1=1000=max
                    logger.info(f"maximum=={self.dataset_meta['classes']}The difference of maximum predicted scores: {score_cls.max(0)}") 
                    logger.info("maximum==Compensation score required for the line head: mean3-")
                    logger.info(f"average==={self.dataset_meta['classes']}The difference of average predicted scores: {score_cls.mean(0)}") 
                    logger.info("average===Compensation score required for the line head: mean3-") 
                #----------------------add
                if self.classwise:  # Compute per-category AP
                    # Compute per-category AP
                    # from https://github.com/facebookresearch/detectron2/
                    precisions = coco_eval.eval['precision']
                    # precision: (iou, recall, cls, area range, max dets)
                    assert len(cat_ids) == precisions.shape[2] #--zhou
                    results_per_category = []
                    cat_ids = coco_eval.params.catIds #--zhou, because p.catIds = list(np.unique(p.catIds))
                    for idx, cat_id in enumerate(cat_ids): #--zhou
                        cat_id = int(cat_id)
                        t = []
                        # area range index 0: all area ranges
                        # max dets index -1: typically 100 per image
                        nm = self._coco_api.loadCats(cat_id)[0]
                        precision = precisions[:, :, idx, 0, -1]
                        precision = precision[precision > -1]
                        if precision.size:
                            ap = np.mean(precision)
                        else:
                            ap = float('nan')
                        t.append(f'{nm["name"]}')
                        t.append(f'{round(ap, 3)}')
                        eval_results[f'{nm["name"]}_precision'] = round(ap, 3)

                        # indexes of IoU  @50 and @75
                        for iou in [0, 5]:
                            precision = precisions[iou, :, idx, 0, -1]
                            precision = precision[precision > -1]
                            if precision.size:
                                ap = np.mean(precision)
                            else:
                                ap = float('nan')
                            t.append(f'{round(ap, 3)}')

                        # indexes of area of small, median and large
                        for area in [1, 2, 3]:
                            precision = precisions[:, :, idx, area, -1]
                            precision = precision[precision > -1]
                            if precision.size:
                                ap = np.mean(precision)
                            else:
                                ap = float('nan')
                            t.append(f'{round(ap, 3)}')
                        # change by zhou 240716, add AR------------------
                        recalls = coco_eval.eval['recall'] #[TxKxAxM], (iou, cls, area range, max dets)
                        recalls_here = recalls[:, idx, 0, -1]
                        recalls_here = recalls_here[recalls_here > -1]
                        if recalls_here.size:
                            recalls_here = np.mean(recalls_here)
                        else:
                            recalls_here = float('nan')
                            
                        recalls_here_50 = recalls[0, idx, 0, -1] #iou=50

                        t.append(f'{round(recalls_here, 3)}')
                        t.append(f'{round(recalls_here_50, 3)}')
                        # change by zhou 240716, add AR------------------
                        results_per_category.append(tuple(t))

                    num_columns = len(results_per_category[0])
                    results_flatten = list(
                        itertools.chain(*results_per_category))
                    headers = [
                        'category', 'mAP', 'mAP_50', 'mAP_75', 'mAP_s',
                        'mAP_m', 'mAP_l', 'AR', 'AR_50' 
                    ]
                    results_2d = itertools.zip_longest(*[
                        results_flatten[i::num_columns]
                        for i in range(num_columns)
                    ])
                    table_data = [headers]
                    table_data += [result for result in results_2d]
                    table = AsciiTable(table_data)
                    logger.info('\n' + table.table)

                if metric_items is None:
                    metric_items = [
                        'mAP', 'mAP_50', 'mAP_75', 'mAP_s', 'mAP_m', 'mAP_l',  
                        'AR@1000', 'AR_s@1000', 'AR_m@1000', 'AR_l@1000'#change by zhou 240716, add AR------------------
                    ]

                for metric_item in metric_items:
                    key = f'{metric}_{metric_item}'
                    val = coco_eval.stats[coco_metric_names[metric_item]]
                    eval_results[key] = float(f'{round(val, 3)}')

                ap = coco_eval.stats[:6]
                logger.info(f'{metric}_mAP_copypaste: {ap[0]:.3f} '
                            f'{ap[1]:.3f} {ap[2]:.3f} {ap[3]:.3f} '
                            f'{ap[4]:.3f} {ap[5]:.3f}')
            
        if tmp_dir is not None:
            tmp_dir.cleanup()
        if self.recordMetCha:
            if 'results_per_category' in locals(): #
                self.log_metrics(self.recordMetCha_epoch,eval_results,results_per_category) #当什么都没有检测出来的时候，不存在results_per_category变量，调用log_metrics下面的会报错
            self.recordMetCha_epoch +=1
        # change by zhou 240716------------------
        recordPaper = True
        if recordPaper and 'coco_eval' in locals(): #240723, 因为会有这种情况, The testing results of the whole dataset is empty.
            precisions = coco_eval.eval['precision'] #[TxRxKxAxM], (iou, recall, cls, area range, max dets)
            def compute_average_precision(precision_array):
                valid_precisions = precision_array[precision_array > -1]
                if valid_precisions.size:
                    return np.mean(valid_precisions)
                else:
                    return float('nan')
            # 计算每个 proposal_num 的平均 precision
            average_precisions = []
            for i, proposal_num in enumerate(self.proposal_nums):
                precision_i = precisions[:, :, :, 0, i]
                average_precisions.append(compute_average_precision(precision_i))
            # 计算 line_id 的 AP 和 AP50
            line_id = self._coco_api.get_cat_ids(self.recordMetCha_lineName)[0]-1 #1 cat_ids['line]
            if 'results_per_category' in locals(): #
                line_AP = float(results_per_category[line_id][1])
            else:
                line_AP = precisions[:, :, line_id, 0, -1]
                line_AP = compute_average_precision(line_AP)
            
            if 'results_per_category' in locals(): #
                line_AP_50 = float(results_per_category[line_id][2])
            else:
                line_AP_50 = precisions[0, :, line_id, 0, -1]
                line_AP_50 = compute_average_precision(line_AP_50)
            
            # 计算不同面积范围的 box mAP（排除 line_id 的类别）
            mask = np.ones(precisions.shape[2], dtype=bool)
            mask[line_id] = False
            box_mAP_s = precisions[:, :, mask, 1, -1]
            box_mAP_s = compute_average_precision(box_mAP_s)
            
            box_mAP_m = precisions[:, :, mask, 2, -1]
            box_mAP_m = compute_average_precision(box_mAP_m)
            
            box_mAP_l = precisions[:, :, mask, 3, -1]
            box_mAP_l = compute_average_precision(box_mAP_l)
            
            recalls = coco_eval.eval['recall'] #[TxKxAxM], (iou, cls, area range, max dets)
            mAR_50 = recalls[0, :, 0, -1]
            mAR_50 = compute_average_precision(mAR_50)
            
            logger.info(f'==>mAP@{self.proposal_nums[0]}, mAP@{self.proposal_nums[1]}, mAP@{self.proposal_nums[2]}: \n' 
                       f'{average_precisions[0]:.3f}, {average_precisions[1]:.3f}, {average_precisions[2]:.3f}') #四舍五入
            logger.info(f'==>mAP@100, mAP_50, mAP_75, mAP_s, mAP_m, mAP_l, line AP, line AP_50, mAR, mAR_50: \n' 
                       f'{coco_eval.stats[coco_metric_names["mAP"]]*100:.1f} & {coco_eval.stats[coco_metric_names["mAP_50"]]*100:.1f} & '
                       f'{coco_eval.stats[coco_metric_names["mAP_75"]]*100:.1f} & {coco_eval.stats[coco_metric_names["mAP_s"]]*100:.1f} & '
                       f'{coco_eval.stats[coco_metric_names["mAP_m"]]*100:.1f} & {coco_eval.stats[coco_metric_names["mAP_l"]]*100:.1f} & '
                       f'{line_AP*100:.1f} & {line_AP_50*100:.1f} & '
                       f'{coco_eval.stats[coco_metric_names["AR@1000"]]*100:.1f} & {mAR_50*100:.1f}') 
            logger.info(f'==>mAP@100, mAP_50, line AP, line AP_50, box_mAP_s, box_mAP_m, box_mAP_l: \n' 
                       f'{coco_eval.stats[coco_metric_names["mAP"]]*100:.1f} & {coco_eval.stats[coco_metric_names["mAP_50"]]*100:.1f} & '
                       f'{line_AP*100:.1f} & {line_AP_50*100:.1f} & '
                       f'{box_mAP_s*100:.1f} & {box_mAP_m*100:.1f} &  {box_mAP_l*100:.1f}') 
            
        return eval_results
    
    def log_metrics(self, epoch, eval_results, results_per_category):
        # Initialize category indices on the first run
        if not self.category_indices:
            self.category_indices = {category[0]: idx for idx, category in enumerate(results_per_category)}
        changes_detected = True
        ###log max
        self.update_max('bbox_mAP', epoch, eval_results['bbox_mAP'])
        self.update_max('bbox_mAP_50', epoch, eval_results['bbox_mAP_50'])
        if self.recordMetCha_OtherName is not None:
            self.update_max(self.recordMetCha_OtherName, epoch, float(results_per_category[self.category_indices[self.recordMetCha_OtherName]][2]))
        self.update_max(self.recordMetCha_lineName, epoch, float(results_per_category[self.category_indices[self.recordMetCha_lineName]][2]))
        # Log bbox_mAP and bbox_mAP_50 every 5 epochs
        if epoch % 5 == 0:
            self.bbox_map.append((epoch, eval_results['bbox_mAP']))
            self.bbox_map_50.append((epoch, eval_results['bbox_mAP_50']))
            # changes_detected = True
        # Check and log TP changes
        if self.recordMetCha_OtherName is not None:
            current_tp_value = float(results_per_category[self.category_indices[self.recordMetCha_OtherName]][2])
            if self.previous_tp_value is not None and self.extract_first_decimal(current_tp_value) != self.extract_first_decimal(self.previous_tp_value):
                self.tp_changes.append((epoch, current_tp_value))
                # changes_detected = True
            self.previous_tp_value = current_tp_value
        # Check and log Line changes
        current_line_value = float(results_per_category[self.category_indices[self.recordMetCha_lineName]][2])
        if self.previous_line_value is not None and self.extract_first_decimal(current_line_value) != self.extract_first_decimal(self.previous_line_value):
            self.line_changes.append((epoch, current_line_value))
            # changes_detected = True
        self.previous_line_value = current_line_value
        # Print all metrics if any changes detected
        if changes_detected:
            self.print_all_metrics()
    def update_max(self, metric_name, epoch, value):
        if value > self.max_values[metric_name][1]:
            self.max_values[metric_name] = (epoch, value)
    def extract_first_decimal(self, value):
        return int((value * 10) % 10)
    def print_all_metrics(self):
        self.print_metrics('bbox_mAP', self.bbox_map)
        self.print_metrics('bbox_mAP_50', self.bbox_map_50)
        if self.recordMetCha_OtherName is not None:
            self.print_metrics(self.recordMetCha_OtherName, self.tp_changes)
        self.print_metrics(self.recordMetCha_lineName, self.line_changes)
        self.print_max_metrics()
    def print_max_metrics(self):
        max_bbox_map = self.max_values['bbox_mAP']
        max_bbox_map_50 = self.max_values['bbox_mAP_50']
        max_line = self.max_values[self.recordMetCha_lineName]
        if self.recordMetCha_OtherName is not None:
            max_tp = self.max_values[self.recordMetCha_OtherName]
            max_str = (f"Max bbox_mAP: e{max_bbox_map[0]}={max_bbox_map[1]:.4f}, "
                       f"bbox_mAP_50: e{max_bbox_map_50[0]}={max_bbox_map_50[1]:.4f}, "
                       f"self.recordMetCha_OtherName: e{max_tp[0]}={max_tp[1]:.4f}, "
                       f"{self.recordMetCha_lineName}: e{max_line[0]}={max_line[1]:.4f}")
        else:
            max_str = (f"Max bbox_mAP: e{max_bbox_map[0]}={max_bbox_map[1]:.4f}, "
                       f"bbox_mAP_50: e{max_bbox_map_50[0]}={max_bbox_map_50[1]:.4f}, "
                       f"{self.recordMetCha_lineName}: e{max_line[0]}={max_line[1]:.4f}")
        logger: MMLogger = MMLogger.get_current_instance()
        logger.info(max_str)
    def print_metrics(self, metric_name, metrics_list):
        if metrics_list:
            metrics_str = ', '.join([f"e{e}={v:.4f}" for e, v in metrics_list])
            logger: MMLogger = MMLogger.get_current_instance()
            logger.info(f"{metric_name}: {metrics_str}")

    # copy from TopoNet/projects/toponet/datasets/openlanev2_subset_A_dataset.py
    def line_format_gt(self,gts): #return img_id.lane_centerline.points
        gt_dict = {}
        if 'line_anns' in gts[0].keys():
            for idx, gt in enumerate(gts):
                key = gt['img_id']
                lanes = gt['line_anns']['line_points'] #.line_labels; .line_points
                line_labels = gt['line_anns']['line_labels']
                info = dict(lane_centerline=[])
                if len(lanes) ==0:
                    gt_dict[key] = info
                    continue  
                if self.line_pre is not None and (lanes.shape[-2] !=self.points_for_lossMetric): #----------
                    lanes = prepare_line_points(lanes,self.line_points_inter_method,self.points_for_lossMetric) #input=[1,num_points,pts_dim]
                for idx_, (lane, line_label) in enumerate(zip(lanes, line_labels)):
                    lc_info = dict(points = lane.numpy().astype(np.float32),label = line_label.item())
                    info['lane_centerline'].append(lc_info)
                gt_dict[key] = info
        else: #self._coco_api_line is not None
            cat_ids_line = self._coco_api_line.get_cat_ids(cat_names=self.dataset_meta['line_classes'])
            cat2label_line = {cat_id: i for i, cat_id in enumerate(cat_ids_line)}
            for idx, gt in enumerate(gts):
                img_id = gt['img_id'] 
                ann_ids = self._coco_api_line.get_ann_ids(img_ids=[img_id])
                raw_ann_info = self._coco_api_line.load_anns(ann_ids)
                info = dict(
                    lane_centerline=[]
                )
                for i, ann in enumerate(raw_ann_info):
                    if ann.get(self.recordMetCha_lineName, None) is not None and ann['category_id'] in cat_ids_line:
                        if len(torch.tensor(ann[self.recordMetCha_lineName]).shape)==2: #******注意：一个标注=多条线合并在一个box中，******=3不需要处理
                            points_input = torch.tensor(ann[self.recordMetCha_lineName]).unsqueeze(0)
                        elif len(torch.tensor(ann[self.recordMetCha_lineName]).shape)==3:
                            points_input = torch.tensor(ann[self.recordMetCha_lineName])
                        if self.line_pre is not None and (torch.tensor(ann[self.recordMetCha_lineName]).shape[-2] !=self.points_for_lossMetric): #----------
                            points = prepare_line_points(points_input,self.line_points_inter_method,self.points_for_lossMetric) #input=[1,num_points,pts_dim]
                        else:
                            points = points_input
                        for points_i in points: 
                            lc_info = dict(points = points_i.numpy().astype(np.float32),label=cat2label_line[ann['category_id']])
                            info['lane_centerline'].append(lc_info)
                gt_dict[img_id] = info
        return gt_dict
    # copy from TopoNet/projects/toponet/datasets/openlanev2_subset_A_dataset.py
    def line_format_results(self, preds, gt=False): #return img_id.lane_centerline.points and img_id.lane_centerline.confidence
        pred_dict = {}
        # pred_dict['method'] = 'TopoNet'
        # pred_dict['authors'] = []
        # pred_dict['e-mail'] = 'dummy'
        # pred_dict['institution / company'] = 'OpenDriveLab'
        # pred_dict['country / region'] = 'CN'
        for idx, result in enumerate(preds):
            key = result['img_id']

            pred_info = dict(
                lane_centerline=[]
            )

            if result.get('line_points',None) is not None:
                scores = result['line_scores']
                valid_indices = np.argsort(-scores) # sort from large to small
                lanes = result['line_points'][valid_indices]
                lanes = lanes.reshape(-1, lanes.shape[-1] // 2, 2) #num_lanes,num_points,pts_dim=2
                if self.line_pre is not None and (torch.from_numpy(lanes).shape[-2] !=self.points_for_lossMetric):
                    lanes = prepare_line_points(torch.from_numpy(lanes),self.line_points_inter_method,self.points_for_lossMetric).numpy() 
                scores = scores[valid_indices]
                for pred_idx, (lane, score) in enumerate(zip(lanes, scores)):
                    lc_info = dict(
                        id = 10000 + pred_idx,
                        points = lane.astype(np.float32), #num_points,pts_dim=2
                        confidence = score.item()
                    )
                    pred_info['lane_centerline'].append(lc_info)

            pred_dict[key] = pred_info

        return pred_dict
    # copy from mmdet/evaluation/metrics/coco_metric.py and change
    def results2json(self, results: Sequence[dict],
                     outfile_prefix: str) -> dict:
        """Dump the detection results to a COCO style json file.

        There are 3 types of results: proposals, bbox predictions, mask
        predictions, and they have different data types. This method will
        automatically recognize the type, and dump them to json files.

        Args:
            results (Sequence[dict]): Testing results of the
                dataset.
            outfile_prefix (str): The filename prefix of the json files. If the
                prefix is "somepath/xxx", the json files will be named
                "somepath/xxx.bbox.json", "somepath/xxx.segm.json",
                "somepath/xxx.proposal.json".

        Returns:
            dict: Possible keys are "bbox", "segm", "proposal", and
            values are corresponding filenames.
        """
	#--------------------------cls_infos---------------5
        if self.cls_infos is not None:
            cat_ids = self._coco_api.getCatIds()
        else: #--------------------------cls_infos---------------5
            cat_ids = self.cat_ids+self.cat_ids_line #--zhou
        bbox_json_results = []
        segm_json_results = [] if 'masks' in results[0] else None
        for idx, result in enumerate(results):
            image_id = result.get('img_id', idx)
            labels = result['labels']
            bboxes = result['bboxes']
            scores = result['scores']
            # bbox results
            for i, label in enumerate(labels):
                data = dict()
                data['image_id'] = image_id
                data['bbox'] = self.xyxy2xywh(bboxes[i])
                data['score'] = float(scores[i])
                data['category_id'] = cat_ids[int(label)]  # ------------zhou-cls_infos
                bbox_json_results.append(data)

            if segm_json_results is None:
                continue

            # segm results
            masks = result['masks']
            mask_scores = result.get('mask_scores', scores)
            for i, label in enumerate(labels):
                data = dict()
                data['image_id'] = image_id
                data['bbox'] = self.xyxy2xywh(bboxes[i])
                data['score'] = float(mask_scores[i])
                data['category_id'] = cat_ids[label]  # ------------zhou
                if isinstance(masks[i]['counts'], bytes):
                    masks[i]['counts'] = masks[i]['counts'].decode()
                data['segmentation'] = masks[i]
                segm_json_results.append(data)

        result_files = dict()
        result_files['bbox'] = f'{outfile_prefix}.bbox.json'
        result_files['proposal'] = f'{outfile_prefix}.bbox.json'
        dump(bbox_json_results, result_files['bbox'])

        if segm_json_results is not None:
            result_files['segm'] = f'{outfile_prefix}.segm.json'
            dump(segm_json_results, result_files['segm'])

        return result_files
    
# if __name__ == '__main__':  
#     '''
#     debug filterLineBoxByIoU
#     '''
#     from mmdet.datasets.api_wrappers import COCO
#     from mmyolo.evaluation.metrics.line_coco_metric import filterLineBoxByIoU
#     import torch
#     data_root = '/media/zcf/Elements/dataset/mobile_screen/0LCDMobileScreen/0LCD231201/0LCD240316/' ############local
#     train_ann_file = data_root+'LCDhalf_trainval_20240404_ReDup.json' #data_root+'data_train0704.json'
#     val_ann_file = data_root+'LCDhalf_test_20240404_ReDup.json' # data_root+'data_val0704.json'
#     train_ann_file_filter = data_root+'LCDhalf_trainval_20240404_ReDup_lineIoU0d5.json' # data_root+'data_val0704.json'
#     val_ann_file_filter = data_root+'LCDhalf_test_20240404_ReDup_lineIoU0d5.json' # data_root+'data_val0704.json'
#     class_name = ('TP','line','mura','sq') 
#     class_id = class_name.index('line')+1
    
#     cocoVal = COCO(train_ann_file)
#     cocoVal_lineIoU0d5 = COCO(train_ann_file_filter)
#     line_img_ids = cocoVal.get_img_ids(cat_ids=class_id)
#     for i in line_img_ids:  
#         line_ann_ids = cocoVal.get_ann_ids(img_ids=i,cat_ids=class_id)
#         line_anns = cocoVal.load_anns(ids=line_ann_ids)
#         obj_xx = torch.tensor([i['bbox'] for i in line_anns])
#         obj_line = torch.tensor([i['line'] for i in line_anns]).flatten(1)
#         obj_xx[:,2] = obj_xx[:,0]+obj_xx[:,2]
#         obj_xx[:,3] = obj_xx[:,1]+obj_xx[:,3] #xywh-->xyxy
#         scores = torch.ones(obj_xx.shape[0]).float()
#         obj_xx_up,scores_up = filterLineBoxByIoU(obj_xx,scores,obj_line,iou_threshold=0.5,score_threshold=0.3)
#         if len(obj_xx_up) < len(obj_xx):
#             img_info = cocoVal.load_imgs(ids=i)
#             line_ann_ids_2 = cocoVal_lineIoU0d5.get_ann_ids(img_ids=i,cat_ids=class_id)
#             line_anns_2 = cocoVal_lineIoU0d5.load_anns(ids=line_ann_ids_2)
            
#             obj_xx_2 = torch.tensor([i['bbox'] for i in line_anns_2])
#             obj_xx_2[:,2] = obj_xx_2[:,0]+obj_xx_2[:,2]
#             obj_xx_2[:,3] = obj_xx_2[:,1]+obj_xx_2[:,3] #xywh-->xyxy
#             print(img_info,obj_xx)
#             print(obj_xx_up)
#             print(obj_xx_2)
# if __name__ == '__main__':  
#     '''
#     debug merge_detections
#     '''
#     from mmdet.datasets.api_wrappers import COCO
#     import numpy as np
#     from tqdm import tqdm
#     import torch
#     def calculate_common_iou(boxes1, boxes2):
#         """
#         计算两组边界框之间的交并比 (IoU)
#         boxes1: (N, 4) 数组,表示 N 个边界框的 (x1, y1, x2, y2) 坐标
#         boxes2: (M, 4) 数组,表示 M 个边界框的 (x1, y1, x2, y2) 坐标
#         返回: (N, M) 数组,表示每对边界框之间的 IoU 值
#         """
#         # 计算交叉区域的坐标
#         x1 = np.maximum(boxes1[:, None, 0], boxes2[:, 0])  # (N, M)
#         y1 = np.maximum(boxes1[:, None, 1], boxes2[:, 1])  # (N, M)
#         x2 = np.minimum(boxes1[:, None, 2], boxes2[:, 2])  # (N, M)
#         y2 = np.minimum(boxes1[:, None, 3], boxes2[:, 3])  # (N, M)
    
#         # 计算交叉区域的面积
#         intersection_area = np.maximum(x2 - x1, 0) * np.maximum(y2 - y1, 0)  # (N, M)
    
#         # 计算每个边界框的面积
#         area_boxes1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])  # (N,)
#         area_boxes2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])  # (M,)
    
#         # 计算交并比 (IoU)
#         iou = intersection_area / (area_boxes1[:, None] + area_boxes2 - intersection_area)  # (N, M)
    
#         return iou
#     right_start = 6450-3280
#     left_end = 3280
#     Merge_score_thr = 0.3
#     data_pre = '/media/zcf/Elements/dataset/mobile_screen/0LCDMobileScreen/0LCD231201/0LCD240316/'
#     fullImgAnn = data_pre+'LCDfull_all_20240404_ReDup.json'
#     halfImgAnn = data_pre+'LCDhalf_all_20240404_ReDup.json'
    
#     full_coco = COCO(fullImgAnn)
#     half_coco = COCO(halfImgAnn)
#     # results tuple(dict,dict,dict,dict...) #dict_keys(['img_id', 'bboxes', 'scores', 'labels', 'line_points', 'line_scores', 'line_labels'])
    
#     half_ImgIds = half_coco.getImgIds()
#     half_coco.get_cat_ids(cat_names=('TP','line','mura','sq'))
#     results = []
#     for i in tqdm(half_ImgIds):
#         ann_ids = half_coco.get_ann_ids(img_ids=i)
#         anns = half_coco.load_anns(ids=ann_ids)
#         obj_xx = np.array([i['bbox'] for i in anns]).reshape(-1, 4)
#         obj_xx[:,2] = obj_xx[:,0]+obj_xx[:,2]
#         obj_xx[:,3] = obj_xx[:,1]+obj_xx[:,3] #xywh-->xyxy
#         scores = np.ones(obj_xx.shape[0]).astype(float)
#         result_per = dict()
#         result_per['img_id'] = i
#         result_per['bboxes'] = obj_xx
#         result_per['scores'] = scores
#         result_per['labels'] = np.array([i['category_id'] for i in anns])
#         results.append(result_per)
#     results = tuple(results)
#     #键是file_name,值是对应结果在results列表中的索引。     
#     file_name_indices = { half_coco.loadImgs(ids=[result['img_id']])[0]['file_name']: idx for idx, result in enumerate(results) }
#     ####################################################################
#     full_Imgids = full_coco.getImgIds()
#     full_ImgInfos = full_coco.loadImgs(ids=full_Imgids)
#     full_nameIds = {i['file_name']:ids for ids,i in zip(full_Imgids,full_ImgInfos)}
    
#     for i in full_Imgids:
#         # if i not in [659]:
#         #     continue
#         full_ImgInfos = full_coco.loadImgs(ids=i)
#         left_preds_index = file_name_indices[full_ImgInfos[0]['file_name'].replace('.jpg', '_l.jpg')]
#         right_preds_index = file_name_indices[full_ImgInfos[0]['file_name'].replace('.jpg', '_r.jpg')]
#         left_preds = results[left_preds_index]
#         right_preds = results[right_preds_index]
#         # 重映射到整图中
#         right_preds['bboxes'][:,0] += right_start
#         right_preds['bboxes'][:,2] += right_start
        
#         all_pr = merge_detections(np.concatenate([left_preds['bboxes'], left_preds['scores'][:,None], left_preds['labels'][:,None]],axis=1),
#                          np.concatenate([right_preds['bboxes'], right_preds['scores'][:,None], right_preds['labels'][:,None]],axis=1))
#         all_bboxes, all_scores, all_labels = all_pr[:,:4].numpy(),all_pr[:,4].numpy(),all_pr[:,5].numpy()
#         if all_bboxes.shape[0] != (left_preds['bboxes'].shape[0]+right_preds['bboxes'].shape[0]):
#             print('merged!!!!!!')     
#         #--------------------------full_coco.loadImgs(ids=i)
#         ann_ids = full_coco.get_ann_ids(img_ids=i)
#         anns = full_coco.load_anns(ids=ann_ids)
#         obj_xx = np.array([i['bbox'] for i in anns]).reshape(-1, 4)
#         obj_xx[:,2] = obj_xx[:,0]+obj_xx[:,2]
#         obj_xx[:,3] = obj_xx[:,1]+obj_xx[:,3] #xywh-->xyxy
#         scores = np.ones(obj_xx.shape[0]).astype(float)
#         all_bboxes_gt, all_scores_gt, all_labels_gt = obj_xx,scores,np.array([i['category_id'] for i in anns])
#         #--------------------------full_coco.loadImgs(ids=i)
#         if all_bboxes.shape[0] != all_bboxes_gt.shape[0]:
#             print('error',i)
#         elif all_bboxes.shape[0] == all_bboxes_gt.shape[0]:
#             ious = calculate_common_iou(all_bboxes,all_bboxes_gt)
#             # 检查是否是一对一匹配(ious=1.0)
#             ious_new = ious==1.0
#             is_one_to_one = np.all(ious_new.sum(axis=0) == 1) and np.all(ious_new.sum(axis=1) == 1)
#             if not is_one_to_one:
#                 print('error',i)
                
if __name__ == '__main__':  
    '''
    debug merge_detections
    '''
    from mmengine.fileio import load
    import torch 
    from mmyolo.evaluation import LineCocoMetric
    from mmyolo.evaluation.metrics.line_coco_metric import line2box_torch,filterLineBoxByIoU
    from tqdm import tqdm
    import numpy as np
    overlap_end = 3280
    overlap_start = 3280-110  
    points_for_lossMetric = 4
    num_classes = 4
    loss_bbox_pre=dict(line_points_inter_method ='bezier',# lineSegmentUni,bezier
                        points_for_lossMetric = points_for_lossMetric,
                        inter_reg = None) #None
     
    data_prefix = '/media/zcf/Elements/dataset/mobile_screen/0LCDMobileScreen/0LCD231201/0LCD240316/'
    prediction_path = data_prefix+'0work_dirs1/ex6_lr0d000005/xx_test.pkl'
    val_ann_file = 'anns_20240404_saved/LCDhalf_test_20240404_ReDup.json'
    outputs = load(prediction_path)
    
    metricIns = LineCocoMetric(ann_file=data_prefix + val_ann_file,
                         metric='bbox',
                         classwise=True,
                         line_pre=loss_bbox_pre,
                         box_num_cls=num_classes,
                         merged_preds_cfg=dict(merged_preds=True),#两张半图合并结果在整图上，得到metric
                         )
    class_name = ('TP','mura','sq') #('bubble','scratch','pinhole','tin_ash')
    num_classes = len(class_name) # Number of classes for classification
    line_classes = ('line',)
    num_line_classes = len(line_classes) 
    metainfo = dict(classes=class_name,line_classes=line_classes,bbox_include_line=False,palette=[(220, 20, 60), (119, 11, 32), (0, 0, 142), (0, 0, 230)])
    metricIns.dataset_meta = metainfo
    results = []
    for i in tqdm(outputs):
        result_per = dict()
        result_per['img_id'] = i['img_id']
        bboxes2 = line2box_torch(i['line_pred_instances']['line_points'])
        scores2 = i['line_pred_instances']['line_scores']
        labels2 = i['line_pred_instances']['line_labels']+3
        # line_nms_iou_thre = 0.5
        # line_nms_socre_thre = 1.0
        # if line_nms_iou_thre is not None:
        #     line2 = i['line_pred_instances']['line_points']
        #     all_bboxes,all_scores,all_labels = [], [], []
        #     for label_i in labels2.unique():
        #         indices_i = labels2 == label_i
        #         bboxes2_i, scores2_i = filterLineBoxByIoU(bboxes2[indices_i], scores2[indices_i], line2[indices_i], iou_threshold=line_nms_iou_thre,score_threshold=line_nms_socre_thre)
        #         labels2_i = torch.full_like(scores2_i, fill_value=label_i, dtype=labels2.dtype)
        #         all_bboxes.append(bboxes2_i)
        #         all_scores.append(scores2_i)
        #         all_labels.append(labels2_i)
        #     bboxes2,scores2,labels2 = torch.cat(all_bboxes, dim=0),torch.cat(all_scores, dim=0),torch.cat(all_labels, dim=0)
        result_per['bboxes'] = np.concatenate((i['pred_instances']['bboxes'].numpy(), bboxes2.numpy()), axis=0)
        result_per['scores'] = np.concatenate((i['pred_instances']['scores'].numpy(), scores2.numpy()), axis=0)
        result_per['labels'] = np.concatenate((i['pred_instances']['labels'].numpy(), labels2.numpy()), axis=0)
        results.append((result_per,result_per))
    results = tuple(results)
    wo = metricIns.compute_metrics(results)
