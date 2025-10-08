# ****************Latest version 2025-03-31-17:16:00
# Copyright (c) OpenMMLab. All rights reserved.
from typing import Union,Optional 

import torch
from mmdet.models.data_preprocessors import DetDataPreprocessor
from mmengine.structures import BaseDataElement
from mmdet.models.utils.misc import samplelist_boxtype2tensor

from mmyolo.registry import MODELS
from mmengine.utils import is_seq_of
from mmengine.model import stack_batch
import math
import torch.nn.functional as F
from mmdet.structures import SampleList
from imgaug.augmentables.lines import LineString, LineStringsOnImage
from shapely.geometry import LineString as ShapelyLineString
import numpy as np
from math import factorial
CastData = Union[tuple, dict, BaseDataElement, torch.Tensor, list, bytes, str,
                 None]
pts_dim = 2 
# copy from /topomlp/models/heads/lane_head.py
def control_points_to_lane_points(lanes,n_points): # return [num_lines,n_points,points_dim=2]
    # lanes = lanes.reshape(1, -1, pts_dim)

    def comb(n, k):
        return factorial(n) // (factorial(k) * factorial(n - k))

    n_control = lanes.shape[1]
    A = np.zeros((n_points, n_control))
    t = np.arange(n_points) / (n_points - 1)
    for i in range(n_points):
        for j in range(n_control):
            A[i, j] = comb(n_control - 1, j) * np.power(1 - t[i], n_control - 1 - j) * np.power(t[i], j)
    bezier_A = torch.tensor(A, dtype=torch.float32).to(lanes.device)
    lanes = torch.einsum('ij,njk->nik', bezier_A, lanes)
    # lanes = lanes.reshape(lanes.shape[0], -1)

    return lanes 
# copy from toponet
def fix_pts_interpolate(ls_1,n_points): # return [n_points,points_dim=2]
    ls = ShapelyLineString(ls_1.cpu())
    distances = np.linspace(0, ls.length,n_points)
    # lane = np.array([ls.interpolate(distance).coords[0] for distance in distances])
    lane = torch.tensor([ls.interpolate(distance).coords[0] for distance in distances]).to(ls_1.device)
    return lane

# def interpolate_points_detach(points, n_points):
#     """
#     points: [num_lanes, num_points, pts_dim]
#     n_points: 目标点数
#     returns: ratios [num_lanes, n_points, num_points]
#     """
#     eps_here = 1e-6 # 添加epsilon防止除零
#     points_ori = points
#     num_lanes, num_points, _ = points_ori.shape
    
#     # 计算累积距离
#     segments = points_ori[:, 1:] - points_ori[:, :-1]
#     distances = torch.norm(segments, dim=2)
#     cum_distances = torch.cat([torch.zeros(num_lanes, 1).to(points_ori.device), 
#                              torch.cumsum(distances, dim=1)], dim=1)
    
#     # 归一化距离并计算目标位置
#     cum_distances = cum_distances / (cum_distances[:, -1:] + eps_here)  # 添加epsilon防止除零
#     target_distances = torch.linspace(0, 1, n_points).to(points_ori.device)
    
#     # 计算权重矩阵
#     weights = torch.zeros(num_lanes, n_points, num_points).to(points_ori.device)
#     # 第一个点和最后一个点直接赋值
#     weights[:, 0, 0] = 1.0
#     weights[:, -1, -1] = 1.0
    
#     for i in range(1,n_points-1):
#         dist = target_distances[i]
#         # 找到目标距离所在的区间
#         mask = (cum_distances[:, :-1] <= dist) & (cum_distances[:, 1:] >= dist)
#         idx = torch.where(mask)[1]
        
#         # 计算插值比例
#         ratio = (dist - cum_distances[torch.arange(num_lanes), idx]) / \
#                 (cum_distances[torch.arange(num_lanes), idx + 1] - 
#                  cum_distances[torch.arange(num_lanes), idx]+ eps_here)
        
#         # 设置权重
#         weights[torch.arange(num_lanes), i, idx] = 1 - ratio
#         weights[torch.arange(num_lanes), i, idx + 1] = ratio
    
#     interpolated_points = torch.bmm(weights, points)

#     return interpolated_points

def linear_interpolate(line, n_points):
    # 计算两点之间的向量
    vector = line[1] - line[0]
    # 生成插值比例，形状为[n_points, 1]
    steps = torch.linspace(0, 1, n_points, device=line.device).unsqueeze(1)   
    # 根据比例插值：line[0].unsqueeze(0) [1,num_points_before], steps[1,num_points_after,1], vector[num_points_before]
    interpolated_points = line[0].unsqueeze(0) + steps * vector
    return interpolated_points
def prepare_line_points(line_points,line_points_inter_method,n_points): #input line_points=[num_lanes,num_points,pts_dim],#return line_points=[num_lanes,n_points,pts_dim]
    if len(line_points)==0:
        line_points_return = line_points.reshape((-1,n_points,line_points.shape[-1]))
    elif line_points_inter_method =='bezier':
        line_points_return = control_points_to_lane_points(line_points,n_points) 
    elif line_points_inter_method =='lineSegmentUni':
        if line_points.shape[-2] ==2:
            line_points_return = [linear_interpolate(line, n_points) for line in line_points]
            line_points_return = torch.stack(line_points_return)
        else:
            line_points_return = [fix_pts_interpolate(line, n_points) for line in line_points]  #-----240815_NumLinePoints
            line_points_return = torch.stack(line_points_return)
            # line_points_return = interpolate_points_detach(line_points, n_points) 
    return line_points_return

def linestrings_to_points(lines,line_fix_point): # return [num_lines,n_points,points_dim=2]
    n_points = 2 #defalut
    if line_fix_point is not None:
        line_points_inter_method = line_fix_point['line_points_inter_method']
        n_points = line_fix_point['points_for_lossMetric']
        if n_points =='defalut':
            n_points = len(lines[0].coords) 
    lanes = []
    for line in lines: #line imgaug.augmentables.lines.LineString([(283.21, 588.22), (0.00, 588.70)], label=None)=[(x1,y1),(x2,y2)]
        if line_fix_point is None or (len(line.coords) == n_points): 
            lanes.append(torch.from_numpy(line.coords).unsqueeze(0))
        else:
            lanes.append(prepare_line_points(torch.from_numpy(line.coords).unsqueeze(0),line_points_inter_method,n_points))
    if lanes != []:
        return torch.cat(lanes,dim=0)
    else:
        return torch.tensor(lanes).reshape((-1,n_points,pts_dim))

def samplelist_LineStringsOnImage2tensor(batch_data_samples: SampleList, line_fix_point,
                                         key_names=['gt_line_instances','pred_line_instances','ignored_line_instances']) -> SampleList:
    for data_samples in batch_data_samples:
        for key_name in key_names:
            if key_name in data_samples:
                line_points = data_samples.get(key_name).get('line_points', None)
                if isinstance(line_points, LineStringsOnImage):
                    device = data_samples.get(key_name).get('line_labels').device
                    data_samples.get(key_name).line_points = linestrings_to_points(line_points,line_fix_point).to(device=device)
                    # data_samples.get(key_name).line_points, torch.Size([num_lines,'points_for_lossMetric',points_dim=2])
@MODELS.register_module()
class Det_YOLOv5DetDataPreprocessor(DetDataPreprocessor):
    
    def __init__(self, *args, non_blocking: Optional[bool] = True, LineStringsOnImage2tensor: bool = True, line_fix_point = None,
                 **kwargs): #line_fix_point={'line_points_inter_method': 'bezier', 'points_for_lossMetric': 11, 'inter_reg': False}
        super().__init__(*args, non_blocking=non_blocking, **kwargs)
        self.LineStringsOnImage2tensor = LineStringsOnImage2tensor
        self.line_fix_point = line_fix_point
    
    def forward(self, data: dict, training: bool = False) -> dict:
        """Perform normalization、padding and bgr2rgb conversion based on
        ``BaseDataPreprocessor``.

        Args:
            data (dict): Data sampled from dataloader.
            training (bool): Whether to enable training time augmentation.

        Returns:
            dict: Data in the same format as the model input.
        """
        batch_pad_shape = self._get_pad_shape(data)
        # 因为不想嵌套太多
        # data = super().super().forward(data=data, training=training)
        data = self.BaseDataPreprocessor_forward(data=data, training=training)
        inputs, data_samples = data['inputs'], data['data_samples']

        if data_samples is not None:
            # NOTE the batched image size information may be useful, e.g.
            # in DETR, this is needed for the construction of masks, which is
            # then used for the transformer_head.
            batch_input_shape = tuple(inputs[0].size()[-2:])
            for data_sample, pad_shape in zip(data_samples, batch_pad_shape):
                data_sample.set_metainfo({
                    'batch_input_shape': batch_input_shape,
                    'pad_shape': pad_shape
                })

            if self.boxtype2tensor:
                samplelist_boxtype2tensor(data_samples)
            # 因为只pad下方和右方，所有box和line不用处理,
            # 可以改变line的img shape，但好像用不到，就不更改了
            if self.LineStringsOnImage2tensor:
                samplelist_LineStringsOnImage2tensor(data_samples,self.line_fix_point,key_names=['gt_line_instances','pred_line_instances','ignored_line_instances'])

            if self.pad_mask and training:
                self.pad_gt_masks(data_samples)

            if self.pad_seg and training:
                self.pad_gt_sem_seg(data_samples)

        if training and self.batch_augments is not None:
            print('TODO! line type has not been processed yet')
            for batch_aug in self.batch_augments:
                inputs, data_samples = batch_aug(inputs, data_samples)
        if not training or ('data_samples_yolov5' not in data):
            return {'inputs': inputs, 'data_samples': data_samples}
        data_samples_yolov5 = data['data_samples_yolov5']
        img_metas = [{'batch_input_shape': inputs.shape[2:]}] * len(inputs)
        data_samples_output = {
            'bboxes_labels': data_samples_yolov5['bboxes_labels'],
            'img_metas': img_metas
        }
        if 'masks' in data_samples_yolov5:
            data_samples_output['masks'] = data_samples_yolov5['masks']
        if 'keypoints' in data_samples_yolov5:
            data_samples_output['keypoints'] = data_samples_yolov5['keypoints']
            data_samples_output['keypoints_visible'] = data_samples_yolov5[
                'keypoints_visible']
        # from mmengine.visualization import Visualizer
        # import copy
        # visualizer = Visualizer.get_current_instance()
        # vis_input = inputs[0].cpu().numpy().transpose(1, 2, 0)
        # data_samples_copy = copy.deepcopy(data_samples)
        # visualizer.add_datasample('result2',vis_input,data_samples[0])
        return {'inputs': inputs, 'data_samples': data_samples, 'data_samples_yolov5': data_samples_output}
    
    def BaseDataPreprocessor_forward(self, data: dict, training: bool = False) -> Union[dict, list]:
        """Performs normalization、padding and bgr2rgb conversion based on
        ``BaseDataPreprocessor``.

        Args:
            data (dict): Data sampled from dataset. If the collate
                function of DataLoader is :obj:`pseudo_collate`, data will be a
                list of dict. If collate function is :obj:`default_collate`,
                data will be a tuple with batch input tensor and list of data
                samples.
            training (bool): Whether to enable training time augmentation. If
                subclasses override this method, they can perform different
                preprocessing strategies for training and testing based on the
                value of ``training``.

        Returns:
            dict or list: Data in the same format as the model input.
        """
        data = self.cast_data(data)  # type: ignore
        _batch_inputs = data['inputs']
        # Process data with `pseudo_collate`.
        if is_seq_of(_batch_inputs, torch.Tensor):
            batch_inputs = []
            for _batch_input in _batch_inputs:
                # channel transform
                if self._channel_conversion:
                    _batch_input = _batch_input[[2, 1, 0], ...]
                # Convert to float after channel conversion to ensure
                # efficiency
                _batch_input = _batch_input.float()
                # Normalization.
                if self._enable_normalize:
                    if self.mean.shape[0] == 3:
                        assert _batch_input.dim(
                        ) == 3 and _batch_input.shape[0] == 3, (
                            'If the mean has 3 values, the input tensor '
                            'should in shape of (3, H, W), but got the tensor '
                            f'with shape {_batch_input.shape}')
                    _batch_input = (_batch_input - self.mean) / self.std
                batch_inputs.append(_batch_input)
            # Pad and stack Tensor.
            batch_inputs = stack_batch(batch_inputs, self.pad_size_divisor,
                                       self.pad_value)
        # Process data with `default_collate`.
        elif isinstance(_batch_inputs, torch.Tensor):
            assert _batch_inputs.dim() == 4, (
                'The input of `ImgDataPreprocessor` should be a NCHW tensor '
                'or a list of tensor, but got a tensor with shape: '
                f'{_batch_inputs.shape}')
            if self._channel_conversion:
                _batch_inputs = _batch_inputs[:, [2, 1, 0], ...]
            # Convert to float after channel conversion to ensure
            # efficiency
            _batch_inputs = _batch_inputs.float()
            if self._enable_normalize:
                _batch_inputs = (_batch_inputs - self.mean) / self.std
            h, w = _batch_inputs.shape[2:]
            target_h = math.ceil(
                h / self.pad_size_divisor) * self.pad_size_divisor
            target_w = math.ceil(
                w / self.pad_size_divisor) * self.pad_size_divisor
            pad_h = target_h - h
            pad_w = target_w - w
            batch_inputs = F.pad(_batch_inputs, (0, pad_w, 0, pad_h),
                                 'constant', self.pad_value)
        else:
            raise TypeError('Output of `cast_data` should be a dict of '
                            'list/tuple with inputs and data_samples, '
                            f'but got {type(data)}: {data}')
        data['inputs'] = batch_inputs
        data.setdefault('data_samples', None)
        return data