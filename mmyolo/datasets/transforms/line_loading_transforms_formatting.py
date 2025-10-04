"""
Created on Wed Jan 17 20:05:37 2024

@author: zcf
"""
import math
from copy import deepcopy
from typing import List, Sequence, Tuple, Union

import cv2
import mmcv
import numpy as np
import torch
from mmcv.image.geometric import _scale_size
from mmcv.transforms import BaseTransform, Compose
from mmcv.transforms.utils import cache_randomness
from mmdet.datasets.transforms import FilterAnnotations as FilterDetAnnotations
from mmdet.datasets.transforms import LoadAnnotations as MMDET_LoadAnnotations
from mmdet.datasets.transforms import RandomAffine as MMDET_RandomAffine
from mmdet.datasets.transforms import RandomFlip as MMDET_RandomFlip
from mmdet.datasets.transforms import Resize as MMDET_Resize
from mmdet.structures.bbox import (HorizontalBoxes, autocast_box_type,
                                   get_box_type)
from mmdet.structures.mask import PolygonMasks, polygon_to_bitmap
from numpy import random

from mmyolo.registry import TRANSFORMS
import imgaug.augmenters as iaa
from imgaug.augmentables.lines import LineString, LineStringsOnImage
from mmdet.datasets.transforms import PackDetInputs as MMDET_PackDetInputs
from mmcv.transforms import to_tensor
from mmengine.structures import InstanceData, PixelData
from mmdet.structures import DetDataSample
from mmdet.structures.bbox import BaseBoxes

from typing import Optional
import warnings
import mmengine.fileio as fileio
from mmengine.logging import MMLogger
import os
# TODO: Waiting for MMCV support
TRANSFORMS.register_module(module=Compose, force=True)

@TRANSFORMS.register_module()
class LineLoadAnnotations(MMDET_LoadAnnotations):
    '''
        Required Keys:
    
        - height
        - width
        - instances
          - bbox (optional)
          - bbox_label
          - mask (optional)
          - ignore_flag
          - line_points (add by zhou)==========
          - line_labels (add by zhou)==========
        - seg_map_path (optional)
    
        Added Keys:
    
        - gt_bboxes (BaseBoxes[torch.float32])
        - gt_bboxes_labels (np.int64)
        - gt_masks (BitmapMasks | PolygonMasks)
        - gt_seg_map (np.uint8)
        - gt_ignore_flags (bool)
        - gt_line_points (LineStringsOnImage) (add by zhou)==========
        - gt_labels (np.int64)  (add by zhou)==========
        - gt_line_ignore_flags (bool) (add by zhou)==========
    ''' 

    def __init__(self,
                 with_line: bool = True,
                 poly2mask: bool = False,
                 **kwargs):
        assert not poly2mask, 'Does not support BitmapMasks considering ' \
                              'that bitmap consumes more memory.'
        super().__init__(poly2mask=poly2mask, **kwargs)
        self.with_line = with_line

    def transform(self, results: dict) -> dict:
        """Function to load multiple types annotations.

        Args:
            results (dict): Result dict from :obj:``mmengine.BaseDataset``.

        Returns:
            dict: The dict contains loaded bounding box, label and
            semantic segmentation.
        """
        if self.with_bbox:
            self._load_bboxes(results)
        if self.with_label:
            self._load_key_labels(results,'bbox_label')
        if self.with_mask: #------not check-------------
            self._load_masks(results)
        if self.with_seg: #------not check-------------
            self._load_seg_map(results)
        if self.with_line:
            self._load_key_labels(results,'line_labels')
            self._load_line_points(results,'line_points')
            
        return results
    
    def _load_bboxes(self, results: dict) -> None:
        """Private function to load bounding box annotations.

        Args:
            results (dict): Result dict from :obj:``mmengine.BaseDataset``.
        Returns:
            dict: The dict contains loaded bounding box annotations.
        """
        gt_bboxes = []
        gt_ignore_flags = []
        for instance in results.get('instances', []):
            if 'bbox' in instance.keys(): #--------------add by zhou
                gt_bboxes.append(instance['bbox'])
                gt_ignore_flags.append(instance['ignore_flag'])
        if self.box_type is None:
            results['gt_bboxes'] = np.array(
                gt_bboxes, dtype=np.float32).reshape((-1, 4))
        else:
            _, box_type_cls = get_box_type(self.box_type)
            results['gt_bboxes'] = box_type_cls(gt_bboxes, dtype=torch.float32)
        results['gt_ignore_flags'] = np.array(gt_ignore_flags, dtype=bool)
    
    def _load_key_labels(self, results: dict, label_name ='bbox_label') -> None:
        """Private function to load label annotations.

        Args:
            results (dict): Result dict from :obj:``mmengine.BaseDataset``.

        Returns:
            dict: The dict contains loaded label annotations.
        """
        if label_name=='bbox_label':
            key_name = 'gt_bboxes_labels'
        else:
            key_name = 'gt_'+label_name
        gt_bboxes_labels = []
        for instance in results.get('instances', []):
            if label_name in instance.keys(): #--------------add by zhou
                if label_name=='line_labels' and len(np.array(instance['line_points']).shape)==3:
                    print('******注意：一个标注=多条线合并在一个box中，******')
                    gt_bboxes_labels.extend([instance[label_name]]*np.array(instance['line_points']).shape[0])  
                else:
                    gt_bboxes_labels.append(instance[label_name])
        # TODO: Inconsistent with mmcv, consider how to deal with it later.
        results[key_name] = np.array(
            gt_bboxes_labels, dtype=np.int64)
        
    def _load_line_points(self, results: dict, label_name ='line_points') -> None:
        gt_keypoints = []
        gt_ignore_flags = []
        for instance in results.get('instances', []):
            if label_name in instance.keys(): #--------------add by zhou
                if label_name=='line_points' and len(np.array(instance['line_points']).shape)==3:
                    for instance_i in instance[label_name]:
                        print('******注意：一个标注=多条线合并在一个box中，******')
                        gt_keypoints.append(LineString(instance_i))
                        gt_ignore_flags.append(instance['ignore_flag'])
                else:       
                    gt_keypoints.append(LineString(instance[label_name]))
                    gt_ignore_flags.append(instance['ignore_flag'])
        # 将线段放入 LineStringsOnImage 对象
        results['gt_'+label_name] = LineStringsOnImage(gt_keypoints, shape=results['img'].shape) 
        results['gt_line_ignore_flags'] = np.array(gt_ignore_flags, dtype=bool)
        
@TRANSFORMS.register_module()
class LineRandomFlip(MMDET_RandomFlip):
    '''
        Required Keys:
    
        - img
        - gt_bboxes (BaseBoxes[torch.float32]) (optional)
        - gt_masks (BitmapMasks | PolygonMasks) (optional)
        - gt_seg_map (np.uint8) (optional)
        - gt_line_points (LineStringsOnImage) (optional) (add by zhou)==========
    
        Modified Keys:
    
        - img
        - gt_bboxes
        - gt_masks
        - gt_seg_map
        - gt_line_points (LineStringsOnImage) (add by zhou)==========
    
        Added Keys:
    
        - flip
        - flip_direction
        - homography_matrix
    '''
    # change based on MMDET_RandomFlip
    @autocast_box_type()
    def _flip(self, results: dict) -> None:
        """"""
        # flip image
        results['img'] = mmcv.imflip(
            results['img'], direction=results['flip_direction'])

        img_shape = results['img'].shape[:2]

        # flip bboxes
        if results.get('gt_bboxes', None) is not None:
            results['gt_bboxes'].flip_(img_shape, results['flip_direction'])

        # flip masks
        if results.get('gt_masks', None) is not None:
            results['gt_masks'] = results['gt_masks'].flip(
                results['flip_direction'])

        # flip segs
        if results.get('gt_seg_map', None) is not None:
            results['gt_seg_map'] = mmcv.imflip(
                results['gt_seg_map'], direction=results['flip_direction'])
        
        # flip gt_line_points================add by zhou
        if results.get('gt_line_points', None) is not None:
            if results['flip_direction'] == 'horizontal':
                aug = iaa.Fliplr(1.0)  # 1.0表示100%的几率进行翻转
                results['gt_line_points'] = aug.augment_line_strings(results['gt_line_points']) #[[x1,y1],[x2,y2]],shape=[h,w,c]
            elif results['flip_direction'] == 'vertical':
                aug = iaa.Flipud(1.0)  
                results['gt_line_points'] = aug.augment_line_strings(results['gt_line_points'])
            elif results['flip_direction'] == 'diagonal':
                aug_v = iaa.Flipud(1.0)
                aug_h = iaa.Fliplr(1.0)
                # 执行对角线翻转，先垂直翻转再水平翻转
                results['gt_line_points'] = aug_h.augment_line_strings(aug_v.augment_line_strings(results['gt_line_points']))
        # flip gt_line_points================add by zhou

        # record homography matrix for flip
        self._record_homography_matrix(results)
        
@TRANSFORMS.register_module()
class LineResize(MMDET_Resize):
    '''
        Required Keys:
    
        - img
        - gt_bboxes (BaseBoxes[torch.float32]) (optional)
        - gt_masks (BitmapMasks | PolygonMasks) (optional)
        - gt_seg_map (np.uint8) (optional)
        - gt_line_points (LineStringsOnImage) (optional) (add by zhou)==========
    
        Modified Keys:
    
        - img
        - img_shape
        - gt_bboxes
        - gt_masks
        - gt_seg_map
        - gt_line_points (LineStringsOnImage) (optional) (add by zhou)==========
    
        Added Keys:
    
        - scale
        - scale_factor
        - keep_ratio
        - homography_matrix
    '''
    # change based on MMDET_Resize
    @autocast_box_type()
    def transform(self, results: dict) -> dict:
        if self.scale:
            results['scale'] = self.scale
        else:
            img_shape = results['img'].shape[:2]
            results['scale'] = _scale_size(img_shape[::-1], self.scale_factor)
        self._resize_img(results)
        self._resize_bboxes(results)
        self._resize_masks(results)
        self._resize_seg(results)
        self._resize_line_points(results) #--------add by zhou
        self._record_homography_matrix(results)
        return results
    def _resize_line_points(self, results: dict) -> None:
        """Resize bounding boxes with ``results['scale_factor']``."""
        if results.get('gt_line_points', None) is not None:
            aug = iaa.Resize({"width": results['scale_factor'][0], "height": results['scale_factor'][1]})
            results['gt_line_points'] = aug.augment_line_strings(results['gt_line_points'])
            assert results['gt_line_points'].shape == results['img'].shape, '_resize_line_points: May be an error' #[[x1,y1],[x2,y2]],shape=[h,w,c], results['img'].shape=[h,w,c]
            if self.clip_object_border:
                new_shape = results['img_shape'] #Actually, results['img_shape'] should = results['img'].shape
                if len(results['img_shape'])==2:
                    new_shape = results['img_shape'] + results['gt_line_points'].shape[2:]
                results['gt_line_points'].shape = new_shape
                results['gt_line_points'].clip_out_of_image_()
                        
@TRANSFORMS.register_module()
class LineResizeOriImg(MMDET_Resize):
    '''
        Required Keys:
    
        - img
        - gt_bboxes (BaseBoxes[torch.float32]) (optional)
        - gt_masks (BitmapMasks | PolygonMasks) (optional)
        - gt_seg_map (np.uint8) (optional)
        - gt_line_points (LineStringsOnImage) (optional) (add by zhou)==========
    
        Modified Keys:
    
        - img
        - img_shape
        - gt_bboxes
        - gt_masks
        - gt_seg_map
        - gt_line_points (LineStringsOnImage) (optional) (add by zhou)==========
    
        Added Keys:
    
        - scale
        - scale_factor
        - keep_ratio
        - homography_matrix
    '''
    # change based on MMDET_Resize
    @autocast_box_type()
    def transform(self, results: dict) -> dict:
        if self.scale:
            results['scale'] = self.scale
        else:
            img_shape = results['img'].shape[:2]
            results['scale'] = _scale_size(img_shape[::-1], self.scale_factor)
        self._resize_img(results)
        self._resize_bboxes(results)
        self._resize_masks(results)
        self._resize_seg(results)
        self._resize_line_points(results) #--------add by zhou
        self._record_homography_matrix(results)
        return results
    def _resize_line_points(self, results: dict) -> None:
        """Resize bounding boxes with ``results['scale_factor']``."""
        if results.get('gt_line_points', None) is not None:
            aug = iaa.Resize({"width": results['scale_factor'][0], "height": results['scale_factor'][1]})
            results['gt_line_points'] = aug.augment_line_strings(results['gt_line_points'])
            assert results['gt_line_points'].shape == results['img'].shape, '_resize_line_points: May be an error' #[[x1,y1],[x2,y2]],shape=[h,w,c], results['img'].shape=[h,w,c]
            if self.clip_object_border:
                new_shape = results['img_shape'] #Actually, results['img_shape'] should = results['img'].shape
                if len(results['img_shape'])==2:
                    new_shape = results['img_shape'] + results['gt_line_points'].shape[2:]
                results['gt_line_points'].shape = new_shape
                results['gt_line_points'].clip_out_of_image_()
    def _resize_img(self, results: dict) -> None:
        """Resize images with ``results['scale']``."""

        if results.get('img', None) is not None:
            results['img_ori'] = results['img'] #----------------------zcf 250602
            if self.keep_ratio:
                img, scale_factor = mmcv.imrescale(
                    results['img'],
                    results['scale'],
                    interpolation=self.interpolation,
                    return_scale=True,
                    backend=self.backend)
                # the w_scale and h_scale has minor difference
                # a real fix should be done in the mmcv.imrescale in the future
                new_h, new_w = img.shape[:2]
                h, w = results['img'].shape[:2]
                w_scale = new_w / w
                h_scale = new_h / h
            else:
                img, w_scale, h_scale = mmcv.imresize(
                    results['img'],
                    results['scale'],
                    interpolation=self.interpolation,
                    return_scale=True,
                    backend=self.backend)
            results['img'] = img
            results['img_shape'] = img.shape[:2]
            results['scale_factor'] = (w_scale, h_scale)
            results['keep_ratio'] = self.keep_ratio
                
@TRANSFORMS.register_module()
class LinePackDetInputs(MMDET_PackDetInputs):
    '''
    packed_results.data_samples.gt_line_instances.line_points  #LineStringsOnImage, 一张图像上多个line标注，也可能为 LineStringsOnImage([],shape=img.shape)
    packed_results.data_samples.gt_line_instances.line_labels
    
    packed_results.data_samples.gt_instances.bboxes   #BaseBoxes, 一张图像上多个bbox标注，也可能为
    packed_results.data_samples.gt_instances.labels
    packed_results.data_samples.ignored_instances. (line_points/line_labels/bboxes/labels)
    packed_results.data_samples.proposals (InstanceData) (optional)
    packed_results.data_samples.gt_sem_seg (PixelData) (optional)
    packed_results.data_samples.metainfo (self.meta_keys)
    '''
    mapping_table = {
        'gt_bboxes': 'bboxes',
        'gt_bboxes_labels': 'labels',
        'gt_masks': 'masks',
    }
    #--------add by zhou
    line_mapping_table = {
        'gt_line_points': 'line_points', 
        'gt_line_labels': 'line_labels', 
    }
    #--------add by zhou

    # change based on MMDET_PackDetInputs
    def transform(self, results: dict) -> dict:
        """Method to pack the input data.

        Args:
            results (dict): Result dict from the data pipeline.

        Returns:
            dict:

            - 'inputs' (obj:`torch.Tensor`): The forward data of models.
            - 'data_sample' (obj:`DetDataSample`): The annotation info of the
                sample.
        """
        packed_results = dict()
        if 'img' in results:
            img = results['img']
            if len(img.shape) < 3:
                img = np.expand_dims(img, -1)
            # To improve the computational speed by by 3-5 times, apply:
            # If image is not contiguous, use
            # `numpy.transpose()` followed by `numpy.ascontiguousarray()`
            # If image is already contiguous, use
            # `torch.permute()` followed by `torch.contiguous()`
            # Refer to https://github.com/open-mmlab/mmdetection/pull/9533
            # for more details
            if not img.flags.c_contiguous:
                img = np.ascontiguousarray(img.transpose(2, 0, 1))
                img = to_tensor(img)
            else:
                img = to_tensor(img).permute(2, 0, 1).contiguous()

            packed_results['inputs'] = img

        data_sample = DetDataSample()
        
        def get_instance_data(mapping_table=self.mapping_table,flag='gt_ignore_flags'):
            instance_data = InstanceData()
            ignore_instance_data = InstanceData()
            if flag in results:
                valid_idx = np.where(results[flag] == 0)[0]
                ignore_idx = np.where(results[flag] == 1)[0]
            for key in mapping_table.keys():
                if key not in results:
                    continue
                if key == 'gt_masks' or isinstance(results[key], BaseBoxes):
                    if flag in results:
                        instance_data[
                            mapping_table[key]] = results[key][valid_idx]
                        ignore_instance_data[
                            mapping_table[key]] = results[key][ignore_idx]
                    else:
                        instance_data[mapping_table[key]] = results[key]
                elif isinstance(results[key], LineStringsOnImage): #--------------------add by zhou
                    if flag in results:
                        instance_data[
                            mapping_table[key]] = LineStringsOnImage([results[key].items[i] for i in valid_idx],shape=results[key].shape)
                        ignore_instance_data[
                            mapping_table[key]] = LineStringsOnImage([results[key].items[i] for i in ignore_idx],shape=results[key].shape)
                    else:
                        instance_data[mapping_table[key]] = results[key]
                else:
                    if flag in results:
                        instance_data[mapping_table[key]] = to_tensor(
                            results[key][valid_idx])
                        ignore_instance_data[mapping_table[key]] = to_tensor(
                            results[key][ignore_idx])
                    else:
                        instance_data[mapping_table[key]] = to_tensor(
                            results[key])
            return instance_data, ignore_instance_data
        instance_data, ignore_instance_data = get_instance_data(self.mapping_table,'gt_ignore_flags')    
        data_sample.gt_instances = instance_data
        data_sample.ignored_instances = ignore_instance_data
        #--------------------add by zhou
        instance_data, ignore_instance_data = get_instance_data(self.line_mapping_table,'gt_line_ignore_flags')  
        data_sample.gt_line_instances = instance_data
        data_sample.ignored_line_instances = ignore_instance_data
        #--------------------add by zhou

        if 'proposals' in results:
            proposals = InstanceData(
                bboxes=to_tensor(results['proposals']),
                scores=to_tensor(results['proposals_scores']))
            data_sample.proposals = proposals

        if 'gt_seg_map' in results:
            gt_sem_seg_data = dict(
                sem_seg=to_tensor(results['gt_seg_map'][None, ...].copy()))
            data_sample.gt_sem_seg = PixelData(**gt_sem_seg_data)

        img_meta = {}
        for key in self.meta_keys:
            assert key in results, f'`{key}` is not found in `results`, ' \
                f'the valid keys are {list(results)}.'
            img_meta[key] = results[key]

        data_sample.set_metainfo(img_meta)
        packed_results['data_samples'] = data_sample
        #---------------------zhou
        # from mmengine.visualization import Visualizer
        # visualizer = Visualizer.get_current_instance()
        # visualizer.add_datasample('result',packed_results['inputs'].numpy().transpose(1, 2, 0)[...,[2, 1, 0]], packed_results['data_samples'])
        #---------------------zhou  
        return packed_results
 
