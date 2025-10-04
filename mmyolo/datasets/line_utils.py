
# Copyright (c) OpenMMLab. All rights reserved.
from typing import Sequence
from mmengine.dataset import COLLATE_FUNCTIONS
from mmyolo.datasets import yolov5_collate
from mmengine.dataset import pseudo_collate
# copy from mmyolo/datasets/utils.py:yolov5_collate and 
# 因为训练的时候，yolov5和其他的接口不一样
@COLLATE_FUNCTIONS.register_module()
def pseudo_and_yolov5_collate(data_batch: Sequence,
                   use_ms_training: bool = False) -> dict:
    """Rewrite collate_fn to get faster training speed.

    Args:
       data_batch (Sequence): Batch of data.
       use_ms_training (bool): Whether to use multi-scale training.
    """
    collated_results = pseudo_collate(data_batch)
    
    collated_results['data_samples_yolov5'] = yolov5_collate(data_batch,use_ms_training)['data_samples']
    
    return collated_results #['inputs'],['data_samples'],['data_samples']