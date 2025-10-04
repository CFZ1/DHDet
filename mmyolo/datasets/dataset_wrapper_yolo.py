import copy
import logging
from typing import List, Sequence, Tuple, Union

import numpy as np

from mmengine.logging import print_log
from mmyolo.registry import DATASETS
from mmengine.dataset.base_dataset import BaseDataset, force_full_init, Compose
import random
from typing import Callable
import torch
from mmdet.structures.bbox import HorizontalBoxes

# change from mmengine/dataset/dataset_wrapper.py: ClassBalancedDataset
@DATASETS.register_module()
class DefectNorCatDataset:
    """
    Defect Image & Normal Image, stitch
    """

    def __init__(self,
                 dataset: Union[BaseDataset, dict],
                 defect_pipeline: List[Union[dict, Callable]] = [],
                 nor_pipeline: List[Union[dict, Callable]] = [],
                 lazy_init: bool = False):
        if isinstance(dataset, dict):
            self.dataset = DATASETS.build(dataset)
        elif isinstance(dataset, BaseDataset):
            self.dataset = dataset
        else:
            raise TypeError(
                'elements in datasets sequence should be config or '
                f'`BaseDataset` instance, but got {type(dataset)}')
        self._metainfo = self.dataset.metainfo
        # Build pipeline.
        self.defect_pipeline = Compose(defect_pipeline)
        self.nor_pipeline = Compose(nor_pipeline)
        self._fully_initialized = False
        if not lazy_init:
            self.full_init()

    @property
    def metainfo(self) -> dict:
        """Get the meta information of the repeated dataset.

        Returns:
            dict: The meta information of repeated dataset.
        """
        return copy.deepcopy(self._metainfo)

    def full_init(self):
        """Loop to ``full_init`` each dataset."""
        if self._fully_initialized:
            return

        self.dataset.full_init()
        self._set_group_flag() #---------------new

        self._fully_initialized = True
        
    def _set_group_flag(self):
        """Set flag according to
        self.flag=0: normal img; self.flag=1: defetive img
        self.flag_normal
        self.flag_defetive
        """
        self.flag_normal, self.flag_defetive = [], []
        for idx in range(len(self.dataset)):
            cat_ids = set(self.dataset.get_cat_ids(idx))
            if len(set(cat_ids))==0:
                self.flag_normal.append(idx)
            else:
                self.flag_defetive.append(idx) 

        self.normal_mapping = {i: idx for i, idx in enumerate(self.flag_normal)}
        self.defetive_mapping = {i: idx for i, idx in enumerate(self.flag_defetive)}
        # self.flag = np.zeros(len(self.dataset), dtype=np.uint8)
        # for idx in range(len(self.dataset)):
        #     cat_ids = set(self.dataset.get_cat_ids(idx))
        #     if len(set(cat_ids))!=0:
        #         self.flag[idx] = 1 
    @force_full_init
    def get_cat_ids(self, idx: int) -> List[int]:
        """Get category ids of class balanced dataset by index.

        Args:
            idx (int): Index of data.

        Returns:
            List[int]: All categories in the image of specified index.
        """
        sample_idx = self._get_ori_dataset_idx(idx)
        return self.dataset.get_cat_ids(sample_idx)

    @force_full_init
    def get_data_info(self, idx: int) -> dict:
        """Get annotation by index.

        Args:
            idx (int): Global index of ``ConcatDataset``.

        Returns:
            dict: The idx-th annotation of the dataset.
        """
        sample_idx = self._get_ori_dataset_idx(idx)
        return self.dataset.get_data_info(sample_idx)
    @force_full_init
    def _get_ori_dataset_idx(self, idx: int) -> int:
        """Convert global index to local index.

        Args:
            idx (int): Global index of ``RepeatDataset``.

        Returns:
            int: Local index of data.
        """
        return self.defetive_mapping[idx]
    
    def __getitem__(self, idx):
        if not self._fully_initialized:
            print_log(
                'Please call `full_init` method manually to accelerate '
                'the speed.',
                logger='current',
                level=logging.WARNING)
            self.full_init()

        ori_index = self._get_ori_dataset_idx(idx)
        self.dataset.pipeline = self.defect_pipeline
        defect_img = self.dataset[ori_index]
        
        nor_id = self.normal_mapping[random.randint(0,len(self.flag_normal)-1)]
        self.dataset.pipeline = self.nor_pipeline
        nor_img = self.dataset[nor_id]
        
        cur_dir = np.random.choice(['left','right'], p=[0.5,0.5])
        # Concatenate images horizontally based on the chosen direction
        if cur_dir == 'left':
            new_img = torch.cat((defect_img['inputs'], nor_img['inputs']), dim=2)
            # Adjust bounding box coordinates for the defect image (keep the same y-coordinates)
            # Since defect is on the left, x-coordinates remain the same
            bboxes = defect_img['data_samples'].gt_instances.bboxes.clone()
            
        else:  # if the defect image is on the right
            new_img = torch.cat((nor_img['inputs'], defect_img['inputs']), dim=2)
            # Adjust bounding box coordinates for the defect image
            # Shift x-coordinates by the width of the normal image (1024 in this example)
            shift_x = nor_img['data_samples'].img_shape[1]
            bboxes = defect_img['data_samples'].gt_instances.bboxes.tensor.clone()
            bboxes[:, 0] += shift_x  # Shift x coordinates by the width of the normal image
            bboxes[:, 2] += shift_x  # Shift x coordinates by the width of the normal image
            bboxes = HorizontalBoxes(bboxes)  # Update bounding boxes
        # Create a new data sample for the combined image
        new_data_sample = defect_img['data_samples'].clone()
        new_data_sample.set_metainfo({'img_shape':tuple(new_img.shape[-2:])}) # Update shape for the concatenated image
        new_data_sample.gt_instances.bboxes = bboxes
        return {
                'inputs': new_img,
                'data_samples': new_data_sample
            }

    @force_full_init
    def __len__(self):
        return len(self.flag_defetive) #---------------new

    def get_subset_(self, indices: Union[List[int], int]) -> None:
        """Not supported in ``ClassBalancedDataset`` for the ambiguous meaning
        of sub-dataset."""
        raise NotImplementedError(
            '`ClassBalancedDataset` dose not support `get_subset` and '
            '`get_subset_` interfaces because this will lead to ambiguous '
            'implementation of some methods. If you want to use `get_subset` '
            'or `get_subset_` interfaces, please use them in the wrapped '
            'dataset first and then use `ClassBalancedDataset`.')

    def get_subset(self, indices: Union[List[int], int]) -> 'BaseDataset':
        """Not supported in ``ClassBalancedDataset`` for the ambiguous meaning
        of sub-dataset."""
        raise NotImplementedError(
            '`ClassBalancedDataset` dose not support `get_subset` and '
            '`get_subset_` interfaces because this will lead to ambiguous '
            'implementation of some methods. If you want to use `get_subset` '
            'or `get_subset_` interfaces, please use them in the wrapped '
            'dataset first and then use `ClassBalancedDataset`.')
        
        
        
