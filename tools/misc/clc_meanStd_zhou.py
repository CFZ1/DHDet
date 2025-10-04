"""
Created on Mon Jan 15 09:39:03 2024

@author: zcf
"""

import numpy as np
from PIL import Image
import os
import torch
from tqdm import tqdm
from mmengine.runner import Runner
def get_resolution(image_path):
    with Image.open(image_path) as img:
        width, height = img.size
    return (width, height)

def count_resolutions(image_paths):
    resolution_count = {}

    for path in tqdm(image_paths):
        resolution = get_resolution(path)

        # 将分辨率添加到字典中，如果已存在，则增加计数
        if resolution in resolution_count:
            resolution_count[resolution] += 1
        else:
            resolution_count[resolution] = 1

    return resolution_count

#get_mean_and_std: copy from https://github.com/kuangliu/pytorch-cifar/tree/master/utils.py
def get_mean_and_std(dataloader):
    '''Compute the mean and std value of dataset.'''
    if not isinstance(dataloader, torch.utils.data.DataLoader):
        dataloader = torch.utils.data.DataLoader(dataloader, batch_size=1, shuffle=True, num_workers=0)
    mean = torch.zeros(3)
    std = torch.zeros(3)
    print('==> Computing mean and std..')
    for data_batch in tqdm(dataloader):
        inputs = data_batch['inputs'][0].float()
        for i in range(3):
            mean[i] += inputs[i,:,:].mean()
            std[i] += inputs[i,:,:].std()
    mean.div_(len(dataloader))
    std.div_(len(dataloader))
    return mean, std

dataset_name ='LCD'
if dataset_name =='LCD':
    data_root = '/media/zcf/Elements/dataset/mobile_screen/0LCDMobileScreen/0LCD231201' 
    train_data_prefix = 'images/'  # data_root+'unenhance/'
    train_ann_file = 'coco1030_addline/LCDhalf_trainval.json' #data_root+'data_train0704.json'
    img_scale = (2400, 2400)
    train_file = os.path.join(data_root,'coco1030_addline/LCDhalf_trainval.json')
    class_name = ('TP','line','mura','sq') #('bubble','scratch','pinhole','tin_ash')
    metainfo = dict(classes=class_name, palette=[(220, 20, 60), (119, 11, 32), (0, 0, 142), (0, 0, 230)])
if isinstance(img_scale,int):
    img_scale = (img_scale,img_scale) 
backend_args = None  
train_batch_size_per_gpu=1 
train_num_workers=0
persistent_workers =False
dataset_type = 'mmdet.CocoDataset'
train_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='mmdet.RandomFlip', prob=0.5, direction='horizontal'),
    dict(type='mmdet.RandomFlip', prob=0.5, direction='vertical'),
    dict(type='mmdet.Resize', scale=img_scale, keep_ratio=False),
    dict(
        type='mmdet.PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape', 'flip',
                   'flip_direction'))
]

train_dataloader = dict(
    batch_size=train_batch_size_per_gpu,
    num_workers=train_num_workers,
    persistent_workers=persistent_workers,
    pin_memory=False,
    sampler=dict(type='DefaultSampler', shuffle=True),
    # collate_fn=dict(type='mmyolo.yolov5_collate'),
    dataset=dict(
        type=dataset_type,
        metainfo=metainfo,
        data_root=data_root,
        data_prefix=dict(img=train_data_prefix),
        ann_file=train_ann_file,
        filter_cfg=dict(filter_empty_gt=False, min_size=32),
        pipeline=train_pipeline))            

dataloader = Runner.build_dataloader(train_dataloader)
mean, std = get_mean_and_std(dataloader)