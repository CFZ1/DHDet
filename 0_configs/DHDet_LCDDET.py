# ========================Frequently modified parameters======================
# ---------------------data related---------------------
import copy
data_root = '/root/0LCD240404/'  # Root path of data
work_dir = '/root/autodl-tmp/work_dirs'
train_data_prefix = 'images2_crop/'  # data_root+'unenhance/'
val_data_prefix = 'images2_crop/'  # data_root+'unenhance/'
train_ann_file = 'anns/LCDhalf_train.json'
val_ann_file = 'anns/LCDhalf_val.json'
test_ann_file = 'anns/LCDhalf_test.json'
load_from = None
save_dir = work_dir
vis_val_out_dir = 'val_pred_results' #=work_dir+vis_val_out_dir

class_name = ('TP','mura','sq') #('bubble','scratch','pinhole','tin_ash')
num_classes = len(class_name) # Number of classes for classification
line_classes = ('line',)
num_line_classes = len(line_classes) 
metainfo = dict(classes=class_name,line_classes=line_classes,bbox_include_line=False,palette=[(220, 20, 60), (119, 11, 32), (0, 0, 142), (0, 0, 230)])
dataset_type = 'LineCocoDataset'
img_scale = (2400, 2400)  # 3288,3288,width, height
backend_args = None

train_batch_size_per_gpu = 2
train_num_workers = 2 # Worker to pre-fetch data for each single GPU during training
persistent_workers = True # persistent_workers must be False if num_workers is 0
val_batch_size_per_gpu = 2
val_num_workers = 2 

# randomness = dict(seed=0)
# ---------------------optim---------------------
base_lr = 0.0001 # Base learning rate for optim_wrapper.
weight_decay = 0.0005
eta_min_ratio = 0.1
max_epochs = 60  # Maximum training epochs
save_epoch_intervals = 1 # Save model checkpoint and validation intervals in stage 1
max_keep_ckpts = 2 # The maximum checkpoints to keep.
Logger_interval = 100
randomness = dict(seed=0)
# ---------------------model---------------------
model_test_cfg = dict(
    multi_label=True, # The config of multi-label for multi-class prediction.
    nms_pre=30000,    # The number of boxes before NMS
    score_thr=0.0001,  # Threshold to filter out boxes.
    nms=dict(type='nms', iou_threshold=0.7),  # NMS type and threshold
    max_per_img=300)  # Max number of detections of each image

# ========================Possible modified parameters========================
# -----model related-----
loss_cls_weight = 1.0
loss_bbox_weight = 1.0
loss_bbox_weight2 = 1.0 
linecls_loss_weight=15

# Strides of multi-scale prior box
strides = [8, 16, 32]

norm_cfg = dict(type='BN', momentum=0.03, eps=0.001)  # Normalization config

# Single-scale training is recommended to be turned on, which can speed up training.
env_cfg = dict(cudnn_benchmark=False,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'))
# ===============================Unmodified in most cases====================
clc_order = 'default'
points_for_lossMetric = 6
linereg_loss_weight=0.1
loss_bbox_pre=dict(line_points_inter_method ='lineSegmentUni',
                    points_for_lossMetric = points_for_lossMetric,
                    inter_reg = None)
pred_n_points = 2
model = dict(
    type='coLineDetectorV4',
    HighFeaRefine=True,
    data_preprocessor=dict(
        type='Det_YOLOv5DetDataPreprocessor', #YOLOv5DetDataPreprocessor
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True,
        pad_size_divisor=1,
        line_fix_point=loss_bbox_pre),
    backbone=dict(
        type='mmdet.ResNet',
        depth=50,
        num_stages=4,
        out_indices=(1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=True,
        style='pytorch',
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50')),
    neck=dict(
        type='mmdet.FPN',
        in_channels=[512, 1024, 2048],
        out_channels=256,
        num_outs=3),
    bbox_head=dict(
        type='YOLOv8HeadFlexibleLoss', 
        head_module=dict(
            type='YOLOv8HeadModule',
            num_classes=num_classes,
            in_channels=[256, 256, 256], #256/widen_factor
            widen_factor=1.0,
            reg_max=1,
            norm_cfg=norm_cfg,
            act_cfg=dict(type='SiLU', inplace=True),
            featmap_strides=strides),
        prior_generator=dict(
            type='mmdet.MlvlPointGenerator', offset=0.5, strides=strides),
        bbox_coder=dict(type='DistancePointBBoxCoder'),
        loss_cls=dict(
            type='mmdet.CrossEntropyLoss',
            use_sigmoid=True,
            reduction='none',
            loss_weight=loss_cls_weight),
        loss_bbox=[dict(
            type='IoULoss',
            iou_mode='ciou',
            bbox_format='xyxy',
            reduction='sum',
            loss_weight=loss_bbox_weight,
            return_iou=False),
            dict(type='mmdet.L1Loss', reduction='sum', loss_weight=loss_bbox_weight2)],
        loss_dfl=None, 
        train_cfg=dict(
            assigner=dict(
                type='BatchTaskAlignedAssigner',
                num_classes=num_classes,
                # use_ciou=True,
                topk=13,
                alpha=1,
                beta=6,
                # eps=1e-9
            )),
        test_cfg=model_test_cfg),
    line_head=dict(
        type='coLineHeadv4',
        num_classes=num_line_classes,
        in_channels=256, # neck output
        embed_dims=256,
        num_feature_levels=3,
        num_queries=200,
        with_box_refine=True, 
        as_two_stage=True, 
        code_weights= [1.0 for i in range(points_for_lossMetric*2)],
        sync_cls_avg_factor=True,
        lineRefPts=pred_n_points,
        fineLength=False,
        correctAngle=False,
        correctAngleSigFirst=False,
        transformer=dict(
            decoder=dict(
                type='DeformableDetrTransformerMultiRefDecoder',
                num_layers=3,
                return_intermediate=True,
                layer_cfg=dict(
                    type='DeformableDetrTransformerDecoderLayerForRegist',
                    self_attn_cfg=dict( 
                        embed_dims=256,
                        num_heads=8,
                        dropout=0.1,
                        batch_first=True),
                    cross_attn_cfg=dict(  
                        type='MultiScaleDeformableMultiRefAttention',
                        embed_dims=256,
                        batch_first=True,
                        num_levels=3,
                        num_points=4*points_for_lossMetric),
                    ffn_cfg=dict(
                        embed_dims=256, feedforward_channels=1024, ffn_drop=0.1)),
                    post_norm_cfg=None)),
        positional_encoding=dict(type='mmdet.SinePositionalEncoding',num_feats=256//2,normalize=True,offset=-0.5),
        loss_bbox_pre=loss_bbox_pre,
        loss_cls=dict(
            type='mmdet.FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=linecls_loss_weight),
        loss_bbox=dict(type='L1OrderLoss', loss_weight=linereg_loss_weight, clc_order=clc_order),
        train_cfg=dict(
            assigner=dict(
                type='mmdet.HungarianAssigner',
                match_costs=[
                    dict(type='mmdet.FocalLossCost', weight=linecls_loss_weight),
                    dict(type='LaneL1Cost', weight=linereg_loss_weight, clc_order=clc_order)]),
                    ),
        ) 
)
# ===============================dataset====================
train_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='LineLoadAnnotations', with_bbox=True, with_line=True),
    dict(type='LineRandomFlip', prob=0.5, direction='horizontal'),
    dict(type='LineRandomFlip', prob=0.5, direction='vertical'),
    dict(type='ColorJitter', brightness=0.4, contrast=0.4, saturation=0.4),
    dict(type='LineResize', scale=img_scale, keep_ratio=False),
    dict(
        type='LinePackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape'))
]
test_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='LineLoadAnnotations', with_bbox=True, with_line=True),
    dict(type='LineResize', scale=img_scale, keep_ratio=False),
    dict(
        type='LinePackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor'))
]

train_dataloader = dict(
    batch_size=train_batch_size_per_gpu,
    num_workers=train_num_workers,
    persistent_workers=persistent_workers,
    pin_memory=False,
    sampler=dict(type='DefaultSampler', shuffle=True),
    collate_fn=dict(type='pseudo_and_yolov5_collate'),
    dataset=dict(
        type=dataset_type,
        metainfo=metainfo,
        data_root=data_root,
        data_prefix=dict(img=train_data_prefix),
        ann_file=train_ann_file,
        filter_cfg=dict(filter_empty_gt=False, min_size=32),
        pipeline=train_pipeline))
val_dataloader = dict(
    batch_size=val_batch_size_per_gpu,
    num_workers=val_num_workers,
    persistent_workers=persistent_workers,
    pin_memory=False,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        metainfo=metainfo,
        data_root=data_root,
        test_mode=True,
        data_prefix=dict(img=val_data_prefix),
        ann_file=val_ann_file,
        pipeline=test_pipeline))
test_dataloader = copy.deepcopy(val_dataloader)
test_dataloader['dataset']['ann_file'] = test_ann_file

val_evaluator = dict(type='LineCocoMetric',
                     ann_file=data_root + val_ann_file,
                     metric='bbox',
                     classwise=True,
                     line_pre=loss_bbox_pre,
                     box_num_cls=num_classes,
                     recordMetCha=True,
                     merged_preds_cfg=dict(merged_preds=True),#result combination of two halves
                     )
test_evaluator = copy.deepcopy(val_evaluator)
test_evaluator['ann_file'] = data_root + test_ann_file
# ===================================================
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=max_epochs, val_interval=save_epoch_intervals)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

param_scheduler = dict(type='CosineAnnealingLR', by_epoch=True, T_max=max_epochs,eta_min_ratio=eta_min_ratio) 
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(
        type='AdamW',
        lr=base_lr, 
        weight_decay=weight_decay),
    clip_grad=dict(max_norm=0.1, norm_type=2)
)
default_hooks = dict(
    checkpoint=dict(
        type='CheckpointHook',
        interval=save_epoch_intervals,
        save_best=['coco/bbox_mAP','coco/bbox_mAP_50'],
        max_keep_ckpts=max_keep_ckpts),
    logger=dict(type='LoggerHook', interval=Logger_interval), #and LogProcessor
    timer=dict(type='IterTimerHook'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='LineDetVisualizationHook',draw=False,interval=10, val_out_dir=vis_val_out_dir)) # or draw=False
# ========================default_runtime======================
default_scope = 'mmyolo'
vis_backends = [dict(type='LocalVisBackend')]
visualizer = dict(type='LineDetLocalVisualizer', vis_backends=vis_backends, save_dir=save_dir, name='visualizer')
log_processor = dict(type='LogProcessor', window_size=Logger_interval, by_epoch=True) #and LoggerHook

log_level = 'INFO'
resume = False











