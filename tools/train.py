"""
****************Latest version 2025-04-30-19:00:00
Created on Fri Oct 25 21:34:34 2024

@author: zcf
"""

# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import logging
import os
import os.path as osp

from mmdet.utils import setup_cache_size_limit_of_dynamo
from mmengine.config import Config, DictAction
from mmengine.logging import print_log
from mmengine.runner import Runner

from mmyolo.registry import RUNNERS
from mmyolo.utils import is_metainfo_lower
import time
from mmengine.logging import MMLogger
import shutil
from mmengine.model import is_model_wrapper
# 根据数据集类型来修改全局变量 lineMin 的值, 只有修改全局变量的功能
def update_lineMin(dataset_type):
    try:
        from mmyolo.models.detectors import coLine_detectorv4 as coLine_detectorv4
    except: #因为有时候会做消融实验 
        from mmyolo.models.detectors import coLine_detectorv4_abla as coLine_detectorv4
    if dataset_type == 'LCD':
        from mmyolo.evaluation.metrics import line_coco_metric
        lineMin_global = 30
    elif dataset_type == 'solar':
        from mmyolo.evaluation.metrics import line_coco_metric_overLapCls as line_coco_metric
        lineMin_global = 10
    elif dataset_type == 'mobile':
        from mmyolo.evaluation.metrics import line_coco_metric
        lineMin_global = 30
    elif dataset_type == 'mobile_overLapCls':
        from mmyolo.evaluation.metrics import line_coco_metric_overLapCls as line_coco_metric
        lineMin_global = 30
    else:
        print("Unknown dataset type. Keeping the default value.")
    line_coco_metric.lineMin = lineMin_global
    # line_coco_metric.lineMin = lineMin_global
    coLine_detectorv4.lineMin = lineMin_global
    print(f'====>{dataset_type}===lineMin is changed to: ',lineMin_global)

def parse_args():
    parser = argparse.ArgumentParser(description='Train a detector')
    parser.add_argument('config', help='train config file path')
    parser.add_argument('--lr', type=float, default=None, help="learning rate for model")
    parser.add_argument('--max_epochs', type=float, default=None, help="max_epochs for model")
    parser.add_argument('--work-dir', help='the dir to save logs and models')
    parser.add_argument(
        '--amp',
        action='store_true',
        default=False,
        help='enable automatic-mixed-precision training')
    parser.add_argument(
        '--resume',
        nargs='?',
        type=str,
        const='auto',
        help='If specify checkpoint path, resume from it, while if not '
        'specify, try to auto resume from the latest checkpoint '
        'in the work directory.')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    # When using PyTorch version >= 2.0.0, the `torch.distributed.launch`
    # will pass the `--local-rank` parameter to `tools/train.py` instead
    # of `--local_rank`.
    parser.add_argument('--local_rank', '--local-rank', type=int, default=0)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    return args


def main():
    args = parse_args()

    # Reduce the number of repeated compilations and improve
    # training speed.
    setup_cache_size_limit_of_dynamo()

    # load config
    cfg = Config.fromfile(args.config)
    # replace the ${key} with the value of cfg.key
    # cfg = replace_cfg_vals(cfg)
    cfg.launcher = args.launcher
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)
    #------------new 241017------------
    logger: MMLogger = MMLogger.get_current_instance()
    if 'solar' in cfg.train_ann_file:
        dataset_type = 'solar'
        if 'val_evaluator' in cfg and '_overLapCls' not in cfg.val_evaluator.type:
            cfg.val_evaluator.type = cfg.val_evaluator.type.replace('LineCocoMetric','LineCocoMetric_overLapCls')
            logger.info('for solar dataset, 解决历史版本遗留问题: LineCocoMetric========>LineCocoMetric_overLapCls')
        if 'test_evaluator' in cfg and '_overLapCls' not in cfg.test_evaluator.type:
            cfg.test_evaluator.type = cfg.test_evaluator.type.replace('LineCocoMetric','LineCocoMetric_overLapCls')
            logger.info('for solar dataset, 解决历史版本遗留问题: LineCocoMetric========>LineCocoMetric_overLapCls')
    elif 'LCD' in cfg.train_ann_file:
        dataset_type = 'LCD'
    elif 'mobile' in cfg.train_ann_file:
        dataset_type = 'mobile'
        if 'val_evaluator' in cfg and '_overLapCls' in cfg.val_evaluator.type:
            dataset_type = 'mobile_overLapCls'
    update_lineMin(dataset_type)
    #------------new 241017------------
    # work_dir is determined in this priority: CLI > segment in file > filename
    if args.work_dir is not None:
        # update configs according to CLI args if args.work_dir is not None
        cfg.work_dir = args.work_dir
    elif cfg.get('work_dir', None) is None:
        # use config filename as default work_dir if cfg.work_dir is None
        cfg.work_dir = osp.join('./work_dirs',
                                osp.splitext(osp.basename(args.config))[0])
    #--------------------------zhou
    suffix = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    cfg.work_dir = osp.join(cfg.work_dir,f'{cfg.model.type}_'+suffix)
    if cfg.visualizer.get('save_dir', None) is not None:
        cfg.visualizer.save_dir = osp.join(cfg.visualizer.save_dir,suffix)
    if args.lr is not None:
        cfg.optim_wrapper.optimizer.lr = args.lr
        logger.warning('***************cfg.param_scheduler.lr not changed***************')   
        # if cfg.param_scheduler.type == 'OneCycleLR':
        #     cfg.param_scheduler.eta_max = args.lr
        #     print('cfg.param_scheduler.eta_max',cfg.param_scheduler.eta_max)
        print('cfg.optim_wrapper.optimizer.lr',cfg.optim_wrapper.optimizer.lr)
    if args.max_epochs is not None:
        cfg.train_cfg.max_epochs = args.max_epochs
        if cfg.param_scheduler.type in ['CosineAnnealingLR']:
            cfg.param_scheduler.T_max = args.max_epochs
        else:
            logger.warning('max_epochs as args: You should check cfg.param_scheduler')   
        print('cfg.train_cfg.max_epochs',cfg.train_cfg.max_epochs)
        print('cfg.param_scheduler.T_max',cfg.param_scheduler.T_max)
    #--------------------------zhou
    # enable automatic-mixed-precision training
    if args.amp is True:
        optim_wrapper = cfg.optim_wrapper.type
        if optim_wrapper == 'AmpOptimWrapper':
            print_log(
                'AMP training is already enabled in your config.',
                logger='current',
                level=logging.WARNING)
        else:
            assert optim_wrapper == 'OptimWrapper', (
                '`--amp` is only supported when the optimizer wrapper type is '
                f'`OptimWrapper` but got {optim_wrapper}.')
            cfg.optim_wrapper.type = 'AmpOptimWrapper'
            cfg.optim_wrapper.loss_scale = 'dynamic'

    # resume is determined in this priority: resume from > auto_resume
    if args.resume == 'auto':
        cfg.resume = True
        cfg.load_from = None
    elif args.resume is not None:
        cfg.resume = True
        cfg.load_from = args.resume

    # Determine whether the custom metainfo fields are all lowercase
    is_metainfo_lower(cfg)

    # build the runner from config
    if 'runner_type' not in cfg:
        # build the default runner
        runner = Runner.from_cfg(cfg)
    else:
        # build customized runner from the registry
        # if 'runner_type' is set in the cfg
        runner = RUNNERS.build(cfg)
    #--------------------------zhou
    str_nnn = '*******version 2025-07-16-20:16:00'
    
    # print(str_nnn)
    
    logger.info('\n' + str_nnn)
    #---------------------复制文件
    shutil.copy(args.config, cfg.work_dir)
    # 指定要复制的文件列表
    files_to_copy = [
        osp.join(os.path.dirname(__file__),'./train.py'),
        osp.join(os.path.dirname(__file__),'../mmyolo/models/detectors/coLine_detectorv4.py'),
        osp.join(os.path.dirname(__file__),'../mmyolo/models/dense_heads/line_head_v4.py'),
        osp.join(os.path.dirname(__file__),'../mmyolo/datasets/line_coco.py'),
    ]
    # 遍历文件列表，逐个复制文件
    for file_path in files_to_copy:
        if os.path.exists(file_path):  # 检查文件是否存在
            shutil.copy(file_path, cfg.work_dir)
        else:
            print(f"文件 {file_path} 不存在，跳过复制。")
    #---------------------复制文件
    # start training
    runner.train()


if __name__ == '__main__':
    main()
    
'''
yolov8:
            reg_pred[-1].bias.data[:] = 1.0  # box
            # cls (.01 objects, 80 classes, 640 img)
            cls_pred[-1].bias.data[:self.num_classes] = math.log(
                5 / self.num_classes / (640 / stride)**2)
            
            reg_pred[-1].bias.data[:] = 4.0  # box
            # cls (.01 objects, 80 classes, 640 img)
            cls_pred[-1].bias.data[:self.num_classes] = math.log(
                5 / self.num_classes / (2000 / stride)**2)
'''
