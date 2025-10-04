# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os
import os.path as osp

from mmdet.engine.hooks.utils import trigger_visualization_hook
from mmdet.utils import setup_cache_size_limit_of_dynamo
from mmengine.config import Config, ConfigDict, DictAction
from mmengine.evaluator import DumpResults
from mmengine.runner import Runner

from mmyolo.registry import RUNNERS
from mmyolo.utils import is_metainfo_lower
import time
from mmengine.model import is_model_wrapper
from mmengine.logging import MMLogger
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
        print('hsdhfo',lineMin_global)
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
# TODO: support fuse_conv_bn
def parse_args():
    parser = argparse.ArgumentParser(
        description='MMYOLO test (and eval) a model')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument(
        '--work-dir',
        help='the directory to save the file containing evaluation metrics')
    parser.add_argument(
        '--out',
        type=str,
        help='output result file (must be a .pkl file) in pickle format')
    parser.add_argument(
        '--json-prefix',
        type=str,
        help='the prefix of the output json file without perform evaluation, '
        'which is useful when you want to format the result to a specific '
        'format and submit it to the test server')
    parser.add_argument(
        '--tta',
        action='store_true',
        help='Whether to use test time augmentation')
    parser.add_argument(
        '--show', action='store_true', help='show prediction results')
    parser.add_argument(
        '--deploy',
        action='store_true',
        help='Switch model to deployment mode')
    parser.add_argument(
        '--show-dir',
        help='directory where painted images will be saved. '
        'If specified, it will be automatically saved '
        'to the work_dir/timestamp/show_dir')
    parser.add_argument(
        '--wait-time', type=float, default=2, help='the interval of show (s)')
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
    cfg.work_dir = osp.join(cfg.work_dir,suffix)
    #--------------------------zhou
    cfg.load_from = args.checkpoint

    if args.show or args.show_dir:
        cfg = trigger_visualization_hook(cfg, args)

    if args.deploy:
        cfg.custom_hooks.append(dict(type='SwitchToDeployHook'))

    # add `format_only` and `outfile_prefix` into cfg
    if args.json_prefix is not None:
        cfg_json = {
            'test_evaluator.format_only': True,
            'test_evaluator.outfile_prefix': args.json_prefix
        }
        cfg.merge_from_dict(cfg_json)

    # Determine whether the custom metainfo fields are all lowercase
    is_metainfo_lower(cfg)

    if args.tta:
        #------------------------------zcf use_tta 20240916
        if 'tta_model' not in cfg:
            print("Cannot find ``tta_model`` in config.")
            cfg.tta_model = dict(
                type='mmdet.DetTTAModel',
                tta_cfg=dict(nms=dict(type='nms', iou_threshold=0.7), max_per_img=300))
        if 'tta_pipeline' not in cfg:
            print("Cannot find ``tta_pipeline`` in config.")
            original_pipeline = cfg.test_dataloader.dataset.pipeline
            resize_pipeline = [i for i in original_pipeline if 'Resize' in i['type']]
            tta_pipeline = []
            for item in original_pipeline:
                if 'Resize' not in item['type'] and 'PackDetInputs' not in item['type']:
                    tta_pipeline.append(item)
            tta_pipeline.append(
                dict(type='TestTimeAug',
                     transforms=[
                         [dict(type='LineRandomFlip', prob=1.),
                          dict(type='LineRandomFlip', prob=0.)],
                         [dict(type='PackDetInputs',
                               meta_keys=('img_id', 'img_path', 'ori_shape',
                                          'img_shape', 'scale_factor', 'flip', 'flip_direction'))]
                     ])
            )
            if resize_pipeline !=[]:
                tta_pipeline[-1]['transforms'].insert(0, resize_pipeline)
            cfg.tta_pipeline = tta_pipeline
        #------------------------------zcf use_tta 20240916
        assert 'tta_model' in cfg, 'Cannot find ``tta_model`` in config.' \
                                   " Can't use tta !"
        assert 'tta_pipeline' in cfg, 'Cannot find ``tta_pipeline`` ' \
                                      "in config. Can't use tta !"

        cfg.model = ConfigDict(**cfg.tta_model, module=cfg.model)
        test_data_cfg = cfg.test_dataloader.dataset
        while 'dataset' in test_data_cfg:
            test_data_cfg = test_data_cfg['dataset']

        # batch_shapes_cfg will force control the size of the output image,
        # it is not compatible with tta.
        if 'batch_shapes_cfg' in test_data_cfg:
            test_data_cfg.batch_shapes_cfg = None
        test_data_cfg.pipeline = cfg.tta_pipeline

    # build the runner from config
    if 'runner_type' not in cfg:
        # build the default runner
        runner = Runner.from_cfg(cfg)
    else:
        # build customized runner from the registry
        # if 'runner_type' is set in the cfg
        runner = RUNNERS.build(cfg)

    # add `DumpResults` dummy metric
    if args.out is not None:
        assert args.out.endswith(('.pkl', '.pickle')), \
            'The dump file must be a pkl file.'
        runner.test_evaluator.metrics.append(
            DumpResults(out_file_path=args.out))
    #------------------------------zcf use_tta 20240916
    if args.tta:
        runner.model.module.use_tta = True 
        runner.model.module.box_num_cls = cfg.val_evaluator.box_num_cls
    else:
        runner.model.use_tta = False 
    #------------------------------zcf use_tta 20240916
    # start testing
    runner.test()


if __name__ == '__main__':
    main()

