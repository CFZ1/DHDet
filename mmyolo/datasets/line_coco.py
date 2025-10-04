"""
Latest version 2025-04-04-16:00:00 add line_splitlargeother+img_scale+get_trend_bbox
Created on Tue Jan 16 20:43:45 2024

@author: zcf
"""

import copy
import os.path as osp
from typing import List, Union

from mmengine.fileio import get_local_path

from mmyolo.registry import DATASETS
from mmdet.datasets.api_wrappers import COCO
from mmdet.datasets import BaseDetDataset
def get_trend_bbox(points):
    # 计算首尾点的方向作为整体走势
    start, end = points[0], points[-1]
    dx = end[0] - start[0]
    dy = end[1] - start[1]
    
    # 计算边界
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    
    # 根据整体走势选择对角线
    if dx * dy > 0:  # 走势为左上到右下
        return [[min_x, min_y], [max_x, max_y]], (max_y-min_y)*(max_x-min_x)
    else:  # 走势为左下到右上
        return [[min_x, max_y], [max_x, min_y]], (max_y-min_y)*(max_x-min_x)
# 在CocoDataset基础上修改，不继承是因为不想嵌套太多: mmdetection/mmdet/datasets/coco.py
@DATASETS.register_module()
class LineCocoDataset(BaseDetDataset):
    """Dataset for COCO."""

    METAINFO = {
        'classes': ('TP','mura','sq'),
        'line_classes':('line',),
        'bbox_include_line': False,
        # palette is a list of color tuples, which is used for visualization.
        'palette':
        [(220, 20, 60), (119, 11, 32), (0, 0, 142), (0, 0, 230)],
        'line_splitlargeother': False,
        'img_scale': (2400, 2400),
        'get_trend_bbox': False
    }
    '''
    目前用不到的参数：get_trend_bbox;
    line_splitlargeother=True：表示将line类型的, ann['area']>(96/2400*5120)**2)保存给Line head;
    bbox_include_line=False：表示给line head标注不给box head；=True表示这个标注既给line head，又给box head;
    因为多个图像标注保存成一个json的时候，"line","linestrip"均保存成annotation["line"]，每个标注（包括annotation["line"]）都保存了annotation["bbox"]
    '''
    COCOAPI = COCO
    # ann_id is unique in coco dataset.
    ANN_ID_UNIQUE = True
    # shape_type =['rectangle','line']
    #line类型的标注，我将短边的长度设置为30，形成了bbox，实际上不是bbox，如果bbox_include_line=True and in METAINFO['classes']，将它加入gt bbox训练，否则不用来训练

    def load_data_list(self) -> List[dict]:
        """Load annotations from an annotation file named as ``self.ann_file``

        Returns:
            List[dict]: A list of annotation.
        """ 
        self.ann_file_name = self.ann_file.split('/')[-1]
        if self.metainfo['line_splitlargeother']:
            if 'mobile' in self.ann_file:
                img_ori_size = (5120,5120) 
            else:
                print('error! not implemented!!')
            self.split_area = (96/self.metainfo['img_scale'][0]*img_ori_size[0])*(96/self.metainfo['img_scale'][1]*img_ori_size[1])
        # noqa: E501
        with get_local_path(
                self.ann_file, backend_args=self.backend_args) as local_path:
            self.coco = self.COCOAPI(local_path)
        # The order of returned `cat_ids` will not
        # change with the order of the `classes`
        self.cat_ids = self.coco.get_cat_ids(
            cat_names=self.metainfo['classes'])
        self.cat2label = {cat_id: i for i, cat_id in enumerate(self.cat_ids)}
        
        self.cat_ids_line = self.coco.get_cat_ids(
            cat_names=self.metainfo['line_classes'])
        self.cat2label_line = {cat_id: i for i, cat_id in enumerate(self.cat_ids_line)}
        
        self.cat_img_map = copy.deepcopy(self.coco.cat_img_map)

        img_ids = self.coco.get_img_ids()
        data_list = []
        total_ann_ids = []
        for img_id in img_ids:
            raw_img_info = self.coco.load_imgs([img_id])[0]
            raw_img_info['img_id'] = img_id

            ann_ids = self.coco.get_ann_ids(img_ids=[img_id])
            raw_ann_info = self.coco.load_anns(ann_ids)
            total_ann_ids.extend(ann_ids)

            parsed_data_info = self.parse_data_info({
                'raw_ann_info':
                raw_ann_info,
                'raw_img_info':
                raw_img_info
            })
            data_list.append(parsed_data_info)
        if self.ANN_ID_UNIQUE:
            assert len(set(total_ann_ids)) == len(
                total_ann_ids
            ), f"Annotation ids in '{self.ann_file}' are not unique!"

        del self.coco

        return data_list

    def parse_data_info(self, raw_data_info: dict) -> Union[dict, List[dict]]:
        """Parse raw annotation to target format.

        Args:
            raw_data_info (dict): Raw data information load from ``ann_file``

        Returns:
            Union[dict, List[dict]]: Parsed annotation.
        """
        img_info = raw_data_info['raw_img_info']
        ann_info = raw_data_info['raw_ann_info']

        data_info = {}

        # TODO: need to change data_prefix['img'] to data_prefix['img_path']
        img_path = osp.join(self.data_prefix['img'], img_info['file_name'])
        if self.data_prefix.get('seg', None):
            seg_map_path = osp.join(
                self.data_prefix['seg'],
                img_info['file_name'].rsplit('.', 1)[0] + self.seg_map_suffix)
        else:
            seg_map_path = None
        data_info['img_path'] = img_path
        data_info['img_id'] = img_info['img_id']
        data_info['seg_map_path'] = seg_map_path
        data_info['height'] = img_info['height']
        data_info['width'] = img_info['width']

        instances = []
        for i, ann in enumerate(ann_info):
            instance = {}

            if ann.get('ignore', False):
                continue
            x1, y1, w, h = ann['bbox']
            inter_w = max(0, min(x1 + w, img_info['width']) - max(x1, 0))
            inter_h = max(0, min(y1 + h, img_info['height']) - max(y1, 0))
            if inter_w * inter_h == 0:
                continue
            if ann['area'] <= 0 or w < 1 or h < 1:
                continue
            if (ann['category_id'] not in self.cat_ids) and (ann['category_id'] not in self.cat_ids_line):
                continue
            bbox = [x1, y1, x1 + w, y1 + h]

            if ann.get('iscrowd', False):
                instance['ignore_flag'] = 1
            else:
                instance['ignore_flag'] = 0
                
            if ann.get('segmentation', None):
                instance['mask'] = ann['segmentation']
            # 统一一下模型(coLine_detector、line_coco)、指标评价(line_coco_metric)、数据集(line_coco)中的称呼
            if ann.get('line', None) and ann['category_id'] in self.cat_ids_line:
                # _, area_here = get_trend_bbox(ann['line'])
                if 'mobile' not in self.ann_file_name:
                    instance['line_points'] = ann['line']
                    instance['line_labels'] = self.cat2label_line[ann['category_id']]
                    instance['bbox_label_v2'] = len(self.cat2label) #---------240917 for ClassBalancedDataset
                    
                    if not self.metainfo['bbox_include_line']:
                        instances.append(instance) # opt 1
                        continue
                else:
                    if (self.metainfo['line_splitlargeother']) or (not self.metainfo['line_splitlargeother']):
                    # if (self.metainfo['line_splitlargeother'] and ann['area']>self.split_area) or (not self.metainfo['line_splitlargeother']):
                        if self.metainfo['get_trend_bbox']: 
                            instance['line_points'],_ = get_trend_bbox(ann['line']) # ann['line'] #[[x1,y1],[x2,y2]] #[[[1084.0, 268.0], [1083.0, 2990.0]], [[1092.0, 267.0], [1092.0, 2991.0]], [[1101.0, 268.0], [1099.0, 2990.0]], [[1108.0, 268.0], [1107.0, 2987.0]]] 因为多条线汇聚成一个box
                        else:
                            instance['line_points'] = ann['line']
                        instance['line_labels'] = self.cat2label_line[ann['category_id']]
                        instance['bbox_label_v2'] = len(self.cat2label) #---------240917 for ClassBalancedDataset
                        
                        if (not self.metainfo['bbox_include_line']) and (ann['area']>self.split_area):
                        # if not self.metainfo['bbox_include_line']:
                            instances.append(instance) # opt 1
                            continue
            if ann.get('bbox', None) and ann['category_id'] in self.cat_ids:
                instance['bbox'] = bbox
                instance['bbox_label'] = self.cat2label[ann['category_id']]

            instances.append(instance) # opt 2
        # 不能在这里修改，因为coco_metric的gt直接调用的是COCO，因此直接修改标注
        # if self.metainfo['filterLineBoxByIoU']:
        #     # 计算每对边界框的IoU并存储到数组
        #     obj_xx = torch.tensor([i['bbox'] for i in instances if i['category_id'] in self.cat_ids_line])
        #     if len(obj_xx)>0:
        #         obj_xx[:,2] = obj_xx[:,0]+obj_xx[:,2]
        #         obj_xx[:,3] = obj_xx[:,1]+obj_xx[:,3] #xywh-->xyxy
        #         iou_torch = calculate_iou(obj_xx, obj_xx)
        #         if torch.any(iou_torch > line_merge_iou_threshold):
        #             # 使用networkx找出所有应该合并的边界框组
        #             row, col = torch.nonzero(iou_torch > line_merge_iou_threshold, as_tuple=True)
        #             edges = list(zip(row.tolist(), col.tolist()))
        #             G = nx.Graph()
        #             G.add_edges_from(edges)
        #             # 找出所有连接的组件
        #             components = list(nx.connected_components(G))
        #             # 将IOU大于line_merge_iou_threshold的Line box合并给第一个，其他都置为None，然后移除
        #             for i in components:
        #                 input_bboxs = [instances[j]['line'] for j in i]
        #                 instances[min(i)]['bbox'] = merge_bboxs(input_bboxs)
        #                 for kk in list(i-{min(i)}):
        #                     instances[kk]['bbox'] = None
        #         # 在完成所有的修改之后，遍历并移除bbox为None的标注
        #         instances = [ann for ann in instances if ann['bbox'] is not None]
        data_info['instances'] = instances
        return data_info

    def filter_data(self) -> List[dict]:
        """Filter annotations according to filter_cfg.

        Returns:
            List[dict]: Filtered results.
        """
        if self.test_mode:
            return self.data_list

        if self.filter_cfg is None:
            return self.data_list

        filter_empty_gt = self.filter_cfg.get('filter_empty_gt', False)
        min_size = self.filter_cfg.get('min_size', 0)

        # obtain images that contain annotation
        ids_with_ann = set(data_info['img_id'] for data_info in self.data_list)
        # obtain images that contain annotations of the required categories
        ids_in_cat = set()
        for i, class_id in enumerate(self.cat_ids+self.cat_ids_line): # change by zhou
            ids_in_cat |= set(self.cat_img_map[class_id])
        # merge the image id sets of the two conditions and use the merged set
        # to filter out images if self.filter_empty_gt=True
        ids_in_cat &= ids_with_ann

        valid_data_infos = []
        for i, data_info in enumerate(self.data_list):
            img_id = data_info['img_id']
            width = data_info['width']
            height = data_info['height']
            if filter_empty_gt and img_id not in ids_in_cat:
                continue
            if min(width, height) >= min_size:
                valid_data_infos.append(data_info)

        return valid_data_infos

# import torch
# import numpy as np
# import networkx as nx
# line_min = 30
# line_merge_iou_threshold = 0.5
# # 多对多, 定义一个函数来计算两个边框的交并比
# def calculate_iou(boxes1, boxes2):
#     # 扩展boxes1形状为[11,1,3], 扩展boxes2形状为[1,11,3]
#     boxes1 = boxes1.unsqueeze(1)
#     boxes2 = boxes2.unsqueeze(0)

#     # 计算交叉区域的坐标
#     x1 = torch.maximum(boxes1[:, :, 0], boxes2[:, :, 0])
#     y1 = torch.maximum(boxes1[:, :, 1], boxes2[:, :, 1])
#     x2 = torch.minimum(boxes1[:, :, 2], boxes2[:, :, 2])
#     y2 = torch.minimum(boxes1[:, :, 3], boxes2[:, :, 3])

#     # 计算交叉区域的面积
#     intersection_area = torch.maximum(x2 - x1, torch.zeros_like(x2)) * torch.maximum(y2 - y1, torch.zeros_like(y2))

#     # 计算每对边框的面积
#     area_boxes1 = (boxes1[:, :, 2] - boxes1[:, :, 0]) * (boxes1[:, :, 3] - boxes1[:, :, 1])
#     area_boxes2 = (boxes2[:, :, 2] - boxes2[:, :, 0]) * (boxes2[:, :, 3] - boxes2[:, :, 1])

#     # 计算交并比(返回形状为[11,11]的IoU矩阵)
#     iou = intersection_area / (area_boxes1 + area_boxes2 - intersection_area)
#     # 将对角线元素置为0
#     iou.fill_diagonal_(0)
#     # 将对角线and上半对角角元素置为0元素置为0
#     # iou.tril_(diagonal=-1)
#     return iou

# def merge_bboxs(input_bboxs):
#     # 转换为NumPy数组
#     points_np = np.array(input_bboxs)
#     # 寻找最小的x（x1），最小的y（y1），最大的x（x2），最大的y（y2）
#     x_min = np.min(points_np[:, :, 0])
#     y_min = np.min(points_np[:, :, 1])
#     x_max = np.max(points_np[:, :, 0])
#     y_max = np.max(points_np[:, :, 1])
#     if (y_max-y_min)<line_min:
#         y_max,y_min = (y_max+y_min)/2.+line_min/2.,(y_max+y_min)/2.-line_min/2.
#     if (x_max-x_min)<line_min:
#         x_max,x_min = (x_max+x_min)/2.+line_min/2.,(x_max+x_min)/2.-line_min/2.  
#     print(x_min,x_max,y_min,y_max)
#     return [x_min,y_min,x_max-x_min,y_max-y_min]  
