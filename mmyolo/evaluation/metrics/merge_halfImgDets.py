# """
# Created on Thu Apr 11 15:35:47 2024

# @author: zcf
# """
# #统一一下和图像110交叠区域有重叠的边界框的长和宽
# # from mmdet.datasets.api_wrappers import COCO
# # import numpy as np
# # from tqdm import tqdm
# # right_start = 6450-3280
# # left_end = 3280

# # data_pre = '/media/zcf/Elements/dataset/mobile_screen/0LCDMobileScreen/0LCD231201/0LCD240316/'
# # fullImgAnn = data_pre+'LCDfull_all_20240404_ReDup.json'
# # full_coco = COCO(fullImgAnn)
# # full_AnnIds = full_coco.getAnnIds()
# # # 初始化字典，用于统计每个类别的长度和宽度
# # category_stats = {}
# # for i in tqdm(full_AnnIds):
# #     anns = full_coco.load_anns(ids=i)[0]
# #     obj_xx = np.array(anns['bbox'])
# #     obj_xx[2] = obj_xx[0]+obj_xx[2]
# #     obj_xx[3] = obj_xx[1]+obj_xx[3] #xywh-->xyxy
# #     if obj_xx[2]> right_start and obj_xx[0]< left_end:
# #         # 更新统计字典
# #         category_id = anns['category_id']
# #         if category_id not in category_stats:
# #             category_stats[category_id] = {'widths': [], 'heights': [], 'ann_id': [], 'Incomplete': []}
# #         category_stats[category_id]['widths'].append(anns['bbox'][2])
# #         category_stats[category_id]['heights'].append(anns['bbox'][3])
# #         category_stats[category_id]['ann_id'].append(i)
# #         Incomplete = obj_xx[0]< right_start and obj_xx[2]> left_end
# #         category_stats[category_id]['Incomplete'].append(Incomplete)
# # # 对字典的 key 进行排序
# # sorted_category_ids = sorted(category_stats.keys())
# # # 按照排序后的顺序打印统计结果
# # for category_id in sorted_category_ids:
# #     stats = category_stats[category_id]
# #     print(f"Category ID: {full_coco.load_cats(category_id)[0]['name']}")
# #     print(f"Widths: max={max(stats['widths'])}, min={min(stats['widths'])}")
# #     print(f"Heights: max={max(stats['heights'])}, min={min(stats['heights'])}")
# #     widths = np.array(stats['widths'])
# #     heights = np.array(stats['heights'])
# #     print(f"不会在任何一侧出现完整的标注={sum(stats['Incomplete'])}; count(width>110)={sum(widths>110)}; num_anns={len(stats['widths'])}")
# #     print()
 

# from mmdet.datasets.api_wrappers import COCO
# import numpy as np
# from tqdm import tqdm
# import torch
# right_start = 6450-3280
# left_end = 3280
# Merge_score_thr = 0.3
# data_pre = '/media/zcf/Elements/dataset/mobile_screen/0LCDMobileScreen/0LCD231201/0LCD240316/'
# fullImgAnn = data_pre+'LCDfull_all_20240404_ReDup.json'
# halfImgAnn = data_pre+'LCDhalf_all_20240404_ReDup.json'

# full_coco = COCO(fullImgAnn)
# half_coco = COCO(halfImgAnn)
# # results tuple(dict,dict,dict,dict...) #dict_keys(['img_id', 'bboxes', 'scores', 'labels', 'line_points', 'line_scores', 'line_labels'])

# half_ImgIds = half_coco.getImgIds()
# half_coco.get_cat_ids(cat_names=('TP','line','mura','sq'))
# results = []
# for i in tqdm(half_ImgIds):
#     ann_ids = half_coco.get_ann_ids(img_ids=i)
#     anns = half_coco.load_anns(ids=ann_ids)
#     obj_xx = np.array([i['bbox'] for i in anns]).reshape(-1, 4)
#     obj_xx[:,2] = obj_xx[:,0]+obj_xx[:,2]
#     obj_xx[:,3] = obj_xx[:,1]+obj_xx[:,3] #xywh-->xyxy
#     scores = np.ones(obj_xx.shape[0]).astype(float)
#     result_per = dict()
#     result_per['img_id'] = i
#     result_per['bboxes'] = obj_xx
#     result_per['scores'] = scores
#     result_per['labels'] = np.array([i['category_id'] for i in anns])
#     results.append(result_per)
# results = tuple(results)
# #键是file_name,值是对应结果在results列表中的索引。     
# file_name_indices = { half_coco.loadImgs(ids=[result['img_id']])[0]['file_name']: idx for idx, result in enumerate(results) }

# def merge_detections(left_preds,right_preds,merge_score_thr=0.3,merge_iou_thr=0.7):
#     """
#     合并两张图像的检测结果
#     left_preds: 左图像的检测框坐标+分数+类别 [n1, 4+1+1]
#     right_preds: 右图像的检测框坐标+分数+类别 [n2, 4+1+1]
#     merge_score_thr: 合并时使用的分数阈值
#     """
#     right_start = 6450 - 3280
#     left_end = 3280
#     from mmcv.ops import batched_nms
#     def to_tensor(arr):
#         return torch.from_numpy(arr) if isinstance(arr, np.ndarray) else arr
#     left_preds = to_tensor(left_preds)
#     right_preds = to_tensor(right_preds)
#     # 过滤低分数的检测框
#     left_keep = left_preds[:,4] >= merge_score_thr
#     right_keep = right_preds[:,4] >= merge_score_thr
#     # 找到落在交叠区域内部的检测框
#     left_in_overlap = torch.logical_and(left_preds[:, 2] > right_start, left_keep)
#     right_in_overlap = torch.logical_and(right_preds[:, 0] < left_end, right_keep)
#     labels_i = set(left_preds[:,-1][left_in_overlap].tolist()) & set(right_preds[:,-1][right_in_overlap].tolist())
#     labels_u = set(left_preds[:,-1][left_in_overlap].tolist()) | set(right_preds[:,-1][right_in_overlap].tolist())
#     # 不需要过滤，直接返回
#     if sum(left_in_overlap) == 0 or sum(right_in_overlap) == 0 or len(labels_i)==0:
#         all_preds = torch.cat([left_preds, right_preds], dim=0)
#         # 按照预测分数从大到小排序
#         return all_preds[torch.argsort(-all_preds[:,4])]
#     # 处理不同类别的检测框
#     # 跨越交叠区域的检测框
#     left_in_overlap_large = torch.logical_and(left_preds[:, 0] < right_start, left_in_overlap)
#     right_in_overlap_large = torch.logical_and(right_preds[:, 2] > left_end, right_in_overlap)
#     if sum(left_in_overlap_large) == 0 and sum(right_in_overlap_large) == 0:
#         deal_preds = torch.cat([left_preds[left_in_overlap], right_preds[right_in_overlap]], dim=0)
#     else:
#         deal_preds = [] 
#         for label in labels_u:
#             left_deal_preds = left_preds[torch.logical_and(left_in_overlap, left_preds[:,-1] == label)]
#             right_deal_preds = right_preds[torch.logical_and(right_in_overlap, right_preds[:,-1] == label)]
#             if label not in labels_i:
#                 deal_preds.append(torch.cat([left_deal_preds, right_deal_preds], dim=0))
#                 continue
#             left_overBor = left_in_overlap_large[torch.logical_and(left_in_overlap, left_preds[:,-1] == label)]
#             right_overBor = right_in_overlap_large[torch.logical_and(right_in_overlap, right_preds[:,-1] == label)]     
#             if sum(left_overBor) == 0 and sum(right_overBor) == 0:
#                 deal_preds.append(torch.cat([left_deal_preds, right_deal_preds], dim=0))
#                 continue
#             # 按照预测分数从大到小排序
#             left_deal_preds = left_deal_preds[torch.argsort(-left_deal_preds[:, 4])]
#             right_deal_preds = right_deal_preds[torch.argsort(-right_deal_preds[:, 4])]
#             # 计算交并比 IOU1 = 相交部分 / 在交叠区域的合并部分
#             ious = bbox_overimg_iou(left_deal_preds[:,:4],right_deal_preds[:,:4],right_start,left_end) 
#             # 处理被截断的预测框
#             def merge_predictions(preds, other_preds, ious, iou_threshold):
#                 """
#                 根据给定的IOU阈值合并预测。
#                 :param preds: 主预测张量 [N, 6], 其中前四列是坐标，第五列是分数，第六列是标签。
#                 :param other_preds: 对比预测张量 [M, 6]。
#                 :param ious: 预测之间的IOU矩阵 [N, M]。
#                 :param iou_threshold: IOU合并阈值。
#                 :return: 合并后的预测张量。
#                 """
#                 # 应用掩码和IOU阈值
#                 valid_merge_mask = torch.max(ious, dim=1)[0] > iou_threshold
#                 if not valid_merge_mask.any():
#                     return preds  # 如果没有任何有效合并，直接返回原始预测
#                 # 计算所有可能的合并框坐标
#                 max_iou_idxs = torch.argmax(ious, dim=1)
#                 selected_bboxes = other_preds[max_iou_idxs, :4]
#                 merged_x1 = torch.min(preds[:, 0], selected_bboxes[:, 0])
#                 merged_x2 = torch.max(preds[:, 2], selected_bboxes[:, 2])
#                 merged_y1 = torch.min(preds[:, 1], selected_bboxes[:, 1])
#                 merged_y2 = torch.max(preds[:, 3], selected_bboxes[:, 3])   
#                 # 构建新的预测张量
#                 merged_preds = torch.stack([merged_x1, merged_y1, merged_x2, merged_y2, preds[:, 4], preds[:, 5]], dim=1)     
#                 # 更新预测张量
#                 non_merged_preds = preds[~valid_merge_mask]
#                 final_merged_preds = torch.cat([non_merged_preds, merged_preds[valid_merge_mask]], dim=0)  
#                 return final_merged_preds
#             deal_preds.append(merge_predictions(left_deal_preds, right_deal_preds, ious*right_overBor[None,:], merge_iou_thr))
#             deal_preds.append(merge_predictions(right_deal_preds, left_deal_preds, ious.T*left_overBor[None,:], merge_iou_thr))
#         deal_preds = torch.cat(deal_preds,dim=0)
#     # 执行NMS
#     nms_cfg = dict(type='nms', iou_threshold=0.7)
#     det_bboxes, keep_idxs = batched_nms(deal_preds[:,:4].float(), deal_preds[:,4].float(), deal_preds[:,-1].int(), nms_cfg)
#     deal_preds = deal_preds[keep_idxs]
#     all_preds = torch.cat([left_preds[~left_in_overlap],right_preds[~right_in_overlap],deal_preds], axis=0)
#     return all_preds

# def bbox_overimg_iou(box1, box2, right_start, left_end):
#     """
#     计算一组bbox和一组bbox在交叠区域(交叠区域x方向起点和终点为right_start,left_end)的IoU
#     box1: [N, 4]
#     box2: [M, 4]
#     """
#     x1 = torch.maximum(box1[:, None, 0], box2[:, 0]) # (N, M)
#     y1 = torch.maximum(box1[:, None, 1], box2[:, 1]) # (N, M)
#     x2 = torch.minimum(box1[:, None, 2], box2[:, 2]) # (N, M)
#     y2 = torch.minimum(box1[:, None, 3], box2[:, 3]) # (N, M)
#     # 计算交叉区域的面积
#     intersection_area = torch.maximum(torch.tensor(0), x2 - x1) * torch.maximum(torch.tensor(0), y2 - y1)  # (N, M)
#     x1 = torch.minimum(box1[:, None, 0], box2[:, 0]) # (N, M)
#     y1 = torch.minimum(box1[:, None, 1], box2[:, 1]) # (N, M)
#     x2 = torch.maximum(box1[:, None, 2], box2[:, 2]) # (N, M)
#     y2 = torch.maximum(box1[:, None, 3], box2[:, 3]) # (N, M)
#     x1 = torch.maximum(x1, torch.tensor(right_start)) # (N, M)
#     x2 = torch.minimum(x2, torch.tensor(left_end)) # (N, M)
#     # 计算并集的面积
#     area = (x2 - x1) * (y2 - y1) # (N, M)
#     iou = intersection_area / area # (N, M)
#     return iou

# def calculate_common_iou(boxes1, boxes2):
#     """
#     计算两组边界框之间的交并比 (IoU)
#     boxes1: (N, 4) 数组,表示 N 个边界框的 (x1, y1, x2, y2) 坐标
#     boxes2: (M, 4) 数组,表示 M 个边界框的 (x1, y1, x2, y2) 坐标
#     返回: (N, M) 数组,表示每对边界框之间的 IoU 值
#     """
#     # 计算交叉区域的坐标
#     x1 = np.maximum(boxes1[:, None, 0], boxes2[:, 0])  # (N, M)
#     y1 = np.maximum(boxes1[:, None, 1], boxes2[:, 1])  # (N, M)
#     x2 = np.minimum(boxes1[:, None, 2], boxes2[:, 2])  # (N, M)
#     y2 = np.minimum(boxes1[:, None, 3], boxes2[:, 3])  # (N, M)

#     # 计算交叉区域的面积
#     intersection_area = np.maximum(x2 - x1, 0) * np.maximum(y2 - y1, 0)  # (N, M)

#     # 计算每个边界框的面积
#     area_boxes1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])  # (N,)
#     area_boxes2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])  # (M,)

#     # 计算交并比 (IoU)
#     iou = intersection_area / (area_boxes1[:, None] + area_boxes2 - intersection_area)  # (N, M)

#     return iou
    
# full_Imgids = full_coco.getImgIds()
# full_ImgInfos = full_coco.loadImgs(ids=full_Imgids)
# full_nameIds = {i['file_name']:ids for ids,i in zip(full_Imgids,full_ImgInfos)}

# for i in full_Imgids:
#     # if i not in [659]:
#     #     continue
#     full_ImgInfos = full_coco.loadImgs(ids=i)
#     left_preds_index = file_name_indices[full_ImgInfos[0]['file_name'].replace('.jpg', '_l.jpg')]
#     right_preds_index = file_name_indices[full_ImgInfos[0]['file_name'].replace('.jpg', '_r.jpg')]
#     left_preds = results[left_preds_index]
#     right_preds = results[right_preds_index]
#     # 重映射到整图中
#     right_preds['bboxes'][:,0] += right_start
#     right_preds['bboxes'][:,2] += right_start
    
#     all_pr = merge_detections(np.concatenate([left_preds['bboxes'], left_preds['scores'][:,None], left_preds['labels'][:,None]],axis=1),
#                       np.concatenate([right_preds['bboxes'], right_preds['scores'][:,None], right_preds['labels'][:,None]],axis=1))
#     all_bboxes, all_scores, all_labels = all_pr[:,:4].numpy(),all_pr[:,4].numpy(),all_pr[:,5].numpy()
#     if all_bboxes.shape[0] != (left_preds['bboxes'].shape[0]+right_preds['bboxes'].shape[0]):
#         print('merged!!!!!!')     
#     #--------------------------full_coco.loadImgs(ids=i)
#     ann_ids = full_coco.get_ann_ids(img_ids=i)
#     anns = full_coco.load_anns(ids=ann_ids)
#     obj_xx = np.array([i['bbox'] for i in anns]).reshape(-1, 4)
#     obj_xx[:,2] = obj_xx[:,0]+obj_xx[:,2]
#     obj_xx[:,3] = obj_xx[:,1]+obj_xx[:,3] #xywh-->xyxy
#     scores = np.ones(obj_xx.shape[0]).astype(float)
#     all_bboxes_gt, all_scores_gt, all_labels_gt = obj_xx,scores,np.array([i['category_id'] for i in anns])
#     #--------------------------full_coco.loadImgs(ids=i)
#     if all_bboxes.shape[0] != all_bboxes_gt.shape[0]:
#         print('error',i)
#     elif all_bboxes.shape[0] == all_bboxes_gt.shape[0]:
#         ious = calculate_common_iou(all_bboxes,all_bboxes_gt)
#         # 检查是否是一对一匹配(ious=1.0)
#         ious_new = ious==1.0
#         is_one_to_one = np.all(ious_new.sum(axis=0) == 1) and np.all(ious_new.sum(axis=1) == 1)
#         if not is_one_to_one:
#             print('error',i)
            
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
prediction_path = data_prefix+'work_dirs_3/xx_test.pkl'
val_ann_file = 'LCDhalf_test_20240404_ReDup.json'
outputs = load(prediction_path)

metricIns = LineCocoMetric(ann_file=data_prefix + val_ann_file,
                     metric='bbox',
                     classwise=True,
                     line_pre=loss_bbox_pre,
                     box_num_cls=num_classes)
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
