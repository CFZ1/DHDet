"""
Created on Fri Mar 22 11:14:54 2024

@author: zcf
"""
import glob
import os
from tqdm import tqdm
import json
import copy
import torch
save_width = 6450
save_height = 3280
save_width_half = 3280
#一对多
# def calculate_iou(boxes1, boxes2):
#     # 计算交叉区域的坐标
#     x1 = torch.maximum(boxes1[:, 0], boxes2[:, 0])
#     y1 = torch.maximum(boxes1[:, 1], boxes2[:, 1])
#     x2 = torch.minimum(boxes1[:, 2], boxes2[:, 2])
#     y2 = torch.minimum(boxes1[:, 3], boxes2[:, 3])

#     # 计算交叉区域的面积
#     intersection_area = torch.maximum(x2 - x1, torch.zeros_like(x2)) * torch.maximum(y2 - y1, torch.zeros_like(y2))

#     # 计算每对边框的面积
#     area_boxes1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
#     area_boxes2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])

#     # 计算交并比
#     iou = intersection_area / (area_boxes1 + area_boxes2 - intersection_area)

#     return iou
#多对多
# 定义一个函数来计算两个边框的交并比
def calculate_iou(boxes1, boxes2):
    # 扩展boxes1形状为[11,1,3], 扩展boxes2形状为[1,11,3]
    boxes1 = boxes1.unsqueeze(1)
    boxes2 = boxes2.unsqueeze(0)

    # 计算交叉区域的坐标
    x1 = torch.maximum(boxes1[:, :, 0], boxes2[:, :, 0])
    y1 = torch.maximum(boxes1[:, :, 1], boxes2[:, :, 1])
    x2 = torch.minimum(boxes1[:, :, 2], boxes2[:, :, 2])
    y2 = torch.minimum(boxes1[:, :, 3], boxes2[:, :, 3])

    # 计算交叉区域的面积
    intersection_area = torch.maximum(x2 - x1, torch.zeros_like(x2)) * torch.maximum(y2 - y1, torch.zeros_like(y2))

    # 计算每对边框的面积
    area_boxes1 = (boxes1[:, :, 2] - boxes1[:, :, 0]) * (boxes1[:, :, 3] - boxes1[:, :, 1])
    area_boxes2 = (boxes2[:, :, 2] - boxes2[:, :, 0]) * (boxes2[:, :, 3] - boxes2[:, :, 1])

    # 计算交并比(返回形状为[11,11]的IoU矩阵)
    iou = intersection_area / (area_boxes1 + area_boxes2 - intersection_area)
    # 将对角线元素置为0
    # iou.fill_diagonal_(0)
    # 将对角线and上半对角角元素置为0元素置为0
    iou.tril_(diagonal=-1)
    return iou
def find_duplicate_boxes(obj_xx_ori,ious_threshold=0.5,img_name=None,type1=None):
    if len(obj_xx_ori) < 2:
        return []
    obj_xx = torch.tensor(obj_xx_ori)[:,:4]
    ious = calculate_iou(obj_xx, obj_xx)
    # 找到大于等于0.5的索引
    indices = torch.nonzero(ious >= ious_threshold, as_tuple=False)
    real_indices = []
    if indices.shape[0] > 0:
        # 获取真正的索引
        original_indices = [x[4] for x in obj_xx_ori]
        for index_pair in indices:
            real_index_pair = [original_indices[index_pair[0]], original_indices[index_pair[1]]]
            real_indices.append(real_index_pair)
        print('There may be duplicate bboxes here',img_name,'   ',type1)
        # `indices` 是一个列表包含了所有iou大于0.5的边框对
        for index in indices:
            print("Box {} 和 Box {} 的 IoU 大于 0.5".format(obj_xx[index[0]], obj_xx[index[1]]))
    return real_indices
# 设置要搜索的目录
directory = '/media/zcf/Elements/dataset/mobile_screen/0LCDMobileScreen/0LCD231201/0LCD240316/run4_240404_700full_json'
line_min=30
ious_threshold_glob = 0.5
use_remove = False
if ious_threshold_glob ==1.0:
    use_remove = True

# 构建搜索模式以匹配所有.json文件
pattern = os.path.join(directory, '*.json')
# 使用glob.glob()遍历匹配的文件
file_list = glob.glob(pattern)
for file_name in tqdm(file_list, desc="Converting files", unit="file", colour="green"):
    # if os.path.basename(file_name) not in ['01_主相机灰屏127(240).json']:
    #     continue
    with open(file_name, "r", encoding="utf-8") as f:
        data = json.load(f)
    # 遍历json中的所有<object>元素
    obj_sq_mura = []
    obj_line = []
    obj_TP = []
    for index in range(len(data['shapes'])):
        obj = copy.deepcopy(data['shapes'][index])
        points = obj['points']
        if obj['label'] == 'line':
            assert obj['shape_type'] == 'line', 'line Error'
            x_min = min(points[0][0], points[1][0])
            y_min = min(points[0][1], points[1][1])
            x_max = max(points[0][0], points[1][0])
            y_max = max(points[0][1], points[1][1])
            if (y_max-y_min)<line_min:
                y_max,y_min = (y_max+y_min)/2.+line_min/2.,(y_max+y_min)/2.-line_min/2.
            if (x_max-x_min)<line_min:
                x_max,x_min = (x_max+x_min)/2.+line_min/2.,(x_max+x_min)/2.-line_min/2.  
        else:
            x_min = min(points[0][0], points[2][0])
            y_min = min(points[0][1], points[2][1])
            x_max = max(points[0][0], points[2][0])
            y_max = max(points[0][1], points[2][1])
        if obj['label'] == 'line':
            obj_line.append([x_min,y_min,x_max,y_max,index])
        elif obj['label'] == 'TP':
            obj_TP.append([x_min,y_min,x_max,y_max,index])
        elif obj['label'] in ['sq','mura']:
            obj_sq_mura.append([x_min,y_min,x_max,y_max,index])
        elif obj['label'] in ['unknown','msq']:
            wo=1
        else:
            print(data['imagePath'])
    # indices = find_duplicate_boxes(obj_line,ious_threshold=0.5,img_name=data['imagePath'],type1='line')      
    real_indices = find_duplicate_boxes(obj_TP,ious_threshold=ious_threshold_glob,img_name=data['imagePath'],type1='TP')
    # flag = False
    # if len(real_indices) > 0 and use_remove:
    #     flag = True
    #     new_shapes = [data['shapes'][i] for i in range(len(data['shapes'])) if i not in [j[0] for j in real_indices]]
    real_indices = find_duplicate_boxes(obj_sq_mura,ious_threshold=ious_threshold_glob,img_name=data['imagePath'],type1='sq_mura')
    # if len(real_indices) > 0 and use_remove:
    #     flag = True
    #     new_shapes = [data['shapes'][i] for i in range(len(data['shapes'])) if i not in [j[0] for j in real_indices]]
    # if flag:
    #     data['shapes'] = new_shapes
    #     with open(file_name, "w", encoding="utf-8") as f:
    #         json.dump(data, f, indent=2, ensure_ascii=False)