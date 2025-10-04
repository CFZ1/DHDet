"""
Created on Fri Mar 15 19:38:23 2024

@author: zcf
"""

import numpy as np
from mmdet.datasets.api_wrappers import COCO
import networkx as nx
import json
import copy
import torch
from tqdm import tqdm
#
# 多对多, 定义一个函数来计算两个边框的交并比
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
    iou.fill_diagonal_(0)
    # 将对角线and上半对角角元素置为0元素置为0
    # iou.tril_(diagonal=-1)
    return iou

def merge_bboxs(input_bboxs):
    # 转换为NumPy数组
    points_np = np.array(input_bboxs)
    # 寻找最小的x（x1），最小的y（y1），最大的x（x2），最大的y（y2）
    x_min = np.min(points_np[:, :, 0])
    y_min = np.min(points_np[:, :, 1])
    x_max = np.max(points_np[:, :, 0])
    y_max = np.max(points_np[:, :, 1])
    if (y_max-y_min)<line_min:
        y_max,y_min = (y_max+y_min)/2.+line_min/2.,(y_max+y_min)/2.-line_min/2.
    if (x_max-x_min)<line_min:
        x_max,x_min = (x_max+x_min)/2.+line_min/2.,(x_max+x_min)/2.-line_min/2.  
    print(x_min,y_min,x_max,y_max)
    return [x_min,y_min,x_max-x_min,y_max-y_min]  
def get_nums(components,obj_line,save_width_half,right_st):
    left_counts = []
    right_counts = []
    for comp in components:
        left_count = 0
        right_count = 0
        for idx in comp:
            xmin, ymin = obj_line[idx, 0]
            xmax, ymax = obj_line[idx, 1]  
            # 判断矩形是否在左半图
            if xmin <= save_width_half:
                left_count += 1
            # 判断矩形是否在右半图
            if xmax >= right_st:
                right_count += 1
        left_counts.append(left_count)
        right_counts.append(right_count)
    return left_counts,right_counts
def find_components(coco_local,img_id, catIds, full_mode=False):
    ann_ids = coco_local.getAnnIds(imgIds=img_id, catIds=catIds)
    if len(ann_ids) > 1:
        gts_here = coco_local.load_anns(ids=ann_ids)
        # 初始化一个空的NumPy数组，大小为n*(n-1)/2，其中n是边界框的数量
        num_bboxes = len(gts_here)
        # 计算每对边界框的IoU并存储到数组
        obj_xx = torch.tensor([i['bbox'] for i in gts_here])
        obj_xx[:,2] = obj_xx[:,0]+obj_xx[:,2]
        obj_xx[:,3] = obj_xx[:,1]+obj_xx[:,3] #xywh-->xyxy
        iou_torch = calculate_iou(obj_xx, obj_xx)
        
        if torch.any(iou_torch > iou_threshold):
            # 使用networkx找出所有应该合并的边界框组
            row, col = torch.nonzero(iou_torch > iou_threshold, as_tuple=True)
            edges = list(zip(row.tolist(), col.tolist()))
            G = nx.Graph()
            G.add_edges_from(edges)
            
            # 找出所有连接的组件
            components = list(nx.connected_components(G))
            if full_mode:
                obj_line = torch.tensor([i['line'] for i in gts_here])
                left_counts,right_counts = get_nums(components,obj_line,save_width_half=3280,right_st=6450-3280)
                if sum(left_counts)==0:
                    left_counts = [0]
                if sum(right_counts)==0:
                    right_counts = [0]
                return left_counts,right_counts
            else:
                return [len(i) for i in components]
    if full_mode:
        return [0], [0]
    return [0]
        

pre_path = '/media/zcf/Elements/dataset/mobile_screen/0LCDMobileScreen/0LCD231201/0LCD240316/anns_20240404_copy/'
iou_threshold = 0.5
line_min = 30
check3condition = False
# test_path = pre_path+'LCDhalf_test.json' #4次IOU大于0.5
if not check3condition:
    test_path = pre_path+'LCDhalf_test_20240404_ReDup.json' #4次IOU大于0.5
    class_name = ('TP','mura','sq') #('bubble','scratch','pinhole','tin_ash')
    num_classes = len(class_name) # Number of classes for classification
    line_classes = ('line',)
    num_line_classes = len(line_classes) 
    dataset_meta = dict(classes=class_name,line_classes=line_classes,bbox_include_line=False,palette=[(220, 20, 60), (119, 11, 32), (0, 0, 142), (0, 0, 230)])    
    coco = COCO(test_path)
    coco.dataset['categories']
    cat_ids = coco.get_cat_ids(cat_names=dataset_meta['classes'])
    cat_ids_line = coco.get_cat_ids(cat_names=dataset_meta['line_classes'])
    img_ids = coco.get_img_ids()
    dup_count = 0
    for imgIds_here in img_ids: 
        ann_ids = coco.getAnnIds(imgIds=imgIds_here, catIds=cat_ids_line)
        if len(ann_ids) > 1:
            gts_here = coco.loadAnns(ann_ids)
            img_info = coco.loadImgs(imgIds_here)
            # 初始化一个空的NumPy数组，大小为n*(n-1)/2，其中n是边界框的数量
            num_bboxes = len(gts_here)
            # 计算每对边界框的IoU并存储到数组
            obj_xx = torch.tensor([i['bbox'] for i in gts_here])
            obj_xx[:,2] = obj_xx[:,0]+obj_xx[:,2]
            obj_xx[:,3] = obj_xx[:,1]+obj_xx[:,3] #xywh-->xyxy
            iou_torch = calculate_iou(obj_xx, obj_xx)
            
            if torch.any(iou_torch > iou_threshold):
                # 使用networkx找出所有应该合并的边界框组
                row, col = torch.nonzero(iou_torch > iou_threshold, as_tuple=True)
                edges = list(zip(row.tolist(), col.tolist()))
                G = nx.Graph()
                G.add_edges_from(edges)
                
                # 找出所有连接的组件
                components = list(nx.connected_components(G))
                gts_here_copy = copy.deepcopy(gts_here)
                print('img_info',img_info)
                print('components',components)
                # if len(components) ==2:
                #     print('img_info',img_info)
                #  将IOU大于line_merge_iou_threshold的Line box合并给第一个，其他都置为None，然后移除
                for i in components:
                    input_bboxs = [gts_here[j]['line'] for j in i]
                    gts_here_copy[min(i)]['bbox'] = merge_bboxs(input_bboxs)
                    gts_here_copy[min(i)]['line'] = input_bboxs
                    dup_count +=len(list(i-{min(i)}))
                    for kk in list(i-{min(i)}):
                        gts_here_copy[kk]['bbox'] = None
                # 修改标注
                for i,ann_id in enumerate(ann_ids):
                    for j,dataset_ann in enumerate(coco.dataset['annotations']):
                        if dataset_ann['id'] == ann_id:
                            coco.dataset['annotations'][j] = gts_here_copy[i]
    # 在完成所有的修改之后，遍历并移除bbox为None的标注
    coco.dataset['annotations'] = [ann for ann in coco.dataset['annotations'] if ann['bbox'] is not None]
    # 定义输出路径
    output_path = test_path.split('.json')[0]+'_lineIoU0d5'+'.json'
    # 将修改后的数据集写入新的JSON文件
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(coco.dataset, f, indent=2, ensure_ascii=False)
    print("Modified annotations saved to", output_path)
    print('dup_count',dup_count)
    
if check3condition:
    class_name = ('TP','mura','sq') #('bubble','scratch','pinhole','tin_ash')
    num_classes = len(class_name) # Number of classes for classification
    line_classes = ('line',)
    num_line_classes = len(line_classes) 
    dataset_meta = dict(classes=class_name,line_classes=line_classes,bbox_include_line=False,palette=[(220, 20, 60), (119, 11, 32), (0, 0, 142), (0, 0, 230)])    
   
    fullImgAnn = pre_path+'LCDfull_all_20240404_ReDup.json'
    halfImgAnn = pre_path+'LCDhalf_all_20240404_ReDup.json'
    full_coco = COCO(fullImgAnn)
    half_coco = COCO(halfImgAnn)
    cat_ids_line = full_coco.get_cat_ids(cat_names=dataset_meta['line_classes'])
    #键是file_name,值是对应结果在results列表中的索引。
    half_Imgids = half_coco.getImgIds()  
    file_name_indices = {half_coco.loadImgs(ids=idx_here)[0]['file_name']:idx_here for idx_here in half_Imgids}
    
    full_Imgids = full_coco.getImgIds()
    for i in tqdm(full_Imgids):
        # if not (i==674):
        #     continue 
        full_ImgInfos = full_coco.loadImgs(ids=i)
        left_preds_index = file_name_indices[full_ImgInfos[0]['file_name'].replace('.jpg', '_l.jpg')]
        right_preds_index = file_name_indices[full_ImgInfos[0]['file_name'].replace('.jpg', '_r.jpg')]
        
        left_counts,right_counts = find_components(full_coco,img_id=i, catIds=cat_ids_line,full_mode=True)
        left_counts_2 = find_components(half_coco,img_id=left_preds_index, catIds=cat_ids_line)
        right_counts_2 = find_components(half_coco,img_id=right_preds_index, catIds=cat_ids_line)
        if left_counts ==left_counts_2 and right_counts==right_counts_2:
            wo =1
        else:
            print('error ',i, full_ImgInfos[0]['file_name'],'full_left=',left_counts,'full_right=',right_counts,'half_left=',left_counts_2,'half_right=',right_counts_2)
        
        
        
        
    
                    

