"""
Created on Sun Mar  9 10:16:38 2025

@author: zcf
"""
import torch
def bbox_overimg_iou(box1, box2, right_start, left_end, merge_iou_thr=0.5, eps=1e-6, debug_label=False):
    """
    计算一组bbox和一组bbox在交叠区域的IoU
    box1: [N, 4]
    box2: [M, 4]
    right_start, left_end: 重叠区域的x坐标范围
    """
    # 1. 首先计算两个box的交集
    x1_internal = torch.maximum(box1[:, None, 0], box2[:, 0]) # (N, M)
    y1_internal = torch.maximum(box1[:, None, 1], box2[:, 1]) # (N, M)
    x2_internal = torch.minimum(box1[:, None, 2], box2[:, 2]) # (N, M)
    y2_internal = torch.minimum(box1[:, None, 3], box2[:, 3]) # (N, M)
    
    # 2. 将交集限制在重叠区域内
    x1_internal = torch.maximum(x1_internal, torch.tensor(right_start))
    x2_internal = torch.minimum(x2_internal, torch.tensor(left_end))
    
    # 3. 计算交集面积 (box1∩box2)∩OL-region
    intersection_area = torch.maximum(torch.tensor(0), x2_internal - x1_internal) * \
                       torch.maximum(torch.tensor(0), y2_internal - y1_internal)  # (N, M)

    # 4. 计算box1和box2在重叠区域内的面积
    # box1在重叠区域内的面积
    x1_box1 = torch.maximum(box1[:, None, 0], torch.tensor(right_start))
    x2_box1 = torch.minimum(box1[:, None, 2], torch.tensor(left_end))
    area_box1 = torch.maximum(torch.tensor(0), x2_box1 - x1_box1) * \
                (box1[:, None, 3] - box1[:, None, 1])  # height of box1
    
    # box2在重叠区域内的面积
    x1_box2 = torch.maximum(box2[:, 0], torch.tensor(right_start))
    x2_box2 = torch.minimum(box2[:, 2], torch.tensor(left_end))
    area_box2 = torch.maximum(torch.tensor(0), x2_box2 - x1_box2) * \
                (box2[:, 3] - box2[:, 1])  # height of box2
    
    # 5. 计算并集面积 (box1∪box2)∩OL-region = box1∩OL-region + box2∩OL-region - (box1∩box2)∩OL-region
    union_area = area_box1 + area_box2 - intersection_area
    union_area = torch.maximum(union_area, torch.tensor(eps)) # 避免除0

    # 6. 计算IoU
    iou = intersection_area / union_area 
    
    return iou

# 之前的计算在并集的地方计算错误，应该是两个aera相加，然后减去并集的部分
# def bbox_overimg_iou(box1, box2, right_start, left_end, merge_iou_thr=0.5, debug_label=False):
#     """
#     计算一组bbox和一组bbox在交叠区域(交叠区域x方向起点和终点为right_start,left_end)的IoU
#     box1: [N, 4]
#     box2: [M, 4]
#     """
#     x1_internal = torch.maximum(box1[:, None, 0], box2[:, 0]) # (N, M)
#     y1_internal = torch.maximum(box1[:, None, 1], box2[:, 1]) # (N, M)
#     x2_internal = torch.minimum(box1[:, None, 2], box2[:, 2]) # (N, M)
#     y2_internal = torch.minimum(box1[:, None, 3], box2[:, 3]) # (N, M)
#     # 计算交叉区域的面积
    # intersection_area = torch.maximum(torch.tensor(0), x2_internal - x1_internal) * torch.maximum(torch.tensor(0), y2_internal - y1_internal)  # (N, M)
    # x1 = torch.minimum(box1[:, None, 0], box2[:, 0]) # (N, M)
    # y1 = torch.minimum(box1[:, None, 1], box2[:, 1]) # (N, M)
    # x2 = torch.maximum(box1[:, None, 2], box2[:, 2]) # (N, M)
    # y2 = torch.maximum(box1[:, None, 3], box2[:, 3]) # (N, M)
    # x1 = torch.maximum(x1, torch.tensor(right_start)) # (N, M)
    # x2 = torch.minimum(x2, torch.tensor(left_end)) # (N, M)
    # # 计算并集的面积
    # area = (x2 - x1) * (y2 - y1) # (N, M)
#     iou = intersection_area / area # (N, M)
#     # # 计算并集的面积
#     # intersectionx = torch.maximum(torch.tensor(0), (x2_internal-x1_internal)/(x2-x1))
#     # intersectiony = torch.maximum(torch.tensor(0), (y2_internal-y1_internal)/(y2-y1))
#     # # 找到 iou > 0.5 的位置
#     # mask = iou > 0.3  
#     # # 使用布尔索引获取满足条件的 intersectionx 和 intersectiony 值
#     # x_values = intersectionx[mask]
#     # y_values = intersectiony[mask]
#     # iou = (intersectionx>merge_iou_thr)*(intersectionx+merge_iou_thr)*intersectiony #merge_iou_thr=0.5
#     # iou = (intersectionx>0.1)*intersectiony #merge_iou_thr=0.5
#     # if debug_label and x_values.shape[0]>0:
#     #     print('x_iou',torch.mean(x_values),'y_iou',torch.mean(y_values))
#     return iou


def test_bbox_overimg_iou():
    # 测试用例1: 两个完全重叠的框，都在重叠区域内
    box1 = torch.tensor([[100., 100., 200., 200.]])  # [x1, y1, x2, y2]
    box2 = torch.tensor([[100., 100., 200., 200.]])
    right_start = 0
    left_end = 300
    iou = bbox_overimg_iou(box1, box2, right_start, left_end)
    print("测试用例1 - 完全重叠的框:")
    print(f"Expected IoU: 1.0, Got IoU: {iou.item():.4f}\n")

    # 测试用例2: 部分重叠的框，都在重叠区域内
    box1 = torch.tensor([[100., 100., 200., 200.]])
    box2 = torch.tensor([[150., 150., 250., 250.]])
    right_start = 0
    left_end = 300
    iou = bbox_overimg_iou(box1, box2, right_start, left_end)
    print("测试用例2 - 部分重叠的框:")
    print(f"Got IoU: {iou.item():.4f}\n")

    # 测试用例3: 框部分在重叠区域外
    box1 = torch.tensor([[50., 100., 150., 200.]])
    box2 = torch.tensor([[100., 100., 200., 200.]])
    right_start = 100
    left_end = 200
    iou = bbox_overimg_iou(box1, box2, right_start, left_end)
    print("测试用例3 - 框部分在重叠区域外:")
    print(f"Got IoU: {iou.item():.4f}\n")

    # 测试用例4: 无重叠的框
    box1 = torch.tensor([[100., 100., 150., 150.]])
    box2 = torch.tensor([[200., 200., 250., 250.]])
    right_start = 0
    left_end = 300
    iou = bbox_overimg_iou(box1, box2, right_start, left_end)
    print("测试用例4 - 无重叠的框:")
    print(f"Expected IoU: 0.0, Got IoU: {iou.item():.4f}\n")

    # 测试用例5: 批量测试
    box1 = torch.tensor([
        [100., 100., 200., 200.],
        [300., 300., 400., 400.]
    ])
    box2 = torch.tensor([
        [150., 150., 250., 250.],
        [350., 350., 450., 450.]
    ])
    right_start = 0
    left_end = 500
    iou = bbox_overimg_iou(box1, box2, right_start, left_end)
    print("测试用例5 - 批量测试:")
    print(f"IoU matrix:\n{iou}\n")

    # 测试用例6: 框完全在重叠区域外
    box1 = torch.tensor([[50., 100., 80., 200.]])
    box2 = torch.tensor([[60., 100., 70., 200.]])
    right_start = 100
    left_end = 200
    iou = bbox_overimg_iou(box1, box2, right_start, left_end)
    print("测试用例6 - 框完全在重叠区域外:")
    print(f"Expected IoU: 0.0, Got IoU: {iou.item():.4f}")

# 运行测试
if __name__ == "__main__":
    test_bbox_overimg_iou()
