"""
Created on Wed Feb 28 13:59:49 2024

@author: zcf
"""

import json
import matplotlib.pyplot as plt
from collections import defaultdict
import os
import numpy as np
from pycocotools.coco import COCO

json_pre = '/media/zcf/Elements/dataset/mobile_screen/0LCDMobileScreen/0LCD231201/0LCD240316/'
json_pth = json_pre+'anns_20240404/LCDhalf_all_20240404_ReDup.json' 
save_pth = '/media/zcf/extra/zcf/code/231108_FoundationModels/mmyolo_forv8/mmyolo/'
classes_all = ['TP','line','mura','sq']
save_width = 6450
save_height = 3280
input_w, input_h = 2700,2700

'''统计每张图像注释的个数'''
# coco = COCO(json_pth)
# coco_image_ids = coco.getImgIds()
# # img_info = coco.loadImgs(img_id)[0]
# lenAnnsPerImg = [len(coco.loadAnns(coco.getAnnIds(imgIds=img_id))) for img_id in coco_image_ids]

# 加载 JSON 文件
with open(json_pth, 'r') as f:
    data = json.load(f)   
 # 准备结构存储长度和宽度数据
size_distributions = defaultdict(lambda: {'widths': [], 'heights': [], 'ratio_WH_HWs': []})
# 映射category_id到它的名字
category_names = {cat['id']: cat['name'] for cat in data['categories']}

# 遍历annotations来填充分布数据
for annotation in data['annotations']:
    category_id = annotation['category_id']
    category_name = category_names[category_id]
    bbox = annotation['bbox']
    # bbox format is [x, y, width, height]
    width, height = bbox[2], bbox[3]
    ratio_WH_HW = max(float(width)/float(height),float(height)/float(width))
    size_distributions[category_name]['widths'].append(width)
    size_distributions[category_name]['heights'].append(height)
    size_distributions[category_name]['ratio_WH_HWs'].append(ratio_WH_HW)
    if category_name in classes_all:
        size_distributions['all']['widths'].append(width)
        size_distributions['all']['heights'].append(height)
        size_distributions['all']['ratio_WH_HWs'].append(ratio_WH_HW)
# 现在size_distributions包含了每个类别的边界框的宽度和高度列表
# 收集所有ratio_WH_HW以进行全局排序和归一化
all_ratios = []
for dists in size_distributions.values():
    all_ratios.extend(dists['ratio_WH_HWs'])

# 计算全局的归一化颜色
norm = plt.Normalize(min(all_ratios), max(all_ratios))
cmap = plt.get_cmap('tab20')

# 设置图形大小
fig, ax = plt.subplots(figsize=(10,10))
# 绘制散点图
markers = ['o', 'X', 's', '>']
for i, cls in enumerate(classes_all):
    colors = [cmap(norm(r)) for r in size_distributions[cls]['ratio_WH_HWs']]
    sc = ax.scatter(size_distributions[cls]['widths'], size_distributions[cls]['heights'],
                    c=colors, s=10, marker=markers[i], label=cls)

# 设置颜色条
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
cbar = fig.colorbar(sm)
cbar.set_label('Width/Height Ratio', rotation=270, labelpad=20)

# 设置坐标轴标签
ax.set_xlabel('Width')
ax.set_ylabel('Height')
# 设置图形标题
ax.set_title('Scatter Plot of Object Sizes (All Categories)')
# 设置坐标轴的比例为相等，确保单位长度相同
ax.set_aspect('equal')
# 根据数据的最大值来调整坐标轴的限制
max_width = max([max(size_distributions[cls]['widths']) for cls in classes_all])
max_height = max([max(size_distributions[cls]['heights']) for cls in classes_all])
max_limit = max(max_width, max_height)
ax.set_xlim(-100, max_limit+100)
ax.set_ylim(-100, max_limit+100)
# 添加图例
# ax.legend()
legend = ax.legend()
for legend_handle in legend.legendHandles:  # 遍历图例句柄以移除颜色
    legend_handle.set_color('white') # 设置图例中的图标颜色为黑色
    # 设置背景帧的边框颜色为黑色
    legend_handle.set_edgecolor('black')
# 展示图形
plt.savefig(os.path.join(save_pth, 'f1_WH-eps-converted-to.pdf'), dpi=600)
plt.close()



# 绘制全部类别的宽度和高度的散点图
plt.figure(1,figsize=(5, 5))#int(5/save_width*save_height)))
# 使用内置的颜色映射
colors = plt.cm.tab10(np.linspace(0, 1, len(classes_all)))
for i, cls in enumerate(classes_all):
    plt.scatter(size_distributions[cls]['widths'], size_distributions[cls]['heights'], s=4.0, linewidths=0.0,color=colors[i], label=cls)
# plt.scatter(size_distributions['all']['widths'], size_distributions['all']['heights'], s=0.1, alpha=1.0)
plt.xlabel('Width')
plt.ylabel('Height')
plt.title('Scatter Plot of Object Sizes (All Categories)')
# plt.show()
# plt.subplots_adjust(left=0.138, right=0.99, top=0.97, bottom=0.195)
plt.savefig(os.path.join(save_pth,'f1_WH-eps-converted-to.pdf'), dpi=600)
plt.close()
# 你可以按任何方式处理或展示这些数据
# 例如，打印每个类别的平均宽度和高度
for category, sizes in size_distributions.items():
    avg_width = sum(sizes['widths']) / len(sizes['widths'])
    avg_height = sum(sizes['heights']) / len(sizes['heights'])
    min_width, max_width = min(sizes['widths']), max(sizes['widths'])
    min_height, max_height = min(sizes['heights']), max(sizes['heights'])
    print(f"{category}: avg_width = {avg_width}, min_width = {min_width}, max_width = {max_width}")
    print(f"{category}: avg_height = {avg_height}, min_height = {min_height}, max_height = {max_height}")

# 绘制每个类别的宽度和高度分布直方图
for category, data in size_distributions.items():
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.hist(data['widths'], bins=20, color='blue', alpha=0.7)
    plt.title(f'{category} Width Distribution')
    plt.xlabel('Width')
    plt.ylabel('Frequency')
    plt.subplot(1, 2, 2)
    plt.hist(data['heights'], bins=20, color='green', alpha=0.7)
    plt.title(f'{category} Height Distribution')
    plt.xlabel('Height')
    plt.ylabel('Frequency')
    plt.tight_layout()
    plt.show()


