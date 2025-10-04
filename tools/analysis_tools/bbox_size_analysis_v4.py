import json
import matplotlib.pyplot as plt
from collections import defaultdict
import os
import numpy as np
from pycocotools.coco import COCO

json_pre = '/media/zcf/Elements/dataset/mobile_screen/0LCDMobileScreen/0LCD231201/0LCD240316/'
json_pth = json_pre+'anns_20240404/LCDfull_all_20240404_ReDup.json' 
save_pth = '/media/zcf/extra/zcf/code/231108_FoundationModels/mmyolo_forv8/mmyolo/'
classes_all = ['TP','line','mura','sq']
save_width = 6450
save_height = 3280
input_w, input_h = 2700,2700

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

#-----------------------------------------
'''
步骤1: 计算缺陷尺寸类别的占比
我们将按照缺陷的面积分类:小(面积 ≤ 32x32),中(32x32 < 面积 ≤ 96x96),大(面积 > 96x96)。
'''        
def defect_size_category(width, height):
    area = width * height
    if area <= 32 * 32:
        return 'small'
    elif area <= 96 * 96:
        return 'medium'
    else:
        return 'large'
        
# 初始化计数器
defect_categories = defaultdict(lambda: {'small': 0, 'medium': 0, 'large': 0})

for annotation in data['annotations']:
    category_id = annotation['category_id']
    category_name = category_names[category_id]
    if category_name in classes_all:
        bbox = annotation['bbox']
        size_category = defect_size_category(bbox[2], bbox[3])
        defect_categories[category_name][size_category] += 1
        
# 计算占比
defect_proportions = {}
total = 0
for class_name in classes_all:
    total += sum(defect_categories[class_name].values())
for class_name in classes_all:
    proportions = {k: v / total for k, v in defect_categories[class_name].items()}
    defect_proportions[class_name] = proportions

#-----------------------------------------   
markers = ['o', 'x', 'd', '+']  # 不同形状的标记

# 找到最大的长度和宽度,以设置 x 轴和 y 轴的范围
max_width = max(max(size_distributions[class_name]['widths']) for class_name in classes_all)
max_height = max(max(size_distributions[class_name]['heights']) for class_name in classes_all)

# 绘制散点图和尺寸分布曲线,设置 x 轴和 y 轴的范围
limSpace = 100
aspect_ratio = (max_width+limSpace*2) / (max_height+limSpace*2)
fig_width = 10  # 宽度固定为10英寸
fig_height = fig_width / aspect_ratio  # 高度根据比例自动计算
fig, ax = plt.subplots(figsize=(fig_width, fig_height))

plt.rcParams['font.family'] = 'Times New Roman'
ax.set_xlabel('Width')
ax.set_ylabel('Height')

# 设置坐标轴范围和刻度
ax.set_xlim([-limSpace, max_width + limSpace])
ax.set_ylim([-limSpace, max_height + limSpace])
ax.set_xticks(np.arange(0, max_width + limSpace + 1, 1000))
ax.set_yticks(np.arange(0, max_height + limSpace + 1, 500))

# 绘制散点图
for i, class_name in enumerate(classes_all):
    widths = size_distributions[class_name]['widths']
    heights = size_distributions[class_name]['heights']
    ax.scatter(widths, heights, s=50, marker=markers[i], label=class_name, alpha=0.5)

# 绘制区分不同面积范围的曲线
# 曲线1: 面积等于 32*32 的曲线
area1 = 32 * 32
x_vals1 = np.linspace(0.1, max_width, 400)  # 避免除以零
y_vals1 = area1 / x_vals1
ax.plot(x_vals1, y_vals1, 'b--', label='Small-Medium Object Boundary', linewidth=1)

# 曲线2: 面积等于 96*96 的曲线
area2 = 96 * 96
x_vals2 = np.linspace(0.1, max_width, 400)  # 避免除以零
y_vals2 = area2 / x_vals2
ax.plot(x_vals2, y_vals2, 'r--', label='Medium-Large Object Boundary', linewidth=1)

# 显示图例
ax.legend(loc='lower right')

# 在右上角绘制横向柱状图
ax2 = fig.add_axes([0.6, 0.6, 0.3, 0.3])  

categories = ['small', 'medium', 'large'] 
colors = ['#1f77b4', '#ff7f0e', '#2ca02c']  # 为每个尺寸类别指定一个颜色
width = 0.8  # 设置柱状图的总宽度
# for i, class_name in enumerate(classes_all):
#     proportions = [defect_proportions[class_name][cat] for cat in categories]
#     ax2.barh(class_name, proportions, height=0.2, label=class_name)
    
for i, class_name in enumerate(classes_all):
    left = 0
    for j, category in enumerate(categories):
        proportion = defect_proportions[class_name][category]
        ax2.barh(i, proportion, height=width, left=left, color=colors[j], label=category if i == 0 else '')
        left += proportion

# ax2.set_xlim([0, 1])
# ax2.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1])
# ax2.set_xticklabels(['0%', '20%', '40%', '60%', '80%', '100%'])  
# ax2.legend(loc='upper left', bbox_to_anchor=(1.05, 1), fontsize='small')
# ax2.set_title('Defect Size Proportions')

ax2.set_xlim([0, 1])
ax2.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1])
ax2.set_xticklabels(['0%', '20%', '40%', '60%', '80%', '100%'])  
ax2.set_yticks(range(len(classes_all)))
ax2.set_yticklabels(classes_all)
ax2.legend(loc='upper right', bbox_to_anchor=(1.05, 1), fontsize='small')
ax2.set_title('Defect Size Proportions')

# plt.tight_layout()
plt.savefig(os.path.join(save_pth, 'f12_WH-eps-converted-to.pdf'), dpi=600)
plt.close()
