"""
Created on Mon Mar 25 14:41:35 2024

@author: zcf
"""
# from PIL import Image
# import hashlib
# import glob
# from tqdm import tqdm
# import os
# import shutil

# def compare_images(img_path1, img_path2):
#     """比较两个图像的像素数据"""
#     with Image.open(img_path1) as img1, Image.open(img_path2) as img2:
#         if img1.size != img2.size:
#             return False
#         return list(img1.getdata()) == list(img2.getdata())

# # 假设路径是 "/mnt/data/images2_crop_full"
# path = "/media/zcf/Elements/dataset/mobile_screen/0LCDMobileScreen/0LCD231201/0LCD240316/images2_crop_full"  # 修改为实际的路径
# pattern = os.path.join(path, '*.jpg')
# jpg_files = glob.glob(pattern)

# # 创建一个字典来存储每个文件的哈希值
# hash_dict = {}
# # 用于存储哈希值重复的图像
# duplicates = {}
# # 用于存储像素数据重复的图像
# duplicates_pixels = {}

# for file_path in tqdm(jpg_files):
#     with open(file_path, 'rb') as image_file:
#         data = image_file.read()
#         image_hash = hashlib.md5(data).hexdigest()
#     filename = os.path.basename(file_path)

#     if image_hash in hash_dict:
#         # 如果哈希值已存在
#         original_file = hash_dict[image_hash][0]  # 获取第一个出现的文件作为比较基准
#         if original_file in duplicates:
#             duplicates[original_file].append(filename)
#         else:
#             duplicates[original_file] = [filename] 
#         if compare_images(os.path.join(path,original_file), file_path):
#             if original_file in duplicates_pixels:
#                 duplicates_pixels[original_file].append(filename)
#             else:
#                 duplicates_pixels[original_file] = [filename]    
#     else:
#         hash_dict[image_hash] = [filename]

# 定义源目录和目标目录
# source_dir = "/media/zcf/Elements/dataset/mobile_screen/0LCDMobileScreen/0LCD231201/0LCD240316/images2_crop_full"
# target_dir = "/media/zcf/Elements/dataset/mobile_screen/0LCDMobileScreen/0LCD231201/0LCD240316/images2_crop_full_duplicates"

# # 确保目标目录存在，如果不存在则创建
# if not os.path.exists(target_dir):
#     os.makedirs(target_dir)

# # 遍历字典，移动文件
# for key, values in duplicates.items():
#     for value in values:
#         source_path = os.path.join(source_dir, value)
#         target_path = os.path.join(target_dir, value)
#         # 移动文件
#         if os.path.exists(source_path):
#             shutil.move(source_path, target_path)
#             print(f"Moved {value} to {target_dir}")
#         source_path = os.path.join(source_dir, value).replace('.jpg', '.json')
#         target_path = os.path.join(target_dir, value).replace('.jpg', '.json')
#         # 移动文件
#         if os.path.exists(source_path):
#             shutil.move(source_path, target_path)
#             print(f"Moved {value} to {target_dir}")

from mmdet.datasets.api_wrappers import COCO
import os
import cv2
import numpy as np
from tqdm import tqdm
import glob

# 假设路径是 "/mnt/data/images2_crop_full"
path = "/media/zcf/Elements/dataset/mobile_screen/0LCDMobileScreen/0LCD231201/0LCD240316/" 
path_img_full = path+'images2_crop_full'
path_img_half = path+'images2_crop'
jpg_files_full = glob.glob(os.path.join(path_img_full, '*.jpg'))
jpg_files_half = glob.glob(os.path.join(path_img_half, '*.jpg'))
jpg_files_full = [os.path.basename(i) for i in jpg_files_full]
jpg_files_half = [os.path.basename(i) for i in jpg_files_half]
jpg_files_half = [fname.replace('_l.jpg', '.jpg').replace('_r.jpg', '.jpg') for fname in jpg_files_half]
jpg_files_half = list(set(jpg_files_half))
jpg_files_half_2_RE_jpg_files_full = [i for i in jpg_files_half if i not in jpg_files_full]
jpg_files_full_2_RE_jpg_files_half = [i for i in jpg_files_full if i not in jpg_files_half]
save_width = 6450
save_height= 3280
# 删除, 遍历源文件夹中的所有文件
for img_name in tqdm(jpg_files_half_2_RE_jpg_files_full):
    left_part_path = os.path.join(path_img_half, img_name.split('.jpg')[0]+'_l.jpg') 
    right_part_path = os.path.join(path_img_half, img_name.split('.jpg')[0]+'_r.jpg')
    os.remove(left_part_path)
    os.remove(right_part_path)   
# 增加, 遍历源文件夹中的所有文件
for img_name in tqdm(jpg_files_full_2_RE_jpg_files_half):
    source_path = os.path.join(path_img_full, img_name)
    cropped_img = cv2.imread(source_path)
    left_part = cropped_img[:, :save_height] 
    right_part = cropped_img[:, -save_height:]  
    left_part_path = os.path.join(path_img_half, img_name.split('.jpg')[0]+'_l.jpg') 
    right_part_path = os.path.join(path_img_half, img_name.split('.jpg')[0]+'_r.jpg') 
    cv2.imwrite(left_part_path, left_part)
    cv2.imwrite(right_part_path, right_part)