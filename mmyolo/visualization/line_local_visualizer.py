# ****************Latest version 2025-03-27-17:00:00
# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, List, Optional, Tuple, Union

import cv2
import mmcv
import numpy as np
import torch
from mmengine.dist import master_only
from mmengine.structures import InstanceData, PixelData
from mmdet.visualization import DetLocalVisualizer

from mmyolo.registry import VISUALIZERS
from mmdet.structures import DetDataSample
from mmdet.visualization.palette import _get_adaptive_scales, get_palette
from mmdet.structures.bbox import BaseBoxes
from imgaug.augmentables.lines import LineStringsOnImage


@VISUALIZERS.register_module()
class LineDetLocalVisualizer(DetLocalVisualizer):
    def __init__(self,
                 name: str = 'visualizer',
                 image: Optional[np.ndarray] = None,
                 vis_backends: Optional[Dict] = None,
                 save_dir: Optional[str] = None,
                 bbox_color: Optional[Union[str, Tuple[int]]] = None,
                 text_color: Optional[Union[str,
                                            Tuple[int]]] = (200, 200, 200),
                 mask_color: Optional[Union[str, Tuple[int]]] = None,
                 line_width: Union[int, float] = 3,
                 alpha: float = 0.8,
                 only_plot_SEpoints: bool = False #--------------add for LineDetLocalVisualizer
                 ) -> None:
        super().__init__(
            name=name,
            image=image,
            vis_backends=vis_backends,
            save_dir=save_dir,
            bbox_color=bbox_color,
            text_color=text_color,
            mask_color=mask_color,
            line_width=line_width,
            alpha=alpha
        )
        self.only_plot_SEpoints = only_plot_SEpoints #因为draw_lines绘制的线条只支持：num_ponts=2
    # copy from mmdet/visualization/local_visualizer.py and change
    def _draw_line_instances(self, image: np.ndarray, instances: ['InstanceData'],
                        classes: Optional[List[str]],
                        palette: Optional[List[tuple]]) -> np.ndarray:
        """Draw instances of GT or prediction.

        Args:
            image (np.ndarray): The image to draw.
            instances (:obj:`InstanceData`): Data structure for
                instance-level annotations or predictions.
            classes (List[str], optional): Category information.
            palette (List[tuple], optional): Palette information
                corresponding to the category.

        Returns:
            np.ndarray: the drawn image which channel is RGB.
        """
        self.set_image(image)

        if 'line_points' in instances: #-------zhou
            bboxes = instances.line_points #-------zhou
            labels = instances.line_labels #-------zhou
            # print('bboxes',instances.line_points)
            # if 'line_scores' in instances:
            #     print('score',instances.line_scores)

            max_label = int(max(labels) if len(labels) > 0 else 0)
            text_palette = get_palette(self.text_color, max_label + 1)
            text_colors = [text_palette[label] for label in labels]

            bbox_color = palette if self.bbox_color is None \
                else self.bbox_color
            bbox_palette = get_palette(bbox_color, max_label + 1)
            colors = [bbox_palette[label] for label in labels]
            #-------zhou
            if isinstance(bboxes, LineStringsOnImage):
                # 提取坐标
                x_datas, y_datas = [], []
                x_datas_SEpoints, y_datas_SEpoints = [], []
                # polygons = []
                for line_string in bboxes: #LineStringsOnImage, or tensor
                    x_coords, y_coords = zip(*line_string.coords) 
                    # x_datas.append(list(x_coords)) # Get all points
                    # y_datas.append(list(y_coords))
                    x_datas_SEpoints.append([x_coords[0],x_coords[-1]]) # Get start points and end points
                    y_datas_SEpoints.append([y_coords[0],y_coords[-1]])
                    # Keep each segment of each line
                    for i in range(len(x_coords) - 1): 
                        x_datas.append([x_coords[i], x_coords[i+1]])
                        y_datas.append([y_coords[i], y_coords[i+1]])
                    # polygons.append(line_string.coords.reshape(-1))
                x_datas, y_datas = np.array(x_datas), np.array(y_datas)
                x_datas_SEpoints, y_datas_SEpoints = np.array(x_datas_SEpoints), np.array(y_datas_SEpoints) 
            elif isinstance(bboxes, torch.Tensor):
                if bboxes.dim() ==2:
                    bboxes = bboxes.reshape(bboxes.shape[0],-1,2)
                # Get start points (every point excluding the last one in each line)
                # The shape will be [num_lines, num_points - 1, 2]
                start_points = bboxes[:,:-1,:]
                # Get end points (every point excluding the first one in each line)
                # The shape will be [num_lines, num_points - 1, 2=xy]
                end_points = bboxes[:,1:,:]
                # Combine start and end points
                line_segments = torch.stack((start_points, end_points),dim=2) #x, [num_lines,num_points-1,2,2] 
                x_datas = line_segments[...,0].numpy() #x, [num_lines,num_points-1,2=xx]
                y_datas = line_segments[...,1].numpy() #y, [num_lines,num_points-1,2=yy]
            if self.only_plot_SEpoints:
                self.draw_lines(x_datas_SEpoints,y_datas_SEpoints,
                    colors=colors,
                    line_widths=self.line_width)
            else:
                self.draw_lines(x_datas,y_datas,
                    colors=colors,
                    line_widths=self.line_width)
            #-------zhou
            if isinstance(bboxes, LineStringsOnImage):
                positions = np.concatenate((x_datas_SEpoints[:,0].reshape(-1, 1),y_datas_SEpoints[:,0].reshape(-1, 1)),axis=-1) + self.line_width
                scales = [20.0]*x_datas_SEpoints.shape[0]
            else:
                positions = np.concatenate((x_datas[:,0,0].reshape(-1, 1),y_datas[:,0,0].reshape(-1, 1)),axis=-1) + self.line_width
                scales = [20.0]*x_datas.shape[0]
            #不限制一下，text的右边会超过图像边缘
            positions[:, 0] = np.clip(positions[:, 0], 0, self.width - 200)
            positions[:, 1] = np.clip(positions[:, 1], 0, self.height - 50)

            for i, (pos, label) in enumerate(zip(positions, labels)):
                label_text = classes[
                    label] if classes is not None else f'class {label}'
                if 'line_scores' in instances:
                    score = round(float(instances.line_scores[i]) * 100, 1)
                    label_text += f': {score}'

                self.draw_texts(
                    label_text,
                    pos,
                    colors=text_colors[i],
                    font_sizes=int(scales[i]),
                    bboxes=[{
                        'facecolor': 'black',
                        'alpha': 0.8,
                        'pad': 0.7,
                        'edgecolor': 'none'
                    }])
        return self.get_image()
    # copy from mmdet/visualization/local_visualizer.py and change
    @master_only
    def add_datasample(
            self,
            name: str,
            image: np.ndarray,
            data_sample: Optional['DetDataSample'] = None,
            draw_gt: bool = True,
            draw_pred: bool = True,
            show: bool = False,
            wait_time: float = 0,
            # TODO: Supported in mmengine's Viusalizer.
            out_file: Optional[str] = None,
            pred_score_thr: Union[List[float],float] = 0.3,
            step: int = 0) -> None:
        """Draw datasample and save to all backends.

        - If GT and prediction are plotted at the same time, they are
        displayed in a stitched image where the left image is the
        ground truth and the right image is the prediction.
        - If ``show`` is True, all storage backends are ignored, and
        the images will be displayed in a local window.
        - If ``out_file`` is specified, the drawn image will be
        saved to ``out_file``. t is usually used when the display
        is not available.

        Args:
            name (str): The image identifier.
            image (np.ndarray): The image to draw.
            data_sample (:obj:`DetDataSample`, optional): A data
                sample that contain annotations and predictions.
                Defaults to None.
            draw_gt (bool): Whether to draw GT DetDataSample. Default to True.
            draw_pred (bool): Whether to draw Prediction DetDataSample.
                Defaults to True.
            show (bool): Whether to display the drawn image. Default to False.
            wait_time (float): The interval of show (s). Defaults to 0.
            out_file (str): Path to output file. Defaults to None.
            pred_score_thr (float): The threshold to visualize the bboxes
                and masks. Defaults to 0.3.
            step (int): Global step value to record. Defaults to 0.
        """
        image = image.clip(0, 255).astype(np.uint8)
        classes = self.dataset_meta.get('classes', None)
        line_classes = self.dataset_meta.get('line_classes', None)
        palette = self.dataset_meta.get('palette', None)

        gt_img_data = None
        pred_img_data = None

        if data_sample is not None:
            data_sample = data_sample.cpu()

        if draw_gt and data_sample is not None:
            gt_img_data = image
            if 'gt_instances' in data_sample:
                #------------zhou--------------
                if data_sample.gt_instances.bboxes is not None and isinstance(data_sample.gt_instances.bboxes, BaseBoxes):
                    data_sample.gt_instances.bboxes = data_sample.gt_instances.bboxes.tensor
                #------------zhou--------------
                gt_img_data = self._draw_instances(image,
                                                   data_sample.gt_instances,
                                                   classes, palette)
            #------------zhou--------------
            if 'gt_line_instances' in data_sample and (data_sample.gt_line_instances.line_labels.shape[0] !=0):
                gt_img_data = self._draw_line_instances(gt_img_data,
                                                   data_sample.gt_line_instances,
                                                   line_classes, palette[len(classes):])
            #------------zhou--------------
            if 'gt_panoptic_seg' in data_sample:
                assert classes is not None, 'class information is ' \
                                            'not provided when ' \
                                            'visualizing panoptic ' \
                                            'segmentation results.'
                gt_img_data = self._draw_panoptic_seg(
                    gt_img_data, data_sample.gt_panoptic_seg, classes)

        if draw_pred and data_sample is not None:
            pred_img_data = image
            if 'pred_instances' in data_sample:
                pred_instances = data_sample.pred_instances
                pred_instances = pred_instances[
                    pred_instances.scores > pred_score_thr[0]] #----------zhou
                #------------zhou--------------
                if pred_instances.bboxes is not None and isinstance(pred_instances.bboxes, BaseBoxes):
                    pred_instances.bboxes = pred_instances.bboxes.tensor
                #------------zhou--------------
                pred_img_data = self._draw_instances(image, pred_instances,
                                                     classes, palette)
            #------------zhou--------------
            if 'line_pred_instances' in data_sample and (data_sample.line_pred_instances.line_labels.shape[0] !=0):
                line_pred_instances = data_sample.line_pred_instances
                # print('line_pred_instances.line_scores',line_pred_instances.line_scores[:4])
                # print('line_pred_instances.line_points',line_pred_instances.line_points[:4])
                line_pred_instances = line_pred_instances[
                    line_pred_instances.line_scores > pred_score_thr[1]] #----------zhou
                if len(line_pred_instances) !=0:
                    pred_img_data = self._draw_line_instances(pred_img_data,
                                                       line_pred_instances,
                                                       line_classes, palette[len(classes):])
            #------------zhou--------------
            if 'pred_panoptic_seg' in data_sample:
                assert classes is not None, 'class information is ' \
                                            'not provided when ' \
                                            'visualizing panoptic ' \
                                            'segmentation results.'
                pred_img_data = self._draw_panoptic_seg(
                    pred_img_data, data_sample.pred_panoptic_seg.numpy(),
                    classes)

        if gt_img_data is not None and pred_img_data is not None:
            drawn_img = np.concatenate((gt_img_data, pred_img_data), axis=1)
        elif gt_img_data is not None:
            drawn_img = gt_img_data
        elif pred_img_data is not None:
            drawn_img = pred_img_data
        else:
            # Display the original image directly if nothing is drawn.
            drawn_img = image

        # It is convenient for users to obtain the drawn image.
        # For example, the user wants to obtain the drawn image and
        # save it as a video during video inference.
        self.set_image(drawn_img)

        if show:
            self.show(drawn_img, win_name=name, wait_time=wait_time)

        if out_file is not None:
            mmcv.imwrite(drawn_img[..., ::-1], out_file)
        else:
            self.add_image(name, drawn_img, step)
