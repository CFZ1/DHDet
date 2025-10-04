# Copy and change from mmdetection_v3d3d0/mmdet/models/necks/fpn.py
from typing import List, Tuple, Union

import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule
from mmengine.model import BaseModule
from torch import Tensor

from mmdet.registry import MODELS
from mmdet.utils import ConfigType, MultiConfig, OptConfigType
import torch
import numpy as np
# ref cbam.SpatialAttention
class IRIM(BaseModule):
    """
    """
    def __init__(self, in_channels, out_channels):
        super(IRIM, self).__init__()
        # 1x1 convolution for channel reduction
        self.conv1x1 = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        # 3x3 convolution for the spatial attention map
        self.conv3x3 = nn.Conv2d(2, 1, kernel_size=3, padding=1)

    def forward(self, x):
        # Max Pooling and Avg Pooling along the channel dimension
        avg_out = torch.mean(x, dim=1, keepdim=True)   # bs 1 x H x W
        max_out, _ = torch.max(x, dim=1, keepdim=True) # 1 x H x W
        pooled = torch.cat([avg_out, max_out], dim=1)     # 2 x H x W
        # Convolution and Softmax for spatial attention map
        attention_map = self.conv3x3(pooled)  # 1 x H x W
        # Softmax across spatial dimensions (h, w)
        attention_map = F.softmax(attention_map.view(x.size(0), -1), dim=-1) 
        attention_map = attention_map.view(x.size(0), 1, x.size(2), x.size(3))

        # 1x1 convolution to reduce channel dimension
        x_prime = self.conv1x1(x)  # C' x H x W
        
        # Apply spatial attention map
        y = x_prime * attention_map  # Element-wise multiplication
        
        return y

'''
使用einsum进行计算,避免了不必要的permute和reshape操作, 
'''
class OptimizedISM(nn.Module):
    def __init__(self, upsample_cfg: ConfigType = dict(mode='nearest')):
        super(OptimizedISM, self).__init__()

        self.upsample_cfg = upsample_cfg.copy()
        self.temperature = 16.0
        
    def forward(self, D, S): #
        # D: Deep features [batch_size, C, H/2, W/2]
        # S: Shallow features [batch_size, C, H, W]
        # Upsample D to match the size of S, interpolate-->因为在采样过程中还真不能保证下层特征是上层特征的1/2
        prev_shape = S.shape[2:]
        D_prime = F.interpolate(D, size=prev_shape, **self.upsample_cfg)
        # D_prime = self.upsample(D)  # [batch_size, C, H, W]
        
        # Calculate the similarity matrix
        similarity_matrix = torch.einsum('bchw,bcij->bhwij', D_prime, S)
        if self.temperature is None:
            self.temperature = np.power(S.shape[-1]*S.shape[-2], 0.5) #ref https://github.com/icoz69/CEC-CVPR2021/blob/main/models/cec/Network.py
        # Apply SoftMax to normalize the similarity matrix
        similarity_matrix = F.softmax(similarity_matrix / self.temperature, dim=-1)
        
        # Apply the similarity matrix to the shallow features
        Z = torch.einsum('bhwij,bcij->bchw', similarity_matrix, S)
        
        # Element-wise addition of Z and D_prime
        Z.add_(D_prime)  # In-place addition # [batch_size, C, H, W]
        
        return Z
    
@MODELS.register_module()
class IEFPN(BaseModule):

    def __init__(
        self,
        in_channels: List[int],
        out_channels: int,
        use_irim: List[bool]=[True,True,True,True],  # 每一层是否使用 IRIM
        use_ism: List[bool]=[False,True,True],   # 每一层是否使用 ISM
        no_norm_on_lateral: bool = False,
        conv_cfg: OptConfigType = None,
        norm_cfg: OptConfigType = None,
        act_cfg: OptConfigType = None,
        upsample_cfg: ConfigType = dict(mode='nearest'),
        init_cfg: MultiConfig = dict(
            type='Xavier', layer='Conv2d', distribution='uniform')
    ) -> None:
        super().__init__(init_cfg=init_cfg)
        assert isinstance(in_channels, list)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.use_irim = use_irim
        self.use_ism = use_ism
        self.fp16_enabled = False
        self.no_norm_on_lateral = no_norm_on_lateral
        self.upsample_cfg = upsample_cfg.copy()
        self.start_level = 0
        self.backbone_end_level = 4

        self.lateral_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()

        for i in range(self.start_level, self.backbone_end_level):
            if self.use_irim[i]:
                l_conv = IRIM(in_channels[i], out_channels)
            else:
                l_conv = ConvModule(
                    in_channels[i],
                    out_channels,
                    1,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg if not self.no_norm_on_lateral else None,
                    act_cfg=act_cfg,
                    inplace=False) #-----------------------------change 1
            fpn_conv = ConvModule(
                out_channels,
                out_channels,
                3,
                padding=1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg,
                inplace=False)
            self.lateral_convs.append(l_conv)
            self.fpn_convs.append(fpn_conv)
        # 构建ISM模块，仅适用于上层和中间层之间
        self.up_convs = nn.ModuleList()
        for i in range(self.backbone_end_level - 1):
            if self.use_ism[i]:
                self.up_convs.append(OptimizedISM(upsample_cfg))
            else:
                self.up_convs.append(None)  # 如果不使用ISM，保持None

    def forward(self, inputs: Tuple[Tensor]) -> tuple:
        """Forward function.

        Args:
            inputs (tuple[Tensor]): Features from the upstream network, each
                is a 4D-tensor.

        Returns:
            tuple: Feature maps, each is a 4D-tensor.
        """
        assert len(inputs) == len(self.in_channels)

        # build laterals
        laterals = [
            lateral_conv(inputs[i + self.start_level])
            for i, lateral_conv in enumerate(self.lateral_convs)
        ] #low-->high

        # build top-down path，应用ISM进行层间特征融合，最后一层使用简单的上采样
        used_backbone_levels = len(laterals)
        for i in range(used_backbone_levels - 1, 0, -1): #3,2,1
            if self.up_convs[i - 1] is not None:  # 使用ISM
                laterals[i - 1] = self.up_convs[i - 1](laterals[i], laterals[i - 1])
            else:  # 使用简单的上采样和相加
                prev_shape = laterals[i - 1].shape[2:]
                laterals[i - 1] = laterals[i - 1] + F.interpolate(
                    laterals[i], size=prev_shape, **self.upsample_cfg)  
        
        # build outputs
        # part 1: from original levels, 和原始的FPN一样
        outs = [
            self.fpn_convs[i](laterals[i]) for i in range(used_backbone_levels)
        ]

        return tuple(outs)
# # upsample_cfg and interpolate ref fpn.FPN
# class ISM(nn.Module):
#     def __init__(self, upsample_cfg: ConfigType = dict(mode='nearest')):
#         super(ISM, self).__init__()

#         self.upsample_cfg = upsample_cfg.copy()
        
#     def forward(self, D, S): #
#         # D: Deep features [batch_size, C, H/2, W/2]
#         # S: Shallow features [batch_size, C, H, W]
        
#         # Upsample D to match the size of S, interpolate-->因为在采样过程中还真不能保证下层特征是上层特征的1/2
#         prev_shape = S.shape[2:]
#         D_prime = F.interpolate(D, size=prev_shape, **self.upsample_cfg)
#         # D_prime = self.upsample(D)  # [batch_size, C, H, W]
        
#         # Flatten D' and S
#         batch_size, C, H, W = D_prime.shape
#         D_prime_flat = D_prime.view(batch_size, C, -1)  # [batch_size, C, H*W]
#         S_flat = S.view(batch_size, C, -1)  # [batch_size, C, H*W]
#         # Transpose D_prime_flat for matrix multiplication
#         D_prime_flat_T = D_prime_flat.permute(0, 2, 1)  # [batch_size, H*W, C]
#         # Calculate the similarity matrix
#         similarity_matrix = torch.bmm(D_prime_flat_T, S_flat)  # [batch_size, D-H*W, S-H*W]
        
#         # Apply SoftMax to normalize the similarity matrix
#         similarity_matrix = F.softmax(similarity_matrix, dim=-1)
        
#         # Apply the similarity matrix to the shallow features
#         Z_flat = torch.bmm(similarity_matrix, S_flat.permute(0, 2, 1))  # [batch_size, D-H*W, C]
#         Z = Z_flat.permute(0, 2, 1).view(batch_size, C, H, W)  # Reshape back to [batch_size, C, H, W]
        
#         # Element-wise addition of Z and D_prime
#         Y = Z + D_prime  # [batch_size, C, H, W]
        
#         return Y
