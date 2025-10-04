# Copyright (c) OpenMMLab. All rights reserved.
#---------------copy and change from: mmyolo/models/plugins/cbam.py
import torch
from mmdet.utils import OptMultiConfig
from mmengine.model import BaseModule

from mmyolo.registry import MODELS
from .cbam import ChannelAttention,SpatialAttention 

@MODELS.register_module()
class PARACBAM(BaseModule):
    """Convolutional Block Attention Module. arxiv link:
    https://arxiv.org/abs/1807.06521v2.

    Args:
        in_channels (int): The input (and output) channels of the CBAM.
        reduce_ratio (int): Squeeze ratio in ChannelAttention, the intermediate
            channel will be ``int(channels/ratio)``. Defaults to 16.
        kernel_size (int): The size of the convolution kernel in
            SpatialAttention. Defaults to 7.
        act_cfg (dict): Config dict for activation layer in ChannelAttention
            Defaults to dict(type='ReLU').
        init_cfg (dict or list[dict], optional): Initialization config dict.
            Defaults to None.
    """

    def __init__(self,
                 in_channels: int,
                 reduce_ratio: int = 16,
                 kernel_size: int = 7,
                 act_cfg: dict = dict(type='ReLU'),
                 init_cfg: OptMultiConfig = None):
        super().__init__(init_cfg)
        self.channel_attention = ChannelAttention(
            channels=in_channels, reduce_ratio=reduce_ratio, act_cfg=act_cfg)

        self.spatial_attention = SpatialAttention(kernel_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward function."""
        out = self.channel_attention(x) * x + self.spatial_attention(x) * x
        return out
