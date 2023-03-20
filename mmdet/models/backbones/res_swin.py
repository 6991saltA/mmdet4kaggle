""" Swin Transformer
A PyTorch impl of : `Swin Transformer: Hierarchical Vision Transformer using Shifted Windows`
    - https://arxiv.org/pdf/2103.14030

Code/weights from https://github.com/microsoft/Swin-Transformer

"""

import torch
import torch.nn as nn
from ..builder import BACKBONES
from .resnet import ResNet
from .swin import SwinTransformer



@BACKBONES.register_module()
class ResSwinNet(nn.Module):

    def __init__(self, resnet_depth, resnet_num_stages, resnet_out_indices,resnet_frozen_stages,
                 resnet_norm_cfg, resnet_norm_eval, resnet_style, resnet_init_cfg,
                 swinT_embed_dims, swinT_depths, swinT_num_heads, swinT_window_size,
                 swinT_mlp_ratio,swinT_qkv_bias, swinT_qk_scale, swinT_drop_rate,
                 swinT_attn_drop_rate, swinT_drop_path_rate, swinT_patch_norm,
                 swinT_out_indices,swinT_with_cp, swinT_convert_weights, swinT_init_cfg):
        super().__init__()

        self.swinT_out_channels = [96, 192, 384, 768]
        self.resnet_out_channels = [256, 512, 1024, 2048]

        self.conv_list = nn.ModuleList()
        for layer_index in range(len(self.swinT_out_channels)):
            self.conv_list.append(nn.Conv2d(self.swinT_out_channels[layer_index], self.resnet_out_channels[layer_index], kernel_size=1, stride=1))

        self.resnet_backbone = ResNet(depth=resnet_depth, num_stages=resnet_num_stages, out_indices=resnet_out_indices,
                                      frozen_stages=resnet_frozen_stages,norm_cfg=resnet_norm_cfg,
                                      norm_eval=resnet_norm_eval, style=resnet_style, init_cfg=resnet_init_cfg)

        self.swinT_backbone = SwinTransformer(embed_dims=swinT_embed_dims, depths=swinT_depths,
                                         num_heads=swinT_num_heads, window_size=swinT_window_size,
                                         mlp_ratio=swinT_mlp_ratio, qkv_bias=swinT_qkv_bias,
                                         qk_scale=swinT_qk_scale, drop_rate=swinT_drop_rate,
                                         attn_drop_rate=swinT_attn_drop_rate,
                                         drop_path_rate=swinT_drop_path_rate,
                                         patch_norm=swinT_patch_norm, out_indices=swinT_out_indices,
                                         with_cp=swinT_with_cp, convert_weights=swinT_convert_weights,
                                         init_cfg=swinT_init_cfg)

    def forward(self, x):
        outList = []

        x_resnet = self.resnet_backbone(x)
        x_swinT = self.swinT_backbone(x)

        x_resnet2list = list(x_resnet)
        for conv_layer_index in range(len(self.conv_list)):
            x_swinT[conv_layer_index] = self.conv_list[conv_layer_index](x_swinT[conv_layer_index])

        for index in range(len(x_swinT)):
            outList.append(x_resnet2list[index] + x_swinT[index])

        return tuple(outList)

# self.resnet_depth = resnet_depth
# self.resnet_num_stages = resnet_num_stages
# self.resnet_out_indices = resnet_out_indices
# self.resnet_frozen_stages = resnet_frozen_stages
# self.resnet_norm_cfg = resnet_norm_cfg
# self.resnet_norm_eval = resnet_norm_eval
# self.resnet_style = resnet_style
# self.swinT_embed_dims = swinT_embed_dims
# self.swinT_depths = swinT_depths
# self.swinT_num_heads = swinT_num_heads
# self.swinT_window_size = swinT_window_size
# self.swinT_mlp_ratio = swinT_mlp_ratio
# self.swinT_qkv_bias = swinT_qkv_bias
# self.swinT_qk_scale = swinT_qk_scale
# self.swinT_drop_rate = swinT_drop_rate
# self.swinT_attn_drop_rate = swinT_attn_drop_rate
# self.swinT_drop_path_rate = swinT_drop_path_rate
# self.swinT_patch_norm = swinT_patch_norm
# self.swinT_out_indices = swinT_out_indices
# self.swinT_with_cp = swinT_with_cp
# self.swinT_convert_weights = swinT_convert_weights

# def res_swin_Backbone(resnet_pretrained='./pretrained/ressnet50.pth',
#                         swinT_pretrained='./pretrained/fasterrcnn_resnet50_fpn_coco.pth'):
