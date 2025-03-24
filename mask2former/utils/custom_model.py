
from typing import List, Optional
import random
import numpy as np
import torch
from torch import Tensor, nn
from transformers import Mask2FormerConfig, Mask2FormerModel, Mask2FormerForUniversalSegmentation
from transformers.models.mask2former.modeling_mask2former import Mask2FormerPixelLevelModule, \
    Mask2FormerForUniversalSegmentationOutput, Mask2FormerPixelLevelModuleOutput
from transformers.utils.backbone_utils import load_backbone
from mask2former.utils.data_process import calculate_depth_histogram, select_depth_distribution_modes, \
    define_depth_interval_windows, generate_depth_region_masks


# 固定随机种子
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class CustomConfig(Mask2FormerConfig):
    model_type = "mask2former"

    def __init__(self, attribute=1, **kwargs):
        print("[CustomConfig] constructing...")
        self.attribute = attribute
        super().__init__(**kwargs)


class CustomMask2FormerModel(Mask2FormerModel):
    main_input_name = "pixel_values"
    def __init__(self, config, rgb_d=False):
        print("[CustomMask2FormerModel] constructing...")
        super().__init__(config)
        self.pixel_level_module = CustomMask2FormerPixelLevelModule(config, rgb_d=rgb_d)


class CustomMask2FormerForUniversalSegmentation(Mask2FormerForUniversalSegmentation):
    main_input_name = "pixel_values"
    config_class = CustomConfig

    def __init__(self, config, rgb_d=False):
        print("[CustomMask2FormerForUniversalSegmentation] constructing...")
        super().__init__(config)
        set_seed(42)
        self.model = CustomMask2FormerModel(config, rgb_d=rgb_d)


class CustomMask2FormerPixelLevelModule(Mask2FormerPixelLevelModule):
    main_input_name = "pixel_values"
    def __init__(self, config, rgb_d=False):
        print("[CustomMask2FormerPixelLevelModule] constructing...")
        super().__init__(config)
        self.rgb_d = rgb_d
        print(f"[CustomMask2FormerPixelLevelModule] rgb_d={self.rgb_d}")
        if self.rgb_d: # RGB-D
            self.depth_encoder = load_backbone(config)
            self.feature_fuser = FeatureFuser()
        self.color_encoder = load_backbone(config)

    def forward(self, pixel_values: Tensor, output_hidden_states: bool = False) -> Mask2FormerPixelLevelModuleOutput:
        if self.rgb_d: # RGB-D
            # TODO: 数据格式扩充：(1, 6, H, W) -> (1, 30, H, W) [RGB, DECIMATION, RS, SPATIAL, HOLE_FILLING, AHE, LAPLACE, GAUSSIAN, EQ, LT]
            rgb = pixel_values[:, 0:3, :, :]
            decimation = pixel_values[:, 3:6, :, :]
            rs = pixel_values[:, 6:9, :, :]
            spatial = pixel_values[:, 9:12, :, :]
            hole_filling = pixel_values[:, 12:15, :, :]
            ahe = pixel_values[:, 15:18, :, :]
            laplace = pixel_values[:, 18:21, :, :]
            gaussian = pixel_values[:, 21:24, :, :]
            eq = pixel_values[:, 24:27, :, :]
            lt = pixel_values[:, 27:30, :, :]

            color_pixel_values = pixel_values[:, :3, :, :]
            depth_pixel_values = pixel_values[:, 3:6, :, :]
            color_feature_map = self.color_encoder(color_pixel_values).feature_maps
            depth_feature_map = self.depth_encoder(depth_pixel_values).feature_maps
            backbone_features = self.feature_fuser(color_feature_map, depth_feature_map)
        else: # RGB only
            backbone_features = self.color_encoder(pixel_values).feature_maps

        decoder_output = self.decoder(backbone_features, output_hidden_states=output_hidden_states)

        return Mask2FormerPixelLevelModuleOutput(
            encoder_last_hidden_state=backbone_features[-1],
            encoder_hidden_states=tuple(backbone_features) if output_hidden_states else None,
            decoder_last_hidden_state=decoder_output.mask_features,
            decoder_hidden_states=decoder_output.multi_scale_features,
        )


class FeatureFuser(nn.Module):
    def __init__(self):
        super().__init__()
        self.fuse_conv = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(192, 96, 1),
                nn.ReLU()
            ),
            nn.Sequential(
                nn.Conv2d(384, 192, 1),
                nn.ReLU()
            ),
            nn.Sequential(
                nn.Conv2d(768, 384, 1),
                nn.ReLU()
            ),
            nn.Sequential(
                nn.Conv2d(1536, 768, 1),
                nn.ReLU()
            )
        ])

    def forward(self, color_feature_map, depth_feature_map):
        assert len(color_feature_map) == len(depth_feature_map), \
            "the tuple length of color_feature_map and depth_feature_map should be the same"
        size_list = [
            torch.Size([1, 96, 64, 64]),
            torch.Size([1, 192, 32, 32]),
            torch.Size([1, 384, 16, 16]),
            torch.Size([1, 768, 8, 8])
        ]
        # TODO: DEBUG this assert
        # for c, d, size in zip(color_feature_map, depth_feature_map, size_list):
        #     assert c.shape == d.shape == size, \
        #         (f"the shape of color_feature_map and depth_feature_map should be the same, and both of them should be equal to the {size}."
        #          f"But color_feature_map.shape == {c.shape}, depth_feature_map.shape == {d.shape}")

        # merged_map = [torch.cat([color_feature_map[i], depth_feature_map[i]], dim=1) for i in range(len(depth_feature_map))]
        # fused_map = [self.fuse_conv[i](merged_map[i]) for i in range(len(merged_map))]
        merged_map = [torch.cat([c, d], dim=1) for c, d in zip(color_feature_map, depth_feature_map)]
        fused_map = [conv(m) for conv, m in zip(self.fuse_conv, merged_map)]

        return fused_map


class DSAModule(nn.Module):
    def __init__(self, in_channels, out_channels, num_depth_regions=3):
        """
        深度敏感注意力模块 (DSAM)。

        Args:
            in_channels (int): 输入 RGB 特征图的通道数.
            out_channels (int): 输出增强 RGB 特征图的通道数.
            num_depth_regions (int): 深度分解的区域数量 (T).  实际子分支数量为 T+1 (包含剩余区域). 默认 3.
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_depth_regions = num_depth_regions
        self.conv_layers = nn.ModuleList([
            nn.Conv2d(in_channels, out_channels, kernel_size=1) for _ in range(num_depth_regions + 1)
        ]) # T+1 个 1x1 卷积层

    def forward(self, rgb_features, depth_map):
        """
        DSAM 的前向传播过程.

        Args:
            rgb_features (torch.Tensor): 输入 RGB 特征图, 形状为 (B, C_in, H, W).
            depth_map (torch.Tensor): 输入原始深度图, 形状为 (B, 1, H_d, W_d) 或 (B, H_d, W_d) 或 (H_d, W_d)  (单通道).
                                      注意：为了代码的通用性，函数内部会处理不同形状的深度图.

        Returns:
            torch.Tensor: 增强后的 RGB 特征图, 形状为 (B, C_out, H, W).
        """
        # 1. 深度分解 (Depth Decomposition)
        # 确保深度图是 NumPy 数组且为单通道 (如果输入是 Tensor，先转为 NumPy)
        if isinstance(depth_map, torch.Tensor):
            depth_map_np = depth_map.squeeze().cpu().detach().numpy()  # 去除通道维度，转为 NumPy, 放到 CPU
        elif isinstance(depth_map, np.ndarray):
            depth_map_np = depth_map.squeeze() # 确保是单通道
        else:
            raise TypeError("Depth map must be torch.Tensor or numpy.ndarray")

        interval_windows = [] # 初始化为空列表，防止在没有检测到 depth_modes 时报错
        region_masks = []

        hist, bin_edges = calculate_depth_histogram(depth_map_np)
        depth_modes = select_depth_distribution_modes(hist, bin_edges, num_modes=self.num_depth_regions)
        if depth_modes: # 只有当检测到 depth_modes 时才进行后续步骤，防止空列表导致错误
            interval_windows = define_depth_interval_windows(depth_modes)
            region_masks = generate_depth_region_masks(depth_map_np, interval_windows)
        else:
            # 如果没有检测到深度模式，则创建一个全零的掩码列表，保证程序正常运行，但不进行深度引导
            region_masks = [np.zeros_like(depth_map_np, dtype=bool)] * (self.num_depth_regions + 1)


        # 2. 深度敏感注意力 (Depth-Sensitive Attention)
        enhanced_features = 0
        for i in range(len(region_masks)):
            # 将 NumPy mask 转换为 PyTorch Tensor, 并放到与 rgb_features 相同的设备上
            mask_tensor = torch.from_numpy(region_masks[i]).float().unsqueeze(0).unsqueeze(0).to(rgb_features.device) # (1, 1, H_d, W_d)
            # resize mask to match rgb_features' spatial size using adaptive max pooling
            resized_mask = nn.functional.adaptive_max_pool2d(mask_tensor, rgb_features.shape[2:]) # (1, 1, H, W)

            masked_features = rgb_features * resized_mask  # 元素级乘法 (B, C_in, H, W) * (1, 1, H, W)  -> (B, C_in, H, W)
            refined_features = self.conv_layers[i](masked_features) # 1x1 卷积 (B, C_in, H, W) -> (B, C_out, H, W)
            enhanced_features += refined_features # 元素级求和


        output_features = enhanced_features + rgb_features  # 残差连接
        return output_features


# "The features from different modalities of the same scale are always fused,
# while features in different scales are selectively fused."
# (来自相同尺度的不同模态特征总是被融合，而不同尺度的特征则有选择地融合)。