import random
import numpy as np
import torch
import cv2
from torch import Tensor, nn
from transformers import Mask2FormerConfig, Mask2FormerModel, Mask2FormerForUniversalSegmentation, Mask2FormerImageProcessor
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
    def __init__(self, config, rgb_d):
        print("[CustomMask2FormerModel] constructing...")
        super().__init__(config)
        self.pixel_level_module = CustomMask2FormerPixelLevelModule(config, rgb_d=rgb_d)


class CustomMask2FormerForUniversalSegmentation(Mask2FormerForUniversalSegmentation):
    main_input_name = "pixel_values"
    config_class = CustomConfig

    def __init__(self, config, rgb_d='single'):
        print("[CustomMask2FormerForUniversalSegmentation] constructing...")
        super().__init__(config)
        set_seed(42)
        self.model = CustomMask2FormerModel(config, rgb_d=rgb_d)


class CustomMask2FormerPixelLevelModule(Mask2FormerPixelLevelModule):
    main_input_name = "pixel_values"
    def __init__(self, config, rgb_d):
        print("[CustomMask2FormerPixelLevelModule] constructing...")
        super().__init__(config)
        self.rgb_d = rgb_d
        print(f"[CustomMask2FormerPixelLevelModule] rgb_d={self.rgb_d}")
        if self.rgb_d == 'single': # RGB (3 channel)
            pass # use super().encoder

        elif self.rgb_d == 'multi': # RGB-D (6 channel)
            # self.depth_encoder = AutoBackbone.from_pretrained("microsoft/resnet-50")
            self.depth_encoder = load_backbone(config)
            self.feature_fuser = FeatureFuser()

        else: # RGB-D (30 channel)
            self.depth_encoder = load_backbone(config)
            self.feature_fuser = FeatureFuser()
            # self.dsam1 = DSAModule(in_channels=96, out_channels=96, num_depth_regions=3)
            # self.dsam2 = DSAModule(in_channels=96, out_channels=96, num_depth_regions=3)
            self.dsam3 = DSAModule(in_channels=96, out_channels=192, num_depth_regions=3)
            self.dsam4 = DSAModule(in_channels=192, out_channels=384, num_depth_regions=3)
            self.dsam5 = DSAModule(in_channels=384, out_channels=768, num_depth_regions=3)

    def forward(self, pixel_values: Tensor, output_hidden_states: bool = False) -> Mask2FormerPixelLevelModuleOutput:
        if self.rgb_d == 'single': # RGB (3 channel)
            backbone_features = self.encoder(pixel_values).feature_maps

        elif self.rgb_d == 'multi': # RGB-D (6 channel)
            rgb = pixel_values[:, 0:3, :, :]
            depth = pixel_values[:, 3:6, :, :]
            color_feature_map = self.encoder(rgb).feature_maps
            depth_feature_map = self.depth_encoder(depth).feature_maps
            backbone_features = self.feature_fuser(color_feature_map, depth_feature_map)

        else: # RGB-D (18 channel)
            # (batch, 21, H, W) [color_input, fused_img_1, fused_img_2, depth_input, ahe, laplace, gaussian]
            color_input = pixel_values[:, 0:3, :, :]
            fused_img_batch1 = pixel_values[:, 3:6, :, :]
            fused_img_batch2 = pixel_values[:, 6:9, :, :]
            depth_input = pixel_values[:, 9:12, :, :]
            # ahe = pixel_values[:, 12:15, :, :]
            # laplace = pixel_values[:, 15:18, :, :]
            # gaussian = pixel_values[:, 18:21, :, :]

            # rgb backbone
            rgb_backbone_features = self.encoder(color_input).feature_maps # torch.Tensor
            cp_rgb_backbone_features = list(rgb_backbone_features)

            # DSAM
            # dsam_output1 = [self.dsam1(i.unsqueeze(0), self.to_grayscale(j)) for i, j in zip(cp_rgb_backbone_features[0], ahe)] # [B, 96, 64, 64]
            # dsam_output1 = torch.stack(dsam_output1, dim=0).squeeze(1)  # [B, 96, 64, 64]
            # dsam_output2 = [self.dsam2(i.unsqueeze(0), self.to_grayscale(j)) for i, j in zip(dsam_output1, laplace)] # [B, 96, 64, 64]
            # dsam_output2 = torch.stack(dsam_output2, dim=0).squeeze(1)  # [B, 96, 64, 64]
            dsam_output3 = [self.dsam3(i.unsqueeze(0), self.to_grayscale(j)) for i, j in zip(cp_rgb_backbone_features[0], fused_img_batch1)] # [B, 96, 64, 64]
            dsam_output3 = torch.stack(dsam_output3, dim=0).squeeze(1)  # [B, 192, 32, 32]
            cp_rgb_backbone_features[1] += dsam_output3

            dsam_output4 = [self.dsam4(i.unsqueeze(0), self.to_grayscale(j)) for i, j in zip(cp_rgb_backbone_features[1], fused_img_batch1)] # [B, 192, 32, 32]
            dsam_output4 = torch.stack(dsam_output4, dim=0).squeeze(1)  # [B, 384, 16, 16]
            cp_rgb_backbone_features[2] += dsam_output4

            dsam_output5 = [self.dsam5(i.unsqueeze(0), self.to_grayscale(j)) for i, j in zip(rgb_backbone_features[2], fused_img_batch2)] # [B, 384, 16, 16]
            dsam_output5 = torch.stack(dsam_output5, dim=0).squeeze(1)  # [B, 768, 16, 16]
            cp_rgb_backbone_features[3] += dsam_output5

            depth_backbone_features = self.depth_encoder(depth_input).feature_maps
            cp_depth_backbone_features = list(depth_backbone_features)

            # combine the features of rgb and depth
            backbone_features = self.feature_fuser(cp_rgb_backbone_features, cp_depth_backbone_features)

        decoder_output = self.decoder(backbone_features, output_hidden_states=output_hidden_states)

        return Mask2FormerPixelLevelModuleOutput(
            encoder_last_hidden_state=backbone_features[-1],
            encoder_hidden_states=tuple(backbone_features) if output_hidden_states else None,
            decoder_last_hidden_state=decoder_output.mask_features,
            decoder_hidden_states=decoder_output.multi_scale_features,
        )

    def to_grayscale(self, image_data):
        """
        将输入的图片数据（NumPy ndarray 或 PyTorch Tensor）转换为单通道灰度图像。

        Args:
            image_data (numpy.ndarray or torch.Tensor): 输入的图片数据。
                可以是彩色 (RGB) 或灰度图像。
                - NumPy ndarray: 形状可以是 (H, W, 3), (H, W, 1), (H, W) 或 (C, H, W) where C can be 1 or 3.
                - PyTorch Tensor: 形状可以是 (C, H, W), (B, C, H, W) 或 **(H, W, 3)** where C can be 1 or 3. **新增 (H, W, 3) 支持**

        Returns:
            numpy.ndarray or torch.Tensor: 单通道灰度图像，类型与输入相同。
                - NumPy ndarray: 形状 (H, W).
                - PyTorch Tensor: 形状 (1, H, W).

        Raises:
            TypeError: 如果输入数据既不是 NumPy ndarray 也不是 PyTorch Tensor。
            ValueError: 如果输入图像不是灰度或彩色图像 (通道数不是 1 或 3)。
        """
        if isinstance(image_data, np.ndarray):
            # NumPy ndarray input (保持不变)
            ndim = image_data.ndim
            if ndim == 3:
                shape = image_data.shape
                if shape[-1] == 3:  # (H, W, 3) or (C, H, W) if first dim is channel
                    # Assume (H, W, 3) format for ndarray as more common for images
                    rgb_image = image_data
                    r_channel = rgb_image[:, :, 0]
                    g_channel = rgb_image[:, :, 1]
                    b_channel = rgb_image[:, :, 2]
                    grayscale_image = 0.299 * r_channel + 0.587 * g_channel + 0.114 * b_channel
                elif shape[0] == 3:  # (3, H, W) format
                    rgb_image = image_data
                    r_channel = rgb_image[0, :, :]
                    g_channel = rgb_image[1, :, :]
                    b_channel = rgb_image[2, :, :]
                    grayscale_image = 0.299 * r_channel + 0.587 * g_channel + 0.114 * b_channel
                elif shape[-1] == 1 or shape[
                    0] == 1:  # (H, W, 1) or (1, H, W) - already grayscale, just squeeze channel dim
                    grayscale_image = image_data.squeeze()
                else:
                    raise ValueError(
                        "Input NumPy ndarray image should be RGB (H, W, 3) or (C, H, W) with C=3, or grayscale (H, W, 1), (1, H, W) or (H, W).")
            elif ndim == 2:  # (H, W) - already grayscale
                grayscale_image = image_data
            else:
                raise ValueError("Input NumPy ndarray image should be 2D (H, W) or 3D (H, W, C) or (C, H, W).")

            return grayscale_image.astype(image_data.dtype)  # Keep original dtype

        elif isinstance(image_data, torch.Tensor):
            # PyTorch Tensor input (修改部分)
            ndim = image_data.ndim
            if ndim == 4:  # (B, C, H, W)
                channels = image_data.shape[1]  # Channels is at dimension 1
                is_channel_first = True  # Assume (B, C, H, W)
            elif ndim == 3:
                shape = image_data.shape
                if shape[0] == 3:  # (C, H, W)
                    channels = shape[0]  # Channels is at dimension 0
                    is_channel_first = True
                elif shape[-1] == 3:  # **(H, W, 3)**  新增 (H, W, 3) 形状处理
                    channels = shape[-1]  # Channels is at dimension -1
                    is_channel_first = False  # Indicate (H, W, 3) format
                elif shape[0] == 1 or shape[-1] == 1:  # (1, H, W) or (H, W, 1) - grayscale
                    channels = 1
                    is_channel_first = shape[0] == 1  # Check if channel is first dimension
                else:
                    raise ValueError("Input PyTorch Tensor image with ndim=3, but cannot determine channel location.")

            else:
                raise ValueError(
                    "Input PyTorch Tensor image should be 3D (C, H, W) or (H, W, 3) or 4D (B, C, H, W).")  # Modified error message

            if channels == 3:  # RGB image
                r_weight = 0.299
                g_weight = 0.587
                b_weight = 0.114
                if ndim == 3:  # (C, H, W) or (H, W, 3)
                    if is_channel_first:  # (C, H, W)
                        r_channel = image_data[0, :, :]
                        g_channel = image_data[1, :, :]
                        b_channel = image_data[2, :, :]
                    else:  # **(H, W, 3)**  处理 (H, W, 3) 形状
                        r_channel = image_data[:, :, 0]
                        g_channel = image_data[:, :, 1]
                        b_channel = image_data[:, :, 2]
                    grayscale_image = r_weight * r_channel + g_weight * g_channel + b_weight * b_channel
                    grayscale_image = grayscale_image.unsqueeze(0)  # (1, H, W)
                else:  # (B, C, H, W) -  (B, C, H, W) format remains unchanged
                    grayscale_image_batch = r_weight * image_data[:, 0, :, :] + \
                                            g_weight * image_data[:, 1, :, :] + \
                                            b_weight * image_data[:, 2, :, :]
                    grayscale_image = grayscale_image_batch.unsqueeze(1)  # (B, 1, H, W)

            elif channels == 1:  # Already grayscale
                if ndim == 3:  # (C, H, W) or (H, W, 1)
                    if is_channel_first:
                        grayscale_image = image_data  # (1, H, W) - already (1, H, W)
                    else:  # (H, W, 1)
                        grayscale_image = image_data.permute(2, 0, 1)  # Convert (H, W, 1) to (1, H, W)
                else:  # (B, 1, H, W)
                    grayscale_image = image_data  # (B, 1, H, W) - already (B, 1, H, W)
            else:
                raise ValueError(
                    f"Input PyTorch Tensor image should be grayscale (1 channel) or RGB (3 channels), but has {channels} channels.")

            return grayscale_image.to(image_data.dtype)  # Keep original dtype

        else:
            raise TypeError("Input image_data must be a NumPy ndarray or a PyTorch Tensor.")


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

        merged_map = [torch.cat([c, d], dim=1) for c, d in zip(color_feature_map, depth_feature_map)]
        fused_map = [conv(m) for conv, m in zip(self.fuse_conv, merged_map)]

        return fused_map


class SpatialAttention(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        # 注意力图的生成只依赖于空间信息，因此这里的 in_channels 实际上并不直接影响 conv 的输入通道数
        # conv 的输入通道数始终是平均池化和最大池化结果的拼接，即 2
        self.conv = nn.Conv2d(2, 1, kernel_size=1)
        self.sigmoid = nn.Sigmoid()


    def forward(self, x):
        # 平均池化和最大池化
        avg_pool = torch.mean(x, dim=1, keepdim=True)
        max_pool, _ = torch.max(x, dim=1, keepdim=True)

        # 拼接池化结果
        concat = torch.cat([avg_pool, max_pool], dim=1)

        # 通过卷积层
        attention_map = self.conv(concat)
        attention_map = self.sigmoid(attention_map)

        return attention_map


class FeatureFuserWithSpatialAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.spatial_attentions = nn.ModuleList([
            SpatialAttention(192),  # 拼接后通道数
            SpatialAttention(384),
            SpatialAttention(768),
            SpatialAttention(1536)
        ])
        self.fuse_conv = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(192, 96, 1),  # 拼接后通道数作为输入
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
                nn.Conv2d(1536,768, 1),
                nn.ReLU()
            )
        ])


    def forward(self, color_feature_map, depth_feature_map):
        assert len(color_feature_map) == len(depth_feature_map), \
            "the tuple length of color_feature_map and depth_feature_map should be the same"

        fused_map = []
        for i, (color_feat, depth_feat) in enumerate(zip(color_feature_map, depth_feature_map)):
            # 1. 拼接特征图
            merged_feature = torch.cat([color_feat, depth_feat], dim=1)

            # 2. 计算空间注意力权重
            attention_map = self.spatial_attentions [i](merged_feature)

            # 3. 将注意力权重分别应用到颜色和深度特征图
            attended_color_feat = color_feat * attention_map
            attended_depth_feat = depth_feat * attention_map

            # 4. 再次拼接加权后的特征图
            reattended_feature = torch.cat([attended_color_feat, attended_depth_feat], dim=1)

            # 5. 使用 1x1 卷积压缩回原始通道数
            fused_feature = self.fuse_conv [i](reattended_feature)
            fused_map.append(fused_feature)

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
        if in_channels != out_channels:
            self.conv_layers = nn.ModuleList([
                nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1) for _ in range(num_depth_regions + 1)
            ]) # T+1 个 1x1 卷积层
            self.rgb_projection = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1, bias=False)
        else:
            self.conv_layers = nn.ModuleList([
                nn.Conv2d(in_channels, out_channels, kernel_size=1) for _ in
                range(num_depth_regions + 1)
            ])  # T+1 个 1x1 卷积层

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

        # 3. 残差连接 (Residual Connection) -  修改部分
        if self.in_channels != self.out_channels:  # 如果需要 projection
            projected_rgb_features = self.rgb_projection(rgb_features)  # 使用 1x1 卷积调整 rgb_features 通道数
            output_features = enhanced_features + projected_rgb_features  # 残差连接
        else:  # 如果 in_channels == out_channels，则直接残差连接
            output_features = enhanced_features + rgb_features
        return output_features


# "The features from different modalities of the same scale are always fused,
# while features in different scales are selectively fused."
# (来自相同尺度的不同模态特征总是被融合，而不同尺度的特征则有选择地融合)。