import random
from typing import List, Dict, Union

import numpy as np
import torch
import cv2
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torch import Tensor, nn
from transformers import Mask2FormerConfig, Mask2FormerModel, Mask2FormerForUniversalSegmentation, Mask2FormerImageProcessor
from transformers.models.mask2former.modeling_mask2former import Mask2FormerPixelLevelModule, \
    Mask2FormerForUniversalSegmentationOutput, Mask2FormerPixelLevelModuleOutput
from transformers.utils.backbone_utils import load_backbone
from mask2former.utils.data_process import calculate_surface_normals


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
    def __init__(self, config, version):
        print("[CustomMask2FormerModel] constructing...")
        super().__init__(config)
        self.pixel_level_module = CustomMask2FormerPixelLevelModule(config, version=version)


class CustomMask2FormerForUniversalSegmentation(Mask2FormerForUniversalSegmentation):
    main_input_name = "pixel_values"
    config_class = CustomConfig

    def __init__(self, config, version='0.0.0'):
        print("[CustomMask2FormerForUniversalSegmentation] constructing...")
        super().__init__(config)
        set_seed(42)
        self.model = CustomMask2FormerModel(config, version=version)


class CustomMask2FormerPixelLevelModule(Mask2FormerPixelLevelModule):
    main_input_name = "pixel_values"
    def __init__(self, config, version):
        print("[CustomMask2FormerPixelLevelModule] constructing...")
        super().__init__(config)
        self.version = version
        print(f"[CustomMask2FormerPixelLevelModule] Version={self.version}")
        if self.version == '0.0.0': # 3 channel (RGB)
            pass # use super().encoder

        elif self.version == '0.0.1':  # 6 channel (RGB, Gradient-Depth)
            color_map_channels = [96, 192, 384, 768]
            self.depth_gradient_injection = DepthGradientInjection(color_map_channels, 3)

        elif self.version == '0.0.2' :  # 6 channel (RGB, Gradient-Depth)
            color_map_channels = [96, 192, 384, 768]
            self.depth_gradient_injection = DepthGradientInjectionWithMask(color_map_channels, 3)

        elif self.version == '0.0.3' or self.version == '0.0.4' or self.version == '0.0.5' or self.version == '0.0.6':  # 7 channel (RGB, Gradient-Depth, Gradient-Mask)
            color_map_channels = [96, 192, 384, 768]
            self.depth_gradient_injection = DepthGradientInjectionResidual(color_map_channels, 3)

        elif self.version == '0.0.7': # 4 channel (RGB, Gray-Depth)
            color_map_channels = [96, 192, 384, 768]
            self.depth_gradient_injection = DepthGradientInjectionResidual(color_map_channels, 3)
            self.intrinsics_predictor = IntrinsicsPredictorFromDepthImage()

        elif self.version == '0.1.0': # 6 channel (RGB, Depth)
            # self.depth_encoder = AutoBackbone.from_pretrained("microsoft/resnet-50")
            self.depth_encoder = load_backbone(config)
            self.feature_fuser = FeatureFuser()

        elif self.version == '0.1.1': # 6 channel (RGB, Depth)
            self.depth_encoder = load_backbone(config)
            self.feature_fuser = FeatureFuser()
            self.dsam0 = DSAModule(in_channels=96, out_channels=192, num_depth_regions=3)
            self.dsam1 = DSAModule(in_channels=192, out_channels=384, num_depth_regions=3)
            self.dsam2 = DSAModule(in_channels=384, out_channels=768, num_depth_regions=3)

        elif self.version == '0.1.2': # 6 channel (RGB, Depth)
            self.dsam0 = DSAModule(in_channels=96, out_channels=192, num_depth_regions=3)
            self.dsam1 = DSAModule(in_channels=192, out_channels=384, num_depth_regions=3)
            self.dsam2 = DSAModule(in_channels=384, out_channels=768, num_depth_regions=3)

        elif self.version == '0.1.3': # 6 channel (RGB, Depth)
            self.depth_encoder = load_backbone(config)

            # Define the channel counts for the RatioPredictor
            depth_channels = [96, 192, 384, 768]
            self.ratio_predictor = RatioPredictor(depth_channels_list=depth_channels)
            self.dsam0 = DSAModule(in_channels=96, out_channels=192, num_depth_regions=3)
            self.dsam1 = DSAModule(in_channels=192, out_channels=384, num_depth_regions=3)
            self.dsam2 = DSAModule(in_channels=384, out_channels=768, num_depth_regions=3)

        elif self.version == '0.3.0': # 10 channel (RGB, Depth, 3Gradient-Depth, 3Gradient-Depth-Mask)
            self.depth_encoder = load_backbone(config)

            # Define the channel counts for the RatioPredictor
            depth_channels = [96, 192, 384, 768]
            self.ratio_predictor = RatioPredictor(depth_channels_list=depth_channels)
            self.dsam0 = DSAModule(in_channels=96, out_channels=192, num_depth_regions=3)
            self.dsam1 = DSAModule(in_channels=192, out_channels=384, num_depth_regions=3)
            self.dsam2 = DSAModule(in_channels=384, out_channels=768, num_depth_regions=3)

            color_map_channels = [96, 192, 384, 768]
            self.depth_gradient_injection = DepthGradientInjectionResidual(color_map_channels, 3)

        elif self.version == '0.4.0': # 10 channel (RGB, Depth, Gradient-Depth, Gradient-Depth-Mask)

            # Define the channel counts for the RatioPredictor
            # self.ratio_predictor = DepthImageRatioPredictor(3)
            self.ratio_predictor = EnhancedDepthImageRatioPredictor(3)

            self.dsam0 = DSAModule(in_channels=96, out_channels=192, num_depth_regions=3)
            self.dsam1 = DSAModule(in_channels=192, out_channels=384, num_depth_regions=3)
            self.dsam2 = DSAModule(in_channels=384, out_channels=768, num_depth_regions=3)

            color_map_channels = [96, 192, 384, 768]
            self.depth_gradient_injection = DepthGradientInjectionResidual(color_map_channels, 3)

        else: # 9 channel (RGB, Depth, Fused_depth)
            self.depth_encoder = load_backbone(config)
            self.feature_fuser = FeatureFuser()
            self.dsam0 = DSAModule(in_channels=96, out_channels=192, num_depth_regions=3)
            self.dsam1 = DSAModule(in_channels=192, out_channels=384, num_depth_regions=3)
            self.dsam2 = DSAModule(in_channels=384, out_channels=768, num_depth_regions=3)

    def forward(self, pixel_values: Tensor, output_hidden_states: bool = False) -> Mask2FormerPixelLevelModuleOutput:
        # print("testing")
        if self.version == '0.0.0': # 3 channel
            backbone_features = self.encoder(pixel_values).feature_maps

        elif self.version == '0.0.1':  # 6 channel (RGB, Gradient-Depth)
            rgb = pixel_values[:, 0:3, :, :]
            depth = pixel_values[:, 3:6, :, :]

            color_feature_map = self.encoder(rgb).feature_maps
            cp_color_feature_map = list(color_feature_map)
            backbone_features = self.depth_gradient_injection(cp_color_feature_map, depth)

        elif self.version == '0.0.2' or self.version == '0.0.3' or self.version == '0.0.4' or self.version == '0.0.5' or self.version == '0.0.6': # 7 channel (RGB, Gradient-Depth, Gradient-Mask)
            rgb = pixel_values[:, 0:3, :, :]
            depth = pixel_values[:, 3:6, :, :]
            gradient_mask = pixel_values[:, 6:7, :, :]

            color_feature_map = self.encoder(rgb).feature_maps
            cp_color_feature_map = list(color_feature_map)
            backbone_features = self.depth_gradient_injection(cp_color_feature_map, depth, gradient_mask)

        elif self.version == '0.0.7':  # 4 channel (RGB, Gray-Depth)
            rgb = pixel_values[:, 0:3, :, :]
            gray_depth = pixel_values[:, 3:4, :, :]
            gray_depth_np = gray_depth.cpu().numpy().squeeze(1)
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

            color_feature_map = self.encoder(rgb).feature_maps
            cp_color_feature_map = list(color_feature_map)
            depth_h = gray_depth.shape[2]
            depth_w = gray_depth.shape[3]
            predicted_intrinsics = self.intrinsics_predictor(gray_depth, (depth_h, depth_w))
            surface_normals_results = [calculate_surface_normals(i, j) for i, j in zip(gray_depth_np, predicted_intrinsics)]
            surface_normals_np_batch, valid_normal_mask_np_batch = [list(x) for x in zip(*surface_normals_results)]

            # 1. 将每个 NumPy 数组转换为 Tensor 并调整维度
            surface_normals_tensor_list = [
                torch.from_numpy(normals_np).permute(2, 0, 1)  # Permute dimensions from (0, 1, 2) to (2, 0, 1)
                for normals_np in surface_normals_np_batch
            ] # 从 (H, W, D_channels) 转换为 (D_channels, H, W)
            valid_normal_mask_tensor_list = [
                torch.from_numpy(mask_np).unsqueeze(0)  # Add a channel dimension at the beginning
                for mask_np in valid_normal_mask_np_batch
            ] # 从 (H, W) 转换为 (1, H, W) (添加通道维度)

            # 2. 堆叠成批处理 Tensor
            surface_normals_tensor = torch.stack(surface_normals_tensor_list, dim=0) # (B, D_channels, H, W)
            valid_normal_mask_tensor = torch.stack(valid_normal_mask_tensor_list, dim=0) # (B, 1, H, W)

            # 3. 确保数据类型和设备
            surface_normals_tensor = surface_normals_tensor.float().to(device)
            valid_normal_mask_tensor = valid_normal_mask_tensor.float().to(device)

            backbone_features = self.depth_gradient_injection(cp_color_feature_map, surface_normals_tensor, valid_normal_mask_tensor)

        elif self.version == '0.1.0': # 6 channel
            rgb = pixel_values[:, 0:3, :, :]
            depth = pixel_values[:, 3:6, :, :]
            color_feature_map = self.encoder(rgb).feature_maps
            depth_feature_map = self.depth_encoder(depth).feature_maps
            backbone_features = self.feature_fuser(color_feature_map, depth_feature_map)

        elif self.version == '0.1.1': # 6 channel
            rgb = pixel_values[:, 0:3, :, :]
            depth = pixel_values[:, 3:6, :, :]
            color_feature_map = self.encoder(rgb).feature_maps
            depth_feature_map = self.depth_encoder(depth).feature_maps

            # DSAM
            cp_color_feature_map = list(color_feature_map)
            cp_depth_feature_map = list(depth_feature_map)

            dsam_output0 = [self.dsam0(i.unsqueeze(0), self.to_grayscale(j)) for i, j in
                            zip(cp_color_feature_map[0], depth)]  # [B, 96, 64, 64]
            dsam_output0 = torch.stack(dsam_output0, dim=0).squeeze(1)  # [B, 192, 32, 32]
            cp_color_feature_map[1] += dsam_output0

            dsam_output1 = [self.dsam1(i.unsqueeze(0), self.to_grayscale(j)) for i, j in
                            zip(cp_color_feature_map[1], depth)]  # [B, 192, 32, 32]
            dsam_output1 = torch.stack(dsam_output1, dim=0).squeeze(1)  # [B, 384, 16, 16]
            cp_color_feature_map[2] += dsam_output1

            dsam_output2 = [self.dsam2(i.unsqueeze(0), self.to_grayscale(j)) for i, j in
                            zip(cp_color_feature_map[2], depth)]  # [B, 384, 16, 16]
            dsam_output2 = torch.stack(dsam_output2, dim=0).squeeze(1)  # [B, 768, 16, 16]
            cp_color_feature_map[3] += dsam_output2

            # combine the features of rgb and depth
            backbone_features = self.feature_fuser(cp_color_feature_map, cp_depth_feature_map)

        elif self.version == '0.1.2':
            rgb = pixel_values[:, 0:3, :, :]
            depth = pixel_values[:, 3:6, :, :]
            color_feature_map = self.encoder(rgb).feature_maps

            # DSAM
            cp_color_feature_map = list(color_feature_map)

            dsam_output0 = [self.dsam0(i.unsqueeze(0), self.to_grayscale(j)) for i, j in
                            zip(cp_color_feature_map[0], depth)]  # [B, 96, 64, 64]
            dsam_output0 = torch.stack(dsam_output0, dim=0).squeeze(1)  # [B, 192, 32, 32]
            cp_color_feature_map[1] += dsam_output0

            dsam_output1 = [self.dsam1(i.unsqueeze(0), self.to_grayscale(j)) for i, j in
                            zip(cp_color_feature_map[1], depth)]  # [B, 192, 32, 32]
            dsam_output1 = torch.stack(dsam_output1, dim=0).squeeze(1)  # [B, 384, 16, 16]
            cp_color_feature_map[2] += dsam_output1

            dsam_output2 = [self.dsam2(i.unsqueeze(0), self.to_grayscale(j)) for i, j in
                            zip(cp_color_feature_map[2], depth)]  # [B, 384, 16, 16]
            dsam_output2 = torch.stack(dsam_output2, dim=0).squeeze(1)  # [B, 768, 16, 16]
            cp_color_feature_map[3] += dsam_output2

            backbone_features = cp_color_feature_map

        elif self.version == '0.1.3':
            rgb = pixel_values[:, 0:3, :, :]
            depth = pixel_values[:, 3:6, :, :]
            color_feature_map = self.encoder(rgb).feature_maps
            depth_feature_map = self.depth_encoder(depth).feature_maps
            cp_depth_feature_map = list(depth_feature_map)

            # Predict window_size_ratio
            predicted_ratios = self.ratio_predictor(cp_depth_feature_map)
            # print(f"\nForward pass complete. Predicted ratios shape: {predicted_ratios.shape}")
            # print(f"Predicted ratios (first batch item): {predicted_ratios[0].item()}")
            # print(f"Predicted ratios (all batch items):\n{predicted_ratios.squeeze(1).tolist()}")

            # DSAM
            cp_color_feature_map = list(color_feature_map)

            dsam_output0 = [self.dsam0(i.unsqueeze(0), self.to_grayscale(j), k.item()) for i, j, k in
                            zip(cp_color_feature_map[0], depth, predicted_ratios)]  # [B, 96, 64, 64]
            dsam_output0 = torch.stack(dsam_output0, dim=0).squeeze(1)  # [B, 192, 32, 32]
            cp_color_feature_map[1] += dsam_output0

            dsam_output1 = [self.dsam1(i.unsqueeze(0), self.to_grayscale(j), k.item()) for i, j, k in
                            zip(cp_color_feature_map[1], depth, predicted_ratios)]  # [B, 192, 32, 32]
            dsam_output1 = torch.stack(dsam_output1, dim=0).squeeze(1)  # [B, 384, 16, 16]
            cp_color_feature_map[2] += dsam_output1

            dsam_output2 = [self.dsam2(i.unsqueeze(0), self.to_grayscale(j), k.item()) for i, j, k in
                            zip(cp_color_feature_map[2], depth, predicted_ratios)]  # [B, 384, 16, 16]
            dsam_output2 = torch.stack(dsam_output2, dim=0).squeeze(1)  # [B, 768, 16, 16]
            cp_color_feature_map[3] += dsam_output2

            backbone_features = cp_color_feature_map

        elif self.version == '0.3.0':
            rgb = pixel_values[:, 0:3, :, :]
            depth = pixel_values[:, 3:6, :, :]
            gradient_depth = pixel_values[:, 6:9, :, :]
            gradient_mask = pixel_values[:, 9:10, :, :]

            color_feature_map = self.encoder(rgb).feature_maps
            depth_feature_map = self.depth_encoder(depth).feature_maps
            cp_color_feature_map = list(color_feature_map)
            cp_depth_feature_map = list(depth_feature_map)

            # Predict window_size_ratio
            predicted_ratios = self.ratio_predictor(cp_depth_feature_map)

            # DSAM
            dsam_output0 = [self.dsam0(i.unsqueeze(0), self.to_grayscale(j), k.item()) for i, j, k in
                            zip(cp_color_feature_map[0], depth, predicted_ratios)]  # [B, 96, 64, 64]
            dsam_output0 = torch.stack(dsam_output0, dim=0).squeeze(1)  # [B, 192, 32, 32]
            cp_color_feature_map[1] += dsam_output0

            dsam_output1 = [self.dsam1(i.unsqueeze(0), self.to_grayscale(j), k.item()) for i, j, k in
                            zip(cp_color_feature_map[1], depth, predicted_ratios)]  # [B, 192, 32, 32]
            dsam_output1 = torch.stack(dsam_output1, dim=0).squeeze(1)  # [B, 384, 16, 16]
            cp_color_feature_map[2] += dsam_output1

            dsam_output2 = [self.dsam2(i.unsqueeze(0), self.to_grayscale(j), k.item()) for i, j, k in
                            zip(cp_color_feature_map[2], depth, predicted_ratios)]  # [B, 384, 16, 16]
            dsam_output2 = torch.stack(dsam_output2, dim=0).squeeze(1)  # [B, 768, 16, 16]
            cp_color_feature_map[3] += dsam_output2

            backbone_features = self.depth_gradient_injection(cp_color_feature_map, gradient_depth, gradient_mask)

        elif self.version == '0.4.0':
            rgb = pixel_values[:, 0:3, :, :]
            depth = pixel_values[:, 3:6, :, :]
            gradient_depth = pixel_values[:, 6:9, :, :]
            gradient_mask = pixel_values[:, 9:10, :, :]

            color_feature_map = self.encoder(rgb).feature_maps
            # cp_color_feature_map = list(color_feature_map)
            cp1_color_feature_map = [t.detach().clone() for t in list(color_feature_map)]
            cp2_color_feature_map = [t.detach().clone() for t in list(color_feature_map)]

            # Predict window_size_ratio
            predicted_ratios = self.ratio_predictor(depth)

            # DSAM
            dsam_output0 = [self.dsam0(i.unsqueeze(0), self.to_grayscale(j), k.item()) for i, j, k in
                            zip(cp1_color_feature_map[0], depth, predicted_ratios)]  # [B, 96, 64, 64]
            dsam_output0 = torch.stack(dsam_output0, dim=0).squeeze(1)  # [B, 192, 32, 32]
            cp1_color_feature_map[1] += dsam_output0

            dsam_output1 = [self.dsam1(i.unsqueeze(0), self.to_grayscale(j), k.item()) for i, j, k in
                            zip(cp1_color_feature_map[1], depth, predicted_ratios)]  # [B, 192, 32, 32]
            dsam_output1 = torch.stack(dsam_output1, dim=0).squeeze(1)  # [B, 384, 16, 16]
            cp1_color_feature_map[2] += dsam_output1

            dsam_output2 = [self.dsam2(i.unsqueeze(0), self.to_grayscale(j), k.item()) for i, j, k in
                            zip(cp1_color_feature_map[2], depth, predicted_ratios)]  # [B, 384, 16, 16]
            dsam_output2 = torch.stack(dsam_output2, dim=0).squeeze(1)  # [B, 768, 16, 16]
            cp1_color_feature_map[3] += dsam_output2

            cp2_color_feature_map = self.depth_gradient_injection(cp2_color_feature_map, gradient_depth, gradient_mask)
            backbone_features = [i + j for i, j in zip(cp1_color_feature_map, cp2_color_feature_map)]

        else: # 9 channel
            rgb = pixel_values[:, 0:3, :, :]
            depth = pixel_values[:, 3:6, :, :]
            fused_depth = pixel_values[:, 6:9, :, :]
            color_feature_map = self.encoder(rgb).feature_maps
            depth_feature_map = self.depth_encoder(depth).feature_maps

            # DSAM
            cp_color_feature_map = list(color_feature_map)
            cp_depth_feature_map = list(depth_feature_map)

            dsam_output0 = [self.dsam0(i.unsqueeze(0), self.to_grayscale(j)) for i, j in zip(cp_color_feature_map[0], fused_depth)] # [B, 96, 64, 64]
            dsam_output0 = torch.stack(dsam_output0, dim=0).squeeze(1)  # [B, 192, 32, 32]
            cp_color_feature_map[1] += dsam_output0

            dsam_output1 = [self.dsam1(i.unsqueeze(0), self.to_grayscale(j)) for i, j in zip(cp_color_feature_map[1], fused_depth)] # [B, 192, 32, 32]
            dsam_output1 = torch.stack(dsam_output1, dim=0).squeeze(1)  # [B, 384, 16, 16]
            cp_color_feature_map[2] += dsam_output1

            dsam_output2 = [self.dsam2(i.unsqueeze(0), self.to_grayscale(j)) for i, j in zip(cp_color_feature_map[2], fused_depth)] # [B, 384, 16, 16]
            dsam_output2 = torch.stack(dsam_output2, dim=0).squeeze(1)  # [B, 768, 16, 16]
            cp_color_feature_map[3] += dsam_output2

            # combine the features of rgb and depth
            backbone_features = self.feature_fuser(cp_color_feature_map, cp_depth_feature_map)

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

    def forward(self, rgb_features, depth_map, window_size_ratio=0.1):
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

        hist, bin_edges = self._calculate_depth_histogram(depth_map_np)
        depth_modes = self._select_depth_distribution_modes(hist, bin_edges, num_modes=self.num_depth_regions)
        if depth_modes: # 只有当检测到 depth_modes 时才进行后续步骤，防止空列表导致错误
            interval_windows = self._define_depth_interval_windows(depth_modes, window_size_ratio=window_size_ratio)
            region_masks = self._generate_depth_region_masks(depth_map_np, interval_windows)
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

    def _calculate_depth_histogram(self, depth_map, bins=512, value_range=None):
        """
        计算深度图的直方图。

        Args:
            depth_map (numpy.ndarray): 输入深度图 (单通道).
            bins (int): 直方图的柱子数量 (bins). 默认 256.
            value_range (tuple, optional): 深度值的范围 (min, max).
                                          如果不指定，则使用深度图中的最小值和最大值. Defaults to None.

        Returns:
            tuple: 包含直方图计数 (hist) 和 bin 边缘 (bin_edges).
        """
        if value_range is None:
            value_range = (np.nanmin(depth_map), np.nanmax(depth_map))  # 忽略 NaN 值

        hist, bin_edges = np.histogram(depth_map.flatten(), bins=bins, range=value_range, density=False)
        return hist, bin_edges

    def _select_depth_distribution_modes(self, hist, bin_edges, num_modes=3, prominence_threshold=0.01):
        """
        从深度直方图中选择深度分布模式 (峰值).

        Args:
            hist (numpy.ndarray): 直方图计数.
            bin_edges (numpy.ndarray): bin 边缘.
            num_modes (int): 要选择的深度分布模式的数量. 默认 3.
            prominence_threshold (float): 峰值的显著性阈值 (相对于最大峰值高度).
                                         用于过滤不显著的峰值. 默认 0.01.

        Returns:
            list: 包含选定深度分布模式的中心值 (近似).
                  如果找不到足够的显著峰值，则返回少于 num_modes 的列表.
        """
        from scipy.signal import find_peaks

        # 查找峰值索引
        peaks_indices, _ = find_peaks(hist, prominence=prominence_threshold * np.max(hist))  # 使用显著性阈值

        if not peaks_indices.size:  # 如果没有找到峰值
            return []

        # 获取峰值的高度和位置 (近似中心值)
        peak_heights = hist[peaks_indices]
        peak_centers = bin_edges[:-1][peaks_indices] + np.diff(bin_edges)[peaks_indices] / 2.0  # 近似中心值

        # 将峰值按照高度降序排序
        peak_data = sorted(zip(peak_heights, peak_centers), reverse=True)

        selected_modes = [center for _, center in peak_data[:num_modes]]  # 选择前 num_modes 个峰值中心

        return selected_modes

    def _define_depth_interval_windows(self, depth_modes, window_size_ratio=0.1):
        """
        根据深度分布模式定义深度区间窗口.

        Args:
            depth_modes (list): 深度分布模式的中心值列表.
            window_size_ratio (float): 窗口大小相对于深度模式中心值的比例. 默认 0.1.
                                       例如，ratio=0.1，则窗口宽度为中心值的 10%.

        Returns:
            list: 包含深度区间窗口 (元组 (lower_bound, upper_bound)) 的列表.
        """
        interval_windows = []
        for mode_center in depth_modes:
            window_half_width = mode_center * window_size_ratio / 2.0  # 半宽度，保证窗口宽度与中心值比例一致
            lower_bound = max(0, mode_center - window_half_width)  # 保证下界不小于0，假设深度值非负
            upper_bound = mode_center + window_half_width
            interval_windows.append((lower_bound, upper_bound))
        return interval_windows

    def _generate_depth_region_masks(self, depth_map, interval_windows):
        """
        根据深度区间窗口生成深度区域掩码.

        Args:
            depth_map (numpy.ndarray): 输入深度图 (单通道).
            interval_windows (list): 深度区间窗口列表，每个窗口为元组 (lower_bound, upper_bound).

        Returns:
            list: 包含深度区域掩码 (numpy.ndarray, bool 类型) 的列表.
                  最后一个掩码是剩余区域掩码.
        """
        region_masks = []
        combined_mask = np.zeros_like(depth_map, dtype=bool)  # 用于记录已覆盖的区域

        for lower_bound, upper_bound in interval_windows:
            mask = (depth_map >= lower_bound) & (depth_map <= upper_bound)
            region_masks.append(mask)
            combined_mask |= mask  # 累积覆盖区域

        # 生成剩余区域掩码 (深度值不在任何定义的窗口内的区域)
        remaining_mask = ~combined_mask
        region_masks.append(remaining_mask)

        return region_masks

    def histogram_viewer(self, hist, bin_edges):
        # 可视化直方图
        plt.figure(figsize=(8, 4))
        plt.bar(bin_edges[:-1], hist, width=np.diff(bin_edges), align="edge")
        plt.title("Depth Histogram")
        plt.xlabel("Depth Value")
        plt.ylabel("Frequency")
        plt.show()

    def depth_region_viewer(self, interval_windows, region_masks):
        # 可视化深度区域掩码 (前3个区域 + 剩余区域)
        plt.figure(figsize=(12, 3))
        titles = [f"Region Mask {i + 1}" for i in range(len(interval_windows))] + ["Remaining Region Mask"]
        for i, mask in enumerate(region_masks):
            plt.subplot(1, len(region_masks), i + 1)
            plt.imshow(mask, cmap='gray')
            plt.title(titles[i])
            plt.axis('off')
        plt.suptitle("Depth Region Masks Visualization")
        plt.tight_layout()
        plt.show()


class RatioPredictor(nn.Module):
    """
    Predicts the window_size_ratio based on multi-scale input features.
    Takes a list of feature maps from a backbone.
    """
    def __init__(self, depth_channels_list: list[int]):
        """
        Args:
            depth_channels_list (list[int]): A list of channel counts for the
                                             depth feature maps at each scale
                                             (e.g., [96, 192, 384, 768]).
        """
        super().__init__()
        self.depth_channels_list = depth_channels_list
        self.num_scales = len(depth_channels_list)

        # Calculate the total number of features after pooling and concatenation
        # This will be the sum of the channel counts from all scales
        total_pooled_features = sum(depth_channels_list)

        # Define the fully connected layers that take the concatenated features
        self.fc_layers = nn.Sequential(
            nn.Linear(total_pooled_features, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 1) # Output a single scalar ratio per image in the batch
        )

        # Global Average Pooling layer to reduce spatial dimensions
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)

        # Constrain the output range of the parameter (ratio)
        self.output_min = 0.01 # Example minimum ratio
        self.output_max = 0.5  # Example maximum ratio
        self.sigmoid = nn.Sigmoid()

    def forward(self, depth_feature_maps: list[torch.Tensor]) -> torch.Tensor:
        """
        Args:
            depth_feature_maps (list[torch.Tensor]): A list of depth feature maps
                                                    from the depth backbone at different scales.
                                                    Shapes: [(B, C1, H1, W1), (B, C2, H2, W2), ...].

        Returns:
            torch.Tensor: Predicted window_size_ratio (B, 1).
        """
        assert len(depth_feature_maps) == self.num_scales, \
            f"Expected {self.num_scales} depth feature maps, but got {len(depth_feature_maps)}"

        pooled_features = []
        for i, feature_map in enumerate(depth_feature_maps):
            # Ensure channel count matches expected
            assert feature_map.shape[1] == self.depth_channels_list[i], \
                f"Expected {self.depth_channels_list[i]} channels for scale {i}, but got {feature_map.shape[1]}"

            # Apply Global Average Pooling
            pooled = self.global_avg_pool(feature_map) # Shape (B, C_i, 1, 1)

            # Squeeze spatial dimensions to get (B, C_i)
            pooled = pooled.squeeze(-1).squeeze(-1) # Shape (B, C_i)

            pooled_features.append(pooled)

        # Concatenate pooled features from all scales along the channel dimension (dim=1)
        # Resulting shape will be (B, sum(C_i))
        concatenated_features = torch.cat(pooled_features, dim=1)

        # Pass the concatenated features through the fully connected layers
        raw_ratio = self.fc_layers(concatenated_features) # Shape (B, 1)

        # Apply constraint to map output to [output_min, output_max]
        predicted_ratio = self.output_min + (self.output_max - self.output_min) * self.sigmoid(raw_ratio) # Shape (B, 1)

        return predicted_ratio


class IntrinsicsPredictorFromDepthImage(nn.Module):
    """
    Predicts camera intrinsic parameters (fx, fy, cx, cy) based on the input depth image.
    Contains a small convolutional backbone to extract features from the depth image.
    """
    def __init__(self, in_channels: int = 1):
        """
        Args:
            in_channels (int): Number of input channels for the depth image (should be 1).
        """
        super().__init__()
        self.in_channels = in_channels

        # Define a small convolutional backbone to process the depth image
        # This is a simplified example, you might need a more complex one
        # depending on the complexity of the task and data.
        self.conv_backbone = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=2, padding=1), # Reduce spatial size, increase channels
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1), # Reduce spatial size again
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1), # Reduce spatial size again
            nn.ReLU(inplace=True),
            # Add more layers if needed...
        )

        # After the conv backbone, we'll use Global Average Pooling
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)

        # Determine the number of features after the conv backbone and pooling
        # We need to run a dummy forward pass or calculate based on architecture
        # Let's assume the last conv layer outputs 128 channels as per the example above
        # If you change the conv_backbone, update this value.
        pooled_feature_size = 128 # Number of output channels from the last conv layer

        # Define the fully connected layers for prediction
        self.fc_layers = nn.Sequential(
            nn.Linear(pooled_feature_size, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 4) # Output 4 raw values for fx, fy, cx, cy
        )

    def forward(self, depth_image: torch.Tensor, target_size: tuple[int, int]) -> list[
        dict[str, Union[int, int, float, float, bool, bool]]]:
        """
        Args:
            depth_image (torch.Tensor): Input depth image tensor (B, 1, H, W).
            target_size (tuple[int, int]): The target (height, width) the input images
                                           were resized to before being fed to the model.
                                           This is crucial for scaling cx and cy predictions
                                           relative to the image dimensions.

        Returns:
            List[dict[str, Union[int, int, float, float, bool, bool]]]: Predicted camera intrinsics {'fx', 'fy', 'cx', 'cy'}
        """
        # Ensure input is single channel
        assert depth_image.shape[1] == self.in_channels, \
            f"Expected {self.in_channels} input channel(s), but got {depth_image.shape[1]}"

        # Process the depth image through the convolutional backbone
        features = self.conv_backbone(depth_image) # Shape (B, C_last, H', W')

        # Apply Global Average Pooling
        pooled = self.global_avg_pool(features) # Shape (B, C_last, 1, 1)

        # Squeeze spatial dimensions to get (B, C_last)
        pooled = pooled.squeeze(-1).squeeze(-1) # Shape (B, C_last)

        # Pass the pooled features through the fully connected layers
        raw_intrinsics = self.fc_layers(pooled) # Shape (B, 4)

        # Split the raw output into the four intrinsic components
        raw_fx, raw_fy, raw_cx, raw_cy = raw_intrinsics.split(1, dim=1)

        # Apply activations and scale predictions based on target image size
        target_h, target_w = target_size
        target_w_tensor = torch.tensor(target_w, dtype=raw_cx.dtype, device=raw_cx.device)
        target_h_tensor = torch.tensor(target_h, dtype=raw_cy.dtype, device=raw_cy.device)

        # fx and fy must be positive. Use torch.exp.
        predicted_fx = torch.exp(raw_fx)
        predicted_fy = torch.exp(raw_fy)

        # cx and cy are pixel coordinates. Use sigmoid and scale by image dimensions.
        predicted_cx = torch.sigmoid(raw_cx) * target_w_tensor
        predicted_cy = torch.sigmoid(raw_cy) * target_h_tensor

        # Return the predicted intrinsics as a dictionary
        predicted_intrinsics = {
            'fx': predicted_fx, # Shape (B, 1)
            'fy': predicted_fy, # Shape (B, 1)
            'cx': predicted_cx, # Shape (B, 1)
            'cy': predicted_cy  # Shape (B, 1)
        }

        # 获取 batch size
        batch_size = predicted_intrinsics['fx'].shape[0]

        # 使用列表推导式进行转换
        intrinsics_list = [
            {key: tensor[i].item() for key, tensor in predicted_intrinsics.items()}
            for i in range(batch_size)
        ]

        return intrinsics_list


class DepthGradientInjection(nn.Module):
    """
    Injects downsampled depth gradient features into multi-scale color feature maps.
    Assumes depth_gradient_map is at the original image resolution.
    """
    def __init__(self, color_channels: list[int], depth_gradient_channels: int):
        """
        Args:
            color_channels (list[int]): A list of channel counts for the color
                                        feature maps at each scale (e.g., [96, 192, 384, 768]).
            depth_gradient_channels (int): The number of channels in the
                                           preprocessed depth gradient map (e.g., 1 for magnitude, 2 for Gx+Gy, 3 for Normals).
        """
        super().__init__()
        self.color_channels = color_channels
        self.depth_gradient_channels = depth_gradient_channels
        self.num_scales = len(color_channels)

        # Define fusion layers for each scale
        # Each layer takes concatenated color feature and downsampled depth gradient
        # and outputs a fused feature with the same channel count as the color feature.
        self.fusion_layers = nn.ModuleList()
        for channels in color_channels:
            self.fusion_layers.append(
                nn.Sequential(
                    nn.Conv2d(channels + depth_gradient_channels, channels, kernel_size=1),
                    nn.ReLU(inplace=True)
                    # You could add more layers here if needed, e.g., BatchNorm, another Conv
                )
            )

    def forward(self, color_feature_maps: list[torch.Tensor], depth_gradient_map: torch.Tensor) -> list[torch.Tensor]:
        """
        Args:
            color_feature_maps (list[torch.Tensor]): A list of color feature maps
                                                     from the color backbone at different scales.
                                                     Shapes: [(B, C1, H1, W1), (B, C2, H2, W2), ...].
            depth_gradient_map (torch.Tensor | None): The preprocessed depth gradient map
                                                      at the original image resolution (B, D_channels, H, W).
                                                      Can be None if depth is not available (e.g., RGB-only inference).

        Returns:
            list[torch.Tensor]: A list of fused feature maps at the same scales
                                and channel counts as the input color_feature_maps.
        """
        assert len(color_feature_maps) == self.num_scales, \
            f"Expected {self.num_scales} color feature maps, but got {len(color_feature_maps)}"

        fused_feature_maps = []

        for i, color_feat in enumerate(color_feature_maps):
            # Get target size from the current color feature map
            _, _, H_i, W_i = color_feat.shape

            if depth_gradient_map is not None:
                # Ensure depth_gradient_map has the expected number of channels
                assert depth_gradient_map.shape[1] == self.depth_gradient_channels, \
                    f"Expected depth_gradient_map with {self.depth_gradient_channels} channels, but got {depth_gradient_map.shape[1]}"

                # Downsample the high-resolution depth gradient map to the current scale
                # Use bilinear interpolation for continuous gradient values
                downsampled_depth_grad = F.interpolate(
                    depth_gradient_map,
                    size=(H_i, W_i),
                    mode='bilinear',
                    align_corners=False
                )

                # Concatenate color feature and downsampled depth gradient
                merged_feat = torch.cat([color_feat, downsampled_depth_grad], dim=1)

                # Apply the fusion layer
                fused_feat = self.fusion_layers[i](merged_feat)
            else:
                # If depth is not available, just pass the color feature through
                # (or a layer that matches the fusion layer's output if needed)
                fused_feat = color_feat # No fusion, just use color feature

            fused_feature_maps.append(fused_feat)

        return fused_feature_maps


class DepthGradientInjectionWithMask(nn.Module):
    """
    Injects downsampled depth gradient features into multi-scale color feature maps,
    using a mask to identify valid gradient regions.
    Assumes depth_gradient_map and gradient_mask are at the original image resolution.
    """
    def __init__(self, color_channels: list[int], depth_gradient_channels: int):
        super().__init__()
        self.color_channels = color_channels
        self.depth_gradient_channels = depth_gradient_channels
        self.num_scales = len(color_channels)

        # Define fusion layers for each scale
        # Input channels: color_channels + depth_gradient_channels + 1 (for the mask)
        self.fusion_layers = nn.ModuleList()
        for channels in color_channels:
            self.fusion_layers.append(
                nn.Sequential(
                    nn.Conv2d(channels + depth_gradient_channels + 1, channels, kernel_size=1), # Add 1 channel for the mask
                    nn.ReLU(inplace=True)
                )
            )

    def forward(self, color_feature_maps: list[torch.Tensor],
                processed_depth_gradient_map: torch.Tensor,
                gradient_mask: torch.Tensor) -> list[torch.Tensor]:
        """
        Args:
            color_feature_maps (list[torch.Tensor]): Color feature maps (B, C, H, W).
            processed_depth_gradient_map (torch.Tensor | None): Preprocessed depth gradient map (B, D_channels, H, W).
            gradient_mask (torch.Tensor | None): Binary mask (1 if valid gradient, 0 otherwise) (B, 1, H, W).
                                                 Should be float tensor (0.0 or 1.0).
                                                 Can be None if depth is not available.

        Returns:
            list[torch.Tensor]: Fused feature maps.
        """
        assert len(color_feature_maps) == self.num_scales, \
            f"Expected {self.num_scales} color feature maps, but got {len(color_feature_maps)}"

        fused_feature_maps = []

        for i, color_feat in enumerate(color_feature_maps):
            _, _, H_i, W_i = color_feat.shape

            if processed_depth_gradient_map is not None and gradient_mask is not None:
                assert processed_depth_gradient_map.shape[1] == self.depth_gradient_channels
                assert gradient_mask.shape[1] == 1 # Mask should be single channel

                # Downsample depth gradient map and mask
                downsampled_depth_grad = F.interpolate(
                    processed_depth_gradient_map,
                    size=(H_i, W_i),
                    mode='bilinear', # Use bilinear for gradient
                    align_corners=False
                )
                downsampled_mask = F.interpolate(
                    gradient_mask,
                    size=(H_i, W_i),
                    mode='nearest', # Use nearest for binary mask to preserve sharp boundaries
                    # align_corners=False
                )

                # Concatenate color feature, downsampled depth gradient, and downsampled mask
                merged_feat = torch.cat([color_feat, downsampled_depth_grad, downsampled_mask], dim=1)

                # Apply the fusion layer
                fused_feat = self.fusion_layers[i](merged_feat)
            else:
                # If depth or mask is not available, just use color feature
                fused_feat = color_feat

            fused_feature_maps.append(fused_feat)

        return fused_feature_maps


class DepthGradientInjectionResidual(nn.Module):
    """
    Injects gated depth gradient features into multi-scale color feature maps
    using an additive fusion and residual connection, inspired by attention mechanisms.
    Assumes processed_depth_gradient_map and gradient_mask are at the original image resolution.
    """
    def __init__(self, color_channels: list[int], depth_gradient_channels: int):
        super().__init__()
        self.color_channels = color_channels
        self.depth_gradient_channels = depth_gradient_channels
        self.num_scales = len(color_channels)

        # Define layers to process the gated depth gradient features for each scale
        # These layers will project the gated depth gradient to the same channel count as the color feature
        self.depth_enhancement_layers = nn.ModuleList()
        for channels in color_channels:
            self.depth_enhancement_layers.append(
                nn.Sequential(
                    nn.Conv2d(depth_gradient_channels, channels, kernel_size=1), # Project depth channels to color channels
                    nn.ReLU(inplace=True)
                    # You could add more layers here if needed
                )
            )

        # No separate fusion_layers needed for concatenation in this additive scheme
        # The fusion happens via addition, and the enhancement layer acts like the 'conv_layers' in your example

        # Optional: Projection for residual connection if input/output channels differ
        # In this design, the output channels are the same as input color channels due to addition
        # So, a projection for the residual is only needed if the *initial* color_channels
        # were different from the *desired output* channels of this block.
        # However, typically, injection modules maintain the channel count.
        # Let's assume output channels are the same as input color channels for simplicity.
        # If you need to change channel count, you'd add a final projection layer *after* the residual.

    def forward(self, color_feature_maps: list[torch.Tensor],
                processed_depth_gradient_map: torch.Tensor,
                gradient_mask: torch.Tensor) -> list[torch.Tensor]:
        """
        Args:
            color_feature_maps (list[torch.Tensor]): Color feature maps (B, C, H, W).
            processed_depth_gradient_map (torch.Tensor | None): Preprocessed depth gradient map (B, D_channels, H, W).
            gradient_mask (torch.Tensor | None): Binary mask (1 if valid gradient, 0 otherwise) (B, 1, H, W).
                                                 Should be float tensor (0.0 or 1.0).
                                                 Can be None if depth is not available.

        Returns:
            list[torch.Tensor]: Fused feature maps (same shapes as input color_feature_maps).
        """
        assert len(color_feature_maps) == self.num_scales, \
            f"Expected {self.num_scales} color feature maps, but got {len(color_feature_maps)}"

        fused_feature_maps = []

        for i, color_feat in enumerate(color_feature_maps):
            _, _, H_i, W_i = color_feat.shape

            if processed_depth_gradient_map is not None and gradient_mask is not None:
                assert processed_depth_gradient_map.shape[1] == self.depth_gradient_channels
                assert gradient_mask.shape[1] == 1 # Mask should be single channel

                # Downsample depth gradient map and mask
                downsampled_depth_grad = F.interpolate(
                    processed_depth_gradient_map,
                    size=(H_i, W_i),
                    mode='bilinear',
                    align_corners=False
                )
                downsampled_mask = F.interpolate(
                    gradient_mask,
                    size=(H_i, W_i),
                    mode='nearest' # No align_corners for nearest
                )

                # --- Apply Gating ---
                # Multiply the downsampled gradient by the downsampled mask
                # This zeros out gradient values where the mask is 0
                gated_depth_grad = downsampled_depth_grad * downsampled_mask # Element-wise multiplication

                # --- Process Gated Depth Gradient ---
                # Use the enhancement layer to process the gated depth features
                # This projects depth features to the same channel space as color features
                depth_enhancement = self.depth_enhancement_layers[i](gated_depth_grad) # (B, color_channels[i], H_i, W_i)

                # --- Additive Fusion and Residual Connection ---
                # Add the depth enhancement signal to the color features
                fused_feat = color_feat + depth_enhancement # Residual connection is implicit here: color_feat + (processed gated depth)

                # Note: If you wanted a structure exactly like your example (output = original + processed_masked_original),
                # you might process the *color_feat* using the *gated_depth_grad* as attention/guidance,
                # then add the result back to the original color_feat.
                # The current implementation adds the *processed gated depth* to the color_feat.
                # This is a common and effective fusion pattern.

            else:
                # If depth or mask is not available, just pass the color feature through
                fused_feat = color_feat

            fused_feature_maps.append(fused_feat)

        return fused_feature_maps


class DepthImageRatioPredictor(nn.Module):
    """
    使用3通道深度图预测window_size_ratio的模块。
    与RatioPredictor兼容，可以直接替换使用。
    """

    def __init__(self, input_channels: int = 3):
        """
        Args:
            input_channels (int): 输入深度图的通道数，默认为3
        """
        super().__init__()
        self.input_channels = input_channels

        # 深度图特征提取网络
        # 使用轻量级的卷积网络来提取深度图特征
        self.depth_feature_extractor = nn.Sequential(
            # 第一层：提取基础特征
            nn.Conv2d(input_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # 降采样

            # 第二层：提取中级特征
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # 降采样

            # 第三层：提取高级特征
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # 降采样

            # 第四层：进一步特征抽象
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )

        # 全局平均池化层，将空间维度压缩为1x1
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)

        # 全连接层，与原RatioPredictor保持相似的结构
        self.fc_layers = nn.Sequential(
            nn.Linear(256, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),  # 添加dropout防止过拟合
            nn.Linear(64, 32),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(32, 1)  # 输出单个标量比率
        )

        # 输出范围约束，与原模块保持一致
        self.output_min = 0.01  # 最小比率
        self.output_max = 0.5  # 最大比率
        self.sigmoid = nn.Sigmoid()

    def forward(self, depth_image: torch.Tensor) -> torch.Tensor:
        """
        Args:
            depth_image (torch.Tensor): 3通道深度图，形状为 (B, 3, H, W)

        Returns:
            torch.Tensor: 预测的window_size_ratio，形状为 (B, 1)
        """
        # 验证输入形状
        assert depth_image.dim() == 4, f"Expected 4D tensor, got {depth_image.dim()}D"
        assert depth_image.shape[1] == self.input_channels, \
            f"Expected {self.input_channels} channels, got {depth_image.shape[1]}"

        # 通过卷积网络提取深度特征
        depth_features = self.depth_feature_extractor(depth_image)  # (B, 256, H', W')

        # 全局平均池化
        pooled_features = self.global_avg_pool(depth_features)  # (B, 256, 1, 1)

        # 展平特征
        flattened_features = pooled_features.squeeze(-1).squeeze(-1)  # (B, 256)

        # 通过全连接层预测比率
        raw_ratio = self.fc_layers(flattened_features)  # (B, 1)

        # 应用约束将输出映射到[output_min, output_max]范围
        predicted_ratio = self.output_min + (self.output_max - self.output_min) * self.sigmoid(raw_ratio)

        return predicted_ratio


class EnhancedDepthImageRatioPredictor(nn.Module):
    """
    增强版的深度图比率预测器，包含多尺度特征提取和注意力机制。
    与RatioPredictor完全兼容。
    """

    def __init__(self, input_channels: int = 3):
        """
        Args:
            input_channels (int): 输入深度图的通道数，默认为3
        """
        super().__init__()
        self.input_channels = input_channels

        # 多尺度特征提取分支
        self.scale1_conv = nn.Sequential(
            nn.Conv2d(input_channels, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

        self.scale2_conv = nn.Sequential(
            nn.Conv2d(input_channels, 64, kernel_size=5, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

        self.scale3_conv = nn.Sequential(
            nn.Conv2d(input_channels, 64, kernel_size=7, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

        # 特征融合层
        self.feature_fusion = nn.Sequential(
            nn.Conv2d(192, 128, kernel_size=1),  # 3*64=192
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )

        # 注意力机制
        self.attention = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=1),
            nn.Sigmoid()
        )

        # 进一步特征提取
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(4),  # 固定输出尺寸

            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )

        # 全局池化
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)

        # 全连接层
        self.fc_layers = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 1)
        )

        # 输出约束
        self.output_min = 0.01
        self.output_max = 0.5
        self.sigmoid = nn.Sigmoid()

    def forward(self, depth_image: torch.Tensor) -> torch.Tensor:
        """
        Args:
            depth_image (torch.Tensor): 3通道深度图，形状为 (B, 3, H, W)

        Returns:
            torch.Tensor: 预测的window_size_ratio，形状为 (B, 1)
        """
        # 验证输入
        assert depth_image.dim() == 4, f"Expected 4D tensor, got {depth_image.dim()}D"
        assert depth_image.shape[1] == self.input_channels, \
            f"Expected {self.input_channels} channels, got {depth_image.shape[1]}"

        # 多尺度特征提取
        scale1_features = self.scale1_conv(depth_image)  # (B, 64, H, W)
        scale2_features = self.scale2_conv(depth_image)  # (B, 64, H, W)
        scale3_features = self.scale3_conv(depth_image)  # (B, 64, H, W)

        # 特征拼接
        multi_scale_features = torch.cat([scale1_features, scale2_features, scale3_features], dim=1)  # (B, 192, H, W)

        # 特征融合
        fused_features = self.feature_fusion(multi_scale_features)  # (B, 128, H, W)

        # 应用注意力机制
        attention_weights = self.attention(fused_features)  # (B, 128, H, W)
        attended_features = fused_features * attention_weights  # (B, 128, H, W)

        # 进一步特征提取
        extracted_features = self.feature_extractor(attended_features)  # (B, 512, 4, 4)

        # 全局平均池化
        pooled_features = self.global_avg_pool(extracted_features)  # (B, 512, 1, 1)

        # 展平
        flattened_features = pooled_features.squeeze(-1).squeeze(-1)  # (B, 512)

        # 预测比率
        raw_ratio = self.fc_layers(flattened_features)  # (B, 1)

        # 应用约束
        predicted_ratio = self.output_min + (self.output_max - self.output_min) * self.sigmoid(raw_ratio)

        return predicted_ratio

# "The features from different modalities of the same scale are always fused,
# while features in different scales are selectively fused."
# (来自相同尺度的不同模态特征总是被融合，而不同尺度的特征则有选择地融合)。