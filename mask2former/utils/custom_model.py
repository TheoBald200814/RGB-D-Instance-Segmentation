
from typing import List, Optional
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
    define_depth_interval_windows, generate_depth_region_masks, cosine_similarity_fuse_v3, csf_viewer_v2, \
    cosine_similarity_fuse_v3_gpu


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
    def __init__(self, config, rgb_d, image_processor):
        print("[CustomMask2FormerModel] constructing...")
        super().__init__(config)
        self.pixel_level_module = CustomMask2FormerPixelLevelModule(config, rgb_d=rgb_d, image_processor=image_processor)


class CustomMask2FormerForUniversalSegmentation(Mask2FormerForUniversalSegmentation):
    main_input_name = "pixel_values"
    config_class = CustomConfig

    def __init__(self, config, rgb_d='single', image_processor=None):
        print("[CustomMask2FormerForUniversalSegmentation] constructing...")
        super().__init__(config)
        set_seed(42)
        self.model = CustomMask2FormerModel(config, rgb_d=rgb_d, image_processor=image_processor)


class CustomMask2FormerPixelLevelModule(Mask2FormerPixelLevelModule):
    main_input_name = "pixel_values"
    def __init__(self, config, rgb_d, image_processor):
        print("[CustomMask2FormerPixelLevelModule] constructing...")
        super().__init__(config)
        self.rgb_d = rgb_d
        self.image_processor = image_processor
        print(f"[CustomMask2FormerPixelLevelModule] rgb_d={self.rgb_d}")
        if self.rgb_d == 'single': # RGB (3 channel)
            pass # use super().encoder
        elif self.rgb_d == 'multi': # RGB-D (6 channel)
            self.depth_encoder = load_backbone(config)
            self.feature_fuser = FeatureFuser()
        else: # RGB-D (30 channel)
            self.depth_encoder = load_backbone(config)
            self.feature_fuser = FeatureFuser()
            self.dsam1 = DSAModule(in_channels=96, out_channels=96, num_depth_regions=3)
            self.dsam2 = DSAModule(in_channels=96, out_channels=96, num_depth_regions=3)
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
            ahe = pixel_values[:, 12:15, :, :]
            laplace = pixel_values[:, 15:18, :, :]
            gaussian = pixel_values[:, 18:21, :, :]

            # rgb backbone
            rgb_backbone_features = self.encoder(color_input).feature_maps # torch.Tensor
            cp_rgb_backbone_features = list(rgb_backbone_features)

            # DSAM
            dsam_output1 = [self.dsam1(i.unsqueeze(0), self.to_grayscale(j)) for i, j in zip(cp_rgb_backbone_features[0], ahe)] # [B, 96, 64, 64]
            dsam_output1 = torch.stack(dsam_output1, dim=0).squeeze(1)  # [B, 96, 64, 64]
            dsam_output2 = [self.dsam2(i.unsqueeze(0), self.to_grayscale(j)) for i, j in zip(dsam_output1, laplace)] # [B, 96, 64, 64]
            dsam_output2 = torch.stack(dsam_output2, dim=0).squeeze(1)  # [B, 96, 64, 64]
            dsam_output3 = [self.dsam3(i.unsqueeze(0), self.to_grayscale(j)) for i, j in zip(dsam_output2, gaussian)] # [B, 96, 64, 64]
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

            # TODO: 目前丢弃了 Depth Backbone数据流，仅使用融合的EGB Depth Backbone数据流进入Decoder
            # backbone_features = tuple(cp_rgb_backbone_features)

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

    def reverse_image_processor(self, processed_tensor, image_processor):
        """
        反向 image_processor 的预处理操作，将处理后的 Tensor 转换为 OpenCV 可视化的 NumPy 图像。

        Args:
            processed_tensor (torch.Tensor): 经过 image_processor 处理的 Tensor (例如，模型的中间层输出).
                                            假设形状为 [B, C, H, W] 或 [C, H, W].
            image_processor (transformers.AutoImageProcessor): 用于前向预处理的 AutoImageProcessor 实例.

        Returns:
            numpy.ndarray: 反向处理后的 NumPy 图像，形状为 (H, W, C) 或 (H, W)，数据类型为 uint8，
                           可以直接使用 OpenCV (cv2.imshow) 显示.
                           如果输入是 Batch Tensor，则返回 NumPy 图像列表。
                           如果反向处理失败或输入类型不支持，返回 None.
        """
        if not isinstance(processed_tensor, torch.Tensor):
            print("Error: Input processed_tensor must be a PyTorch Tensor.")
            return None

        if not isinstance(image_processor, Mask2FormerImageProcessor):
            print("Error: image_processor must be an instance of transformers.AutoImageProcessor.")
            return None

        # 1. 移除 Batch 维度 (如果存在) 并移动到 CPU
        if processed_tensor.ndim == 4:  # Batch Tensor
            is_batched = True
            tensor_to_reverse = processed_tensor.squeeze(0).cpu()  # 假设 Batch size = 1, 移除 Batch 维度
        elif processed_tensor.ndim == 3:  # 单张图像 Tensor
            is_batched = False
            tensor_to_reverse = processed_tensor.cpu()
        else:
            print(f"Error: Input Tensor ndim={processed_tensor.ndim}, expected 3 or 4.")
            return None

        # 2. 反向 Normalization (如果 image_processor 进行了 Normalization)
        if hasattr(image_processor, 'image_mean') and hasattr(image_processor, 'image_std'):
            mean = torch.tensor(image_processor.image_mean).reshape(-1, 1, 1)  # 调整形状以匹配广播
            std = torch.tensor(image_processor.image_std).reshape(-1, 1, 1)
            tensor_to_reverse = tensor_to_reverse * std + mean

        # 3. Clamp 像素值到 [0, 1] 范围 (或 [0, 255] 如果原始 processor 是针对 0-255 范围的)
        tensor_to_reverse = torch.clamp(tensor_to_reverse, 0, 1)  # 假设 processor 目标范围是 0-1, 常见于 Normalize

        # 4. 将 Tensor 转换为 NumPy array, 并调整通道顺序为 OpenCV 默认的 BGR (如果需要，且假设是 RGB 模型)
        image_np = tensor_to_reverse.permute(1, 2, 0).numpy()  # C, H, W  ->  H, W, C

        if image_np.shape[-1] == 3:  # 如果是彩色图像 (假设是 RGB)
            image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)  # 转换为 BGR for OpenCV (如果模型是 RGB 且 OpenCV 默认 BGR)

        # 5. 缩放到 0-255 范围并转换为 uint8 数据类型 (OpenCV 显示要求)
        image_np = (image_np * 255).astype(np.uint8)

        if is_batched:  # 如果原始输入是 Batch Tensor，则返回图像列表 (目前只处理 Batch size=1)
            return [image_np]  # 返回列表以保持接口一致性
        else:
            return image_np

    def apply_image_processor(self, image_np, image_processor):
        """
        对 NumPy 图像数据应用 image_processor 的前向预处理，转换为 Tensor 数据。

        Args:
            image_np (numpy.ndarray): 输入的 NumPy 图像数据，形状可以是 (H, W, C) 或 (H, W) 或 (C, H, W) 等，
                                      具体取决于 image_processor 的期望输入。
            image_processor (transformers.PreTrainedImageProcessor): 要应用的 PreTrainedImageProcessor 实例
                                                                   (例如 AutoImageProcessor 或 Mask2FormerImageProcessor).

        Returns:
            torch.Tensor: 经过 image_processor 处理后的 Tensor 数据，形状和数据类型由 image_processor 决定。
                          如果输入类型不支持或处理失败，返回 None.
        """
        if not isinstance(image_np, np.ndarray):
            print("Error: Input image_np must be a NumPy ndarray.")
            return None

        if not isinstance(image_processor, Mask2FormerImageProcessor):  # 使用 PreTrainedImageProcessor 进行类型检查
            print("Error: image_processor must be an instance of transformers.PreTrainedImageProcessor.")
            return None

        try:
            # ImageProcessor 期望的输入格式可能是 PIL Image, NumPy array, 或者 list of images.
            # 这里我们尝试直接传入 NumPy array，并假设 image_processor 接受这种格式.
            # 不同的 image_processor 可能对输入格式有不同的要求，需要查阅具体 image_processor 的文档.

            processed_inputs = image_processor(images=image_np,
                                               return_tensors="pt")  # return_tensors="pt" 确保返回 PyTorch Tensor

            #  大多数 ImageProcessor 返回一个字典，其中 'pixel_values' 键对应处理后的 Tensor
            if 'pixel_values' in processed_inputs:
                processed_tensor = processed_inputs.pixel_values
                return processed_tensor
            else:
                print("Error: image_processor did not return 'pixel_values' in its output.")
                return None

        except Exception as e:
            print(f"Error during image_processor application: {e}")
            return None

    def reverse_image_processor_gpu(self, processed_tensor, image_processor):
        """
        反向 image_processor 的预处理操作，直接在 Tensor 上处理并返回 Tensor。

        Args:
            processed_tensor (torch.Tensor): 经过 image_processor 处理的 Tensor (例如，模型的中间层输出).
                                            假设形状为 [B, C, H, W] 或 [C, H, W].
            image_processor (transformers.PreTrainedImageProcessor): 用于前向预处理的 PreTrainedImageProcessor 实例.

        Returns:
            torch.Tensor: 反向处理后的 Tensor，形状与输入 processed_tensor 移除 batch 维度后一致 (C, H, W).
                          如果反向处理失败或输入类型不支持，返回 None.
        """
        if not isinstance(processed_tensor, torch.Tensor):
            print("Error: Input processed_tensor must be a PyTorch Tensor.")
            return None

        if not isinstance(image_processor, Mask2FormerImageProcessor):  # 使用 PreTrainedImageProcessor 进行类型检查
            print("Error: image_processor must be an instance of transformers.PreTrainedImageProcessor.")
            return None

        # 1. 移除 Batch 维度 (如果存在) -  **保持在原始设备上**
        if processed_tensor.ndim == 4:  # Batch Tensor
            tensor_to_reverse = processed_tensor.squeeze(0)  # 假设 Batch size = 1, 移除 Batch 维度
        elif processed_tensor.ndim == 3:  # 单张图像 Tensor
            tensor_to_reverse = processed_tensor
        else:
            print(f"Error: Input Tensor ndim={processed_tensor.ndim}, expected 3 or 4.")
            return None

        # 2. 反向 Normalization (如果 image_processor 进行了 Normalization) - **保持在原始设备上**
        if hasattr(image_processor, 'image_mean') and hasattr(image_processor, 'image_std'):
            mean = torch.tensor(image_processor.image_mean).reshape(-1, 1, 1).to(tensor_to_reverse.device)  # 移动到相同设备
            std = torch.tensor(image_processor.image_std).reshape(-1, 1, 1).to(tensor_to_reverse.device)  # 移动到相同设备
            tensor_to_reverse.mul_(std).add_(
                mean)  # 使用 in-place 操作: *= 和 +=  (等价于 tensor_to_reverse = tensor_to_reverse * std + mean)

        # 3. Clamp 像素值到 [0, 1] 范围 (或 [0, 255] 如果原始 processor 是针对 0-255 范围的) - **保持在原始设备上**
        tensor_to_reverse.clamp_(0,
                                 1)  # 使用 in-place clamp_ 操作 (等价于 tensor_to_reverse = torch.clamp(tensor_to_reverse, 0, 1))

        # 4. 返回 Tensor，形状为 (C, H, W) - **保持在原始设备上，保持 Tensor 类型**
        return tensor_to_reverse

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