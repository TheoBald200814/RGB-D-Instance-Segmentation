import cv2
import numpy as np
import torch

from mask2former.utils.custom_model import DSAModule
from mask2former.utils.data_process import (calculate_depth_histogram, histogram_viewer,
                                            define_depth_interval_windows, select_depth_distribution_modes,
                                            generate_depth_region_masks, depth_region_viewer)

def depth_filter_process():
    import os
    img_dir = "/Users/theobald/Documents/code_lib/python_lib/shrimpDetection/log/25_03_21/arch_img"
    img_path_list = [os.path.join(img_dir, i) for i in os.listdir(img_dir) if i != '.DS_Store']
    for img_path in img_path_list:
        print(os.path.basename(img_path))
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # 1. 计算深度直方图
        hist, bin_edges = calculate_depth_histogram(img)
        histogram_viewer(hist, bin_edges) # 直方图可视化

        # 2. 选择深度分布模式
        depth_modes = select_depth_distribution_modes(hist, bin_edges, num_modes=3)
        # print(f"Selected Depth Modes: {depth_modes}")

        # 3. 定义深度区间窗口
        interval_windows = define_depth_interval_windows(depth_modes)
        # print(f"Depth Interval Windows: {interval_windows}")

        # 4. 生成深度区域掩码
        region_masks = generate_depth_region_masks(img, interval_windows)
        # print(f"Number of Region Masks: {len(region_masks)}")

        depth_region_viewer(interval_windows, region_masks) # 深度图可视化

def call_dsam():
    # 示例深度图 (NumPy 数组)
    dummy_depth_map_np = np.random.rand(64, 64) * 5.0
    dummy_depth_map_np[10:20, 10:20] += 2.0
    dummy_depth_map_np[40:50, 40:50] += 4.0
    dummy_depth_map_np[dummy_depth_map_np < 0.5] = np.nan
    dummy_depth_map_tensor = torch.from_numpy(dummy_depth_map_np).float().unsqueeze(0).unsqueeze(0) # (1, 1, 64, 64)

    # 示例 RGB 特征图 (PyTorch Tensor)
    input_rgb_features = torch.randn(1, 64, 64, 64) # (B, C_in, H, W)

    # 创建 DSAM 模块实例
    dsam_module = DSAModule(in_channels=64, out_channels=64, num_depth_regions=3) # 可以自定义 in_channels, out_channels, num_depth_regions

    # 执行前向传播
    output_features = dsam_module(input_rgb_features, dummy_depth_map_tensor)

    # 打印输出特征图的形状
    print("DSAM Output Feature Shape:", output_features.shape) # 预期输出: torch.Size([1, 128, 64, 64])


    # 测试深度图输入为 NumPy 数组的情况
    output_features_numpy_depth = dsam_module(input_rgb_features, dummy_depth_map_np)
    print("DSAM Output Feature Shape (NumPy Depth Input):", output_features_numpy_depth.shape) # 预期输出: torch.Size([1, 128, 64, 64])


if __name__ == '__main__':
    depth_filter_process()
    call_dsam()