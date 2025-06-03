import cv2
from transformers import AutoImageProcessor, Mask2FormerForUniversalSegmentation
from PIL import Image
from types import SimpleNamespace
import torch.nn.functional as F
import torch
import numpy as np
import matplotlib.pyplot as plt
import random
import os
from typing import List, Dict, Union, Optional, Tuple
from pathlib import Path
from tqdm import tqdm


def predictor(image_path, model_path, save=None):
    image_processor = AutoImageProcessor.from_pretrained(os.path.join(model_path, "preprocessor_config.json"))
    model = Mask2FormerForUniversalSegmentation.from_pretrained(model_path)
    image = Image.open(image_path)
    inputs = image_processor(image, return_tensors="pt")

    with torch.no_grad():
        outputs = model(**inputs)

    # Model predicts class_queries_logits of shape `(batch_size, num_queries)`
    # and masks_queries_logits of shape `(batch_size, num_queries, height, width)`
    class_queries_logits = outputs.class_queries_logits
    masks_queries_logits = outputs.masks_queries_logits

    # Perform post-processing to get instance segmentation map
    pred_instance_map = image_processor.post_process_instance_segmentation(
        outputs, target_sizes=[(image.height, image.width)]
    )[0]

    # 假设image是之前加载的PIL.Image对象
    image = np.array(image)  # 将PIL Image转换为NumPy数组
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)  # 转换颜色空间以适应OpenCV (RGB -> BGR)

    # 获取分割掩码
    mask = pred_instance_map['segmentation'].numpy().astype(np.uint8)

    # 创建一个颜色映射，用于给不同实例着色
    colors = np.random.randint(0, 255, (len(pred_instance_map['segments_info']), 3), dtype=np.uint8)

    # 为每个实例创建彩色掩码
    colored_mask = np.zeros_like(image)
    for segment in pred_instance_map['segments_info']:
        color = colors[segment['id']]
        colored_mask[mask == segment['id']] = color

    # 叠加彩色掩码到原始图像上
    alpha = 0.5  # 控制透明度
    overlay = cv2.addWeighted(image, 1 - alpha, colored_mask, alpha, 0)

    if save is not None:
        save_path = os.path.join(save, os.path.basename(image_path))
        cv2.imwrite(save_path, overlay)

    if image.shape == overlay.shape:
        show = cv2.hconcat([image, overlay])
        cv2.imshow('input & output', show)
    else:
        cv2.imshow('input', image)
        cv2.imshow('output', overlay)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def process_prediction(result, image_processor, test_dataset, version, save_dir):
    raw_predictions = result.predictions
    label_ids = result.label_ids

    class_and_mask_logits_np_list = [[batch_class, batch_mask]
                                     for epoch_sample in raw_predictions
                                     for batch_class, batch_mask in zip(epoch_sample[0], epoch_sample[1])]

    class_and_mask_logits_pt_list = [[torch.from_numpy(sample[0]), torch.from_numpy(sample[1])]
                                     for sample in class_and_mask_logits_np_list]

    label_and_id_np_list = [[batch_label, batch_id]
                            for epoch_sample in label_ids
                            for batch_label, batch_id in zip(epoch_sample[0], epoch_sample[1])]
    if version == '0.0.0':
        image_list = [sample.permute(1, 2, 0).numpy().astype("uint8") for sample in test_dataset["image"]]
    else:
        image_list = [sample[0].permute(1, 2, 0).numpy().astype("uint8") for sample in test_dataset["image"]]

    name_list = [os.path.basename(path).split('.')[0] for path in test_dataset["annotation"]]
    original_sizes = get_original_image_sizes_from_image_list(image_list)
    resized_label_ids = resize_masks_to_original_size(label_and_id_np_list, original_sizes)

    sample_class_logits_pt = torch.stack([i[0] for i in class_and_mask_logits_pt_list])
    sample_mask_logits_pt = torch.stack([i[1] for i in class_and_mask_logits_pt_list])
    simulated_outputs = SimpleNamespace(
        class_queries_logits=sample_class_logits_pt,
        masks_queries_logits=sample_mask_logits_pt
    )
    pred_instance_map = image_processor.post_process_instance_segmentation(
        simulated_outputs, target_sizes=original_sizes
    )

    # 示例1：保存完整的对比图像（原图+预测+标签）
    save_comparison_visualization(
        image_list=image_list,
        predicted_instance_maps=pred_instance_map,
        label_data=resized_label_ids,
        save_dir=save_dir,
        layout='horizontal',  # 水平排列
        show_titles=True,
        show_info_text=True,
        image_names=name_list
    )

    # # 示例2：保存简洁的并排对比（仅预测+标签）
    # save_side_by_side_comparison(
    #     image_list=image_list,
    #     predicted_instance_maps=predicted_instance_maps,
    #     label_data=resized_label_ids,
    #     save_dir="./side_by_side_results",
    #     include_original=False  # 不包含原图
    # )
    #
    # # 示例3：保存混合叠加对比图像
    # save_overlay_blend_comparison(
    #     image_list=image_list,
    #     predicted_instance_maps=predicted_instance_maps,
    #     label_data=resized_label_ids,
    #     save_dir="./blend_comparison"
    # )


def resize_masks_to_original_size(label_and_id_np_list: List[List],
                                  original_image_sizes: List[Tuple[int, int]],
                                  processed_size: Tuple[int, int] = (256, 256)) -> List[np.ndarray]:
    """
    将经过image_processor处理的mask标签从处理后的尺寸调整回原始图像尺寸

    Args:
        label_and_id_np_list: 从result.label_ids提取的标签列表，每个元素包含mask和id信息
        original_image_sizes: 原始图像尺寸列表，格式为[(height, width), ...]
        processed_size: image_processor处理后的尺寸，默认为(256, 256)

    Returns:
        调整到原始尺寸的标签列表
    """
    resized_labels = []

    for idx, (labels, original_size) in enumerate(zip(label_and_id_np_list, original_image_sizes)):
        original_height, original_width = original_size

        # 处理每个样本的标签数据
        if len(labels) == 2:  # 假设结构为 [masks, ids]
            masks, ids = labels

            # 将numpy数组转换为torch张量进行插值
            if isinstance(masks, np.ndarray):
                masks_tensor = torch.from_numpy(masks).float()
            else:
                masks_tensor = masks.float()

            # 确保张量形状正确 (N, H, W) 或 (H, W)
            if masks_tensor.dim() == 2:
                masks_tensor = masks_tensor.unsqueeze(0)  # 添加batch维度
            elif masks_tensor.dim() == 4:
                masks_tensor = masks_tensor.squeeze(1)  # 移除可能的通道维度

            # 使用双线性插值调整mask尺寸
            # 对于mask，通常使用nearest插值以保持离散的标签值
            resized_masks = F.interpolate(
                masks_tensor.unsqueeze(1),  # 添加通道维度 (N, 1, H, W)
                size=(original_height, original_width),
                mode='nearest'  # 使用最近邻插值保持标签的离散性
            ).squeeze(1)  # 移除通道维度 (N, H, W)

            # 转换回numpy数组
            resized_masks_np = resized_masks.numpy()

            # 如果原始只有一个mask，移除batch维度
            if resized_masks_np.shape[0] == 1:
                resized_masks_np = resized_masks_np[0]

            # 处理ids（如果需要的话，ids通常不需要调整尺寸）
            resized_labels.append([resized_masks_np, ids])
        else:
            # 如果结构不同，需要根据实际情况调整
            print(f"警告：样本 {idx} 的标签结构不符合预期，跳过处理")
            resized_labels.append(labels)

    return resized_labels


def get_original_image_sizes_from_image_list(image_list: List[np.ndarray]) -> List[Tuple[int, int]]:
    """
    从原始图像列表中提取图像尺寸信息

    Args:
        image_list: 原始图像列表

    Returns:
        图像尺寸列表 [(height, width), ...]
    """
    sizes = []
    for img in image_list:
        if len(img.shape) == 3:  # (H, W, C)
            height, width = img.shape[:2]
        elif len(img.shape) == 2:  # (H, W)
            height, width = img.shape
        else:
            raise ValueError(f"不支持的图像形状: {img.shape}")
        sizes.append((height, width))
    return sizes


def save_comparison_visualization(image_list: List[np.ndarray],
                                  predicted_instance_maps: List[Dict],
                                  label_data: List[np.ndarray],
                                  save_dir: str = "./comparison_results",
                                  alpha: float = 0.6,
                                  class_names: Optional[Dict[int, str]] = None,
                                  image_names: Optional[List[str]] = None,
                                  save_format: str = 'png',
                                  dpi: int = 300,
                                  layout: str = 'horizontal',  # 'horizontal', 'vertical', 'grid'
                                  show_titles: bool = True,
                                  show_info_text: bool = True):
    """
    将标签图和模型预测图保存为一张对比图像

    Args:
        image_list: 原始图像列表
        predicted_instance_maps: 预测结果列表
        label_data: 标签数据列表
        save_dir: 保存目录
        alpha: 透明度
        class_names: 类别名称映射
        image_names: 图像名称列表
        save_format: 保存格式
        dpi: 分辨率
        layout: 布局方式 ('horizontal': 水平排列, 'vertical': 垂直排列, 'grid': 网格排列)
        show_titles: 是否显示标题
        show_info_text: 是否显示实例信息文本
    """

    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    num_images = len(image_list)

    for idx in tqdm(range(num_images)):
        # 获取基础文件名
        if image_names is not None and idx < len(image_names):
            base_name = Path(image_names[idx]).stem
        else:
            base_name = f"image_{idx:04d}"

        # 创建对比图像
        _create_and_save_comparison(
            image_list[idx],
            predicted_instance_maps[idx],
            label_data[idx],
            save_path / f"{base_name}_comparison.{save_format}",
            alpha=alpha,
            class_names=class_names,
            layout=layout,
            show_titles=show_titles,
            show_info_text=False,
            dpi=dpi,
            image_index=idx
        )


def _create_and_save_comparison(original_image: np.ndarray,
                                prediction: Dict,
                                label_info: List[np.ndarray],
                                save_path: Path,
                                alpha: float = 0.6,
                                class_names: Optional[Dict[int, str]] = None,
                                layout: str = 'horizontal',
                                show_titles: bool = True,
                                show_info_text: bool = True,
                                dpi: int = 300,
                                image_index: int = 0):
    """创建并保存对比图像"""

    # 确保图像数据类型正确
    if original_image.dtype != np.uint8:
        if original_image.max() <= 1.0:
            original_image = (original_image * 255).astype(np.uint8)
        else:
            original_image = original_image.astype(np.uint8)

    # 创建预测结果叠加图像
    prediction_overlay = _create_prediction_overlay(
        original_image, prediction, alpha, class_names
    )

    # 创建标签数据叠加图像
    label_overlay = _create_label_overlay(
        original_image, label_info, alpha, class_names
    )

    # 根据布局方式创建图像
    if layout == 'horizontal':
        # 水平排列：原图 | 预测 | 标签
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        images = [original_image, prediction_overlay, label_overlay]
        titles = ['Original Image', 'Predict Image', 'Label Image'] if show_titles else [None, None, None]

    elif layout == 'vertical':
        # 垂直排列：原图 / 预测 / 标签
        fig, axes = plt.subplots(3, 1, figsize=(8, 18))
        images = [original_image, prediction_overlay, label_overlay]
        titles = ['Original Image', 'Predict Image', 'Label Image'] if show_titles else [None, None, None]

    elif layout == 'grid':
        # 2x2网格：原图 | 预测
        #          标签 | 空白
        fig, axes = plt.subplots(2, 2, figsize=(12, 12))
        axes = axes.flatten()
        images = [original_image, prediction_overlay, label_overlay, None]
        titles = ['Original Image', 'Predict Image', 'Label Image', ''] if show_titles else [None, None, None, None]

    # 显示图像
    for i, (ax, img, title) in enumerate(zip(axes, images, titles)):
        if img is not None:
            ax.imshow(img)
            if title:
                ax.set_title(title, fontsize=14, fontweight='bold')
            ax.axis('off')

            # 添加信息文本
            if show_info_text:
                if i == 1:  # 预测结果
                    segments_info = prediction['segments_info']
                    _add_info_text(ax, segments_info, class_names, text_type='prediction')
                elif i == 2:  # 标签数据
                    masks, ids = label_info
                    unique_ids = np.unique(ids[ids > 0]) if hasattr(ids, '__len__') else [ids] if ids > 0 else []
                    _add_label_info_text(ax, unique_ids, class_names)
        else:
            ax.axis('off')  # 隐藏空白子图

    plt.tight_layout()
    plt.savefig(save_path, dpi=dpi, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print(f"已保存对比图像: {save_path}")


def save_side_by_side_comparison(image_list: List[np.ndarray],
                                 predicted_instance_maps: List[Dict],
                                 label_data: List[np.ndarray],
                                 save_dir: str = "./side_by_side_results",
                                 alpha: float = 0.6,
                                 class_names: Optional[Dict[int, str]] = None,
                                 image_names: Optional[List[str]] = None,
                                 save_format: str = 'png',
                                 dpi: int = 300,
                                 include_original: bool = False):
    """
    将预测图和标签图并排保存（仅两张图的简洁版本）

    Args:
        include_original: 是否包含原始图像（True时为三张图，False时为两张图）
    """

    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    num_images = len(image_list)

    for idx in range(num_images):
        # 获取基础文件名
        if image_names is not None and idx < len(image_names):
            base_name = Path(image_names[idx]).stem
        else:
            base_name = f"image_{idx:04d}"

        original_image = image_list[idx].copy()

        # 确保图像数据类型正确
        if original_image.dtype != np.uint8:
            if original_image.max() <= 1.0:
                original_image = (original_image * 255).astype(np.uint8)
            else:
                original_image = original_image.astype(np.uint8)

        # 创建叠加图像
        prediction_overlay = _create_prediction_overlay(
            original_image, predicted_instance_maps[idx], alpha, class_names
        )
        label_overlay = _create_label_overlay(
            original_image, label_data[idx], alpha, class_names
        )

        # 创建并排图像
        if include_original:
            fig, axes = plt.subplots(1, 3, figsize=(18, 6))
            axes[0].imshow(original_image)
            axes[0].set_title('原始图像', fontsize=14, fontweight='bold')
            axes[0].axis('off')

            axes[1].imshow(prediction_overlay)
            axes[1].set_title('模型预测', fontsize=14, fontweight='bold')
            axes[1].axis('off')

            axes[2].imshow(label_overlay)
            axes[2].set_title('真实标签', fontsize=14, fontweight='bold')
            axes[2].axis('off')
        else:
            fig, axes = plt.subplots(1, 2, figsize=(16, 8))
            axes[0].imshow(prediction_overlay)
            axes[0].set_title('模型预测', fontsize=14, fontweight='bold')
            axes[0].axis('off')

            axes[1].imshow(label_overlay)
            axes[1].set_title('真实标签', fontsize=14, fontweight='bold')
            axes[1].axis('off')

        plt.tight_layout()

        filename = f"{base_name}_side_by_side.{save_format}"
        filepath = save_path / filename
        plt.savefig(filepath, dpi=dpi, bbox_inches='tight',
                    facecolor='white', edgecolor='none')
        plt.close()
        print(f"已保存并排对比图像: {filepath}")


def save_overlay_blend_comparison(image_list: List[np.ndarray],
                                  predicted_instance_maps: List[Dict],
                                  label_data: List[np.ndarray],
                                  save_dir: str = "./blend_comparison",
                                  alpha_pred: float = 0.6,
                                  alpha_label: float = 0.6,
                                  image_names: Optional[List[str]] = None,
                                  save_format: str = 'png',
                                  dpi: int = 300):
    """
    创建预测和标签的混合叠加图像（在同一张图上显示两种结果的差异）

    Args:
        alpha_pred: 预测结果的透明度
        alpha_label: 标签的透明度
    """

    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    num_images = len(image_list)

    for idx in range(num_images):
        # 获取基础文件名
        if image_names is not None and idx < len(image_names):
            base_name = Path(image_names[idx]).stem
        else:
            base_name = f"image_{idx:04d}"

        original_image = image_list[idx].copy()

        # 确保图像数据类型正确
        if original_image.dtype != np.uint8:
            if original_image.max() <= 1.0:
                original_image = (original_image * 255).astype(np.uint8)
            else:
                original_image = original_image.astype(np.uint8)

        # 创建混合叠加图像
        blend_image = _create_blend_overlay(
            original_image,
            predicted_instance_maps[idx],
            label_data[idx],
            alpha_pred=alpha_pred,
            alpha_label=alpha_label
        )

        # 保存图像
        plt.figure(figsize=(10, 10))
        plt.imshow(blend_image)
        plt.title(f'预测vs标签混合对比 (红色=预测, 蓝色=标签, 紫色=重叠)',
                  fontsize=14, fontweight='bold')
        plt.axis('off')

        filename = f"{base_name}_blend_comparison.{save_format}"
        filepath = save_path / filename
        plt.savefig(filepath, dpi=dpi, bbox_inches='tight',
                    facecolor='white', edgecolor='none')
        plt.close()
        print(f"已保存混合对比图像: {filepath}")


def _create_blend_overlay(original_image: np.ndarray,
                          prediction: Dict,
                          label_info: List[np.ndarray],
                          alpha_pred: float = 0.6,
                          alpha_label: float = 0.6) -> np.ndarray:
    """创建预测和标签的混合叠加图像"""

    # 获取预测分割图
    segmentation_tensor = prediction['segmentation']
    segments_info = prediction['segments_info']

    if isinstance(segmentation_tensor, torch.Tensor):
        pred_segmentation = segmentation_tensor.cpu().numpy()
    else:
        pred_segmentation = segmentation_tensor

    # 获取标签分割图
    masks, ids = label_info
    if masks.ndim == 2:
        label_segmentation = masks
    else:
        # 如果是多个mask，需要合并
        label_segmentation = np.zeros(masks.shape[1:], dtype=np.int32)
        for i in range(masks.shape[0]):
            if ids[i] > 0:
                label_segmentation[masks[i] > 0] = ids[i]

    # 创建叠加图像
    overlay_image = original_image.copy().astype(np.float32)

    # 创建预测mask（红色通道）
    pred_mask = np.zeros_like(pred_segmentation, dtype=bool)
    for segment in segments_info:
        segment_id = segment['id']
        pred_mask |= (pred_segmentation == segment_id)

    # 创建标签mask（蓝色通道）
    label_mask = label_segmentation > 0

    # 应用颜色
    # 预测区域用红色
    overlay_image[pred_mask, 0] = (1 - alpha_pred) * overlay_image[pred_mask, 0] + alpha_pred * 255

    # 标签区域用蓝色
    overlay_image[label_mask, 2] = (1 - alpha_label) * overlay_image[label_mask, 2] + alpha_label * 255

    # 重叠区域会显示为紫色（红+蓝）

    return np.clip(overlay_image, 0, 255).astype(np.uint8)


def _create_prediction_overlay(original_image: np.ndarray,
                               prediction: Dict,
                               alpha: float,
                               class_names: Optional[Dict[int, str]] = None) -> np.ndarray:
    """创建预测结果的叠加图像"""
    segmentation_tensor = prediction['segmentation']
    segments_info = prediction['segments_info']

    # 转换张量为numpy数组
    if isinstance(segmentation_tensor, torch.Tensor):
        segmentation_map = segmentation_tensor.cpu().numpy()
    else:
        segmentation_map = segmentation_tensor

    # 创建叠加图像
    overlay_image = original_image.copy().astype(np.float32)

    # 为每个实例生成随机颜色
    for segment in segments_info:
        segment_id = segment['id']

        # 生成随机颜色
        color = [random.randint(50, 255) for _ in range(3)]

        # 创建当前实例的mask
        instance_mask = (segmentation_map == segment_id)

        # 将颜色应用到mask区域
        for c in range(3):
            overlay_image[instance_mask, c] = (
                    (1 - alpha) * overlay_image[instance_mask, c] +
                    alpha * color[c]
            )

    return np.clip(overlay_image, 0, 255).astype(np.uint8)


def _create_label_overlay(original_image: np.ndarray,
                          label_info: List[np.ndarray],
                          alpha: float,
                          class_names: Optional[Dict[int, str]] = None) -> np.ndarray:
    """创建标签数据的叠加图像"""
    masks, ids = label_info

    # 创建叠加图像
    overlay_image = original_image.copy().astype(np.float32)

    # 获取所有唯一的实例ID（排除背景0）
    if masks.ndim == 3:  # 多个mask (N, H, W)
        for i in range(masks.shape[0]):
            mask = masks[i]
            instance_id = ids[i] if hasattr(ids, '__len__') else ids

            if instance_id > 0:  # 排除背景
                # 生成随机颜色
                color = [random.randint(50, 255) for _ in range(3)]

                # 创建当前实例的mask
                instance_mask = (mask > 0)

                # 将颜色应用到mask区域
                for c in range(3):
                    overlay_image[instance_mask, c] = (
                            (1 - alpha) * overlay_image[instance_mask, c] +
                            alpha * color[c]
                    )

    elif masks.ndim == 2:  # 单个分割图 (H, W)，每个像素值代表实例ID
        unique_ids = np.unique(masks)
        unique_ids = unique_ids[unique_ids > 0]  # 排除背景

        for instance_id in unique_ids:
            # 生成随机颜色
            color = [random.randint(50, 255) for _ in range(3)]

            # 创建当前实例的mask
            instance_mask = (masks == instance_id)

            # 将颜色应用到mask区域
            for c in range(3):
                overlay_image[instance_mask, c] = (
                        (1 - alpha) * overlay_image[instance_mask, c] +
                        alpha * color[c]
                )

    return np.clip(overlay_image, 0, 255).astype(np.uint8)


def _add_info_text(ax, segments_info: List[Dict],
                   class_names: Optional[Dict[int, str]] = None,
                   text_type: str = 'prediction'):
    """在图像上添加实例信息文本"""
    y_offset = 10
    for i, segment in enumerate(segments_info[:5]):  # 最多显示5个实例的信息
        label_id = segment['label_id']
        score = segment.get('score', 0.0)

        # 获取类别名称
        class_name = class_names.get(label_id, f'Class_{label_id}') if class_names else f'ID:{label_id}'

        if text_type == 'prediction':
            text = f'{class_name}, Score:{score:.3f}'
        else:
            text = f'{class_name}'

        ax.text(10, y_offset, text,
                color='white', fontsize=8,
                bbox=dict(boxstyle="round,pad=0.3", facecolor='black', alpha=0.7))
        y_offset += 20


def _add_label_info_text(ax, unique_ids: np.ndarray,
                         class_names: Optional[Dict[int, str]] = None):
    """在标签图像上添加实例信息文本"""
    y_offset = 10
    for i, instance_id in enumerate(unique_ids[:5]):  # 最多显示5个实例的信息
        # 获取类别名称
        class_name = class_names.get(instance_id, f'Instance_{instance_id}') if class_names else f'ID:{instance_id}'

        ax.text(10, y_offset, class_name,
                color='white', fontsize=8,
                bbox=dict(boxstyle="round,pad=0.3", facecolor='blue', alpha=0.7))
        y_offset += 20


def main():
    image_path = "/Users/theobald/Documents/code_lib/python_lib/huggingface_data_process/datasets/local/360.jpg"
    model_path = "/Users/theobald/Documents/code_lib/python_lib/huggingface_data_process/checkpoint/local/checkpoint-868"
    predictor(image_path, model_path)


if __name__ == '__main__':
    main()