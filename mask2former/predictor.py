import json

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
from pycocotools import mask as coco_mask


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


def calculate_mask_iou(mask1: np.ndarray, mask2: np.ndarray) -> float:
    """
    计算两个掩码的IoU

    Args:
        mask1: 第一个二值掩码
        mask2: 第二个二值掩码

    Returns:
        IoU值
    """
    if mask1.shape != mask2.shape:
        return 0.0

    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()

    if union == 0:
        return 0.0

    return intersection / union


def match_predictions_to_gt(
        prediction_masks: List[np.ndarray],
        prediction_labels: List[int],
        prediction_scores: List[float],
        gt_masks: List[np.ndarray],
        gt_labels: List[int],
        iou_threshold: float = 0.5
) -> Tuple[List[int], List[bool], List[int]]:
    """
    将预测结果与GT进行匹配

    Args:
        prediction_masks: 预测掩码列表
        prediction_labels: 预测标签列表
        prediction_scores: 预测分数列表
        gt_masks: GT掩码列表
        gt_labels: GT标签列表
        iou_threshold: IoU阈值

    Returns:
        matched_gt_indices: 每个预测对应的GT索引（-1表示无匹配）
        is_matched: 每个预测是否匹配成功
        gt_match_counts: 每个GT被匹配的次数
    """
    num_predictions = len(prediction_masks)
    num_gt = len(gt_masks)

    matched_gt_indices = [-1] * num_predictions
    is_matched = [False] * num_predictions
    gt_match_counts = [0] * num_gt

    # 计算所有IoU
    iou_matrix = np.zeros((num_predictions, num_gt))
    for i, pred_mask in enumerate(prediction_masks):
        for j, gt_mask in enumerate(gt_masks):
            # 可选：额外检查标签是否匹配（如果标签系统一致的话）
            # if prediction_labels[i] == gt_labels[j]:
            iou_matrix[i, j] = calculate_mask_iou(pred_mask, gt_mask)

    # 贪心匹配：按IoU从高到低进行匹配
    used_gt = set()

    # 获取所有IoU > threshold的配对，按IoU降序排列
    candidates = []
    for i in range(num_predictions):
        for j in range(num_gt):
            if iou_matrix[i, j] >= iou_threshold:
                candidates.append((i, j, iou_matrix[i, j]))

    # 按IoU降序排序
    candidates.sort(key=lambda x: x[2], reverse=True)

    # 进行匹配
    for pred_idx, gt_idx, iou_val in candidates:
        if not is_matched[pred_idx] and gt_idx not in used_gt:
            matched_gt_indices[pred_idx] = gt_idx
            is_matched[pred_idx] = True
            gt_match_counts[gt_idx] += 1
            used_gt.add(gt_idx)

    return matched_gt_indices, is_matched, gt_match_counts


def generate_consistent_colors(num_colors: int, seed: int = 42) -> List[Tuple[int, int, int]]:
    """
    生成一致的颜色列表

    Args:
        num_colors: 需要的颜色数量
        seed: 随机种子，确保颜色一致性

    Returns:
        颜色列表
    """
    np.random.seed(seed)
    colors = []
    for i in range(num_colors):
        # 生成较鲜艳的颜色，避免过暗
        color = [np.random.randint(50, 255) for _ in range(3)]
        colors.append(tuple(color))

    # 重置随机种子避免影响其他随机操作
    np.random.seed()
    return colors


def _create_consistent_prediction_overlay(
        original_image: np.ndarray,
        prediction: Dict,
        gt_color_mapping: Dict[int, Tuple[int, int, int]],
        gt_prediction: Optional[Dict] = None,
        alpha: float = 0.6,
        iou_threshold: float = 0.5,
        unmatched_color: Tuple[int, int, int] = (255, 0, 0),  # 红色表示误检
        class_names: Optional[Dict[int, str]] = None
) -> np.ndarray:
    """
    创建与GT颜色一致的预测结果叠加图像

    Args:
        original_image: 原始图像
        prediction: 预测结果
        gt_color_mapping: GT实例ID到颜色的映射
        gt_prediction: GT预测结果（用于匹配）
        alpha: 透明度
        iou_threshold: IoU匹配阈值
        unmatched_color: 未匹配实例的颜色
        class_names: 类别名称映射

    Returns:
        叠加后的图像
    """
    segmentation_tensor = prediction['segmentation']
    segments_info = prediction['segments_info']

    # 转换张量为numpy数组
    if isinstance(segmentation_tensor, torch.Tensor):
        segmentation_map = segmentation_tensor.cpu().numpy()
    else:
        segmentation_map = segmentation_tensor

    # 创建叠加图像
    overlay_image = original_image.copy().astype(np.float32)

    # 如果没有GT数据，使用原有的随机颜色逻辑
    if gt_prediction is None:
        return _create_prediction_overlay(original_image, prediction, alpha, class_names)

    # 提取预测掩码和信息
    prediction_masks = []
    prediction_labels = []
    prediction_scores = []

    for segment in segments_info:
        segment_id = segment['id']
        mask = (segmentation_map == segment_id)
        if np.sum(mask) > 0:  # 确保掩码不为空
            prediction_masks.append(mask)
            prediction_labels.append(segment['label_id'])
            prediction_scores.append(segment.get('score', 1.0))

    # 提取GT掩码和信息
    gt_segmentation = gt_prediction['segmentation']
    gt_segments_info = gt_prediction['segments_info']

    if isinstance(gt_segmentation, torch.Tensor):
        gt_segmentation = gt_segmentation.cpu().numpy()

    gt_masks = []
    gt_labels = []
    gt_ids = []

    for gt_segment in gt_segments_info:
        gt_id = gt_segment['id']
        gt_mask = (gt_segmentation == gt_id)
        if np.sum(gt_mask) > 0:
            gt_masks.append(gt_mask)
            gt_labels.append(gt_segment['label_id'])
            gt_ids.append(gt_id)

    # 进行匹配
    if prediction_masks and gt_masks:
        matched_gt_indices, is_matched, gt_match_counts = match_predictions_to_gt(
            prediction_masks, prediction_labels, prediction_scores,
            gt_masks, gt_labels, iou_threshold
        )
    else:
        matched_gt_indices = []
        is_matched = []

    # 应用颜色
    for i, segment in enumerate(segments_info):
        segment_id = segment['id']
        instance_mask = (segmentation_map == segment_id)

        if np.sum(instance_mask) == 0:
            continue

        # 确定使用的颜色
        if i < len(is_matched) and is_matched[i]:
            # 匹配成功，使用GT颜色
            gt_idx = matched_gt_indices[i]
            if gt_idx >= 0 and gt_idx < len(gt_ids):
                gt_id = gt_ids[gt_idx]
                color = gt_color_mapping.get(gt_id, unmatched_color)
            else:
                color = unmatched_color
        else:
            # 未匹配，使用特殊颜色
            color = unmatched_color

        # 应用颜色到mask区域
        for c in range(3):
            overlay_image[instance_mask, c] = (
                    (1 - alpha) * overlay_image[instance_mask, c] + alpha * color[c]
            )

    return np.clip(overlay_image, 0, 255).astype(np.uint8)


def _create_gt_overlay_with_consistent_colors(
        original_image: np.ndarray,
        gt_prediction: Dict,
        gt_color_mapping: Dict[int, Tuple[int, int, int]],
        alpha: float = 0.6
) -> np.ndarray:
    """
    使用一致颜色映射创建GT叠加图像
    """
    segmentation_tensor = gt_prediction['segmentation']
    segments_info = gt_prediction['segments_info']

    # 转换张量为numpy数组
    if isinstance(segmentation_tensor, torch.Tensor):
        segmentation_map = segmentation_tensor.cpu().numpy()
    else:
        segmentation_map = segmentation_tensor

    # 创建叠加图像
    overlay_image = original_image.copy().astype(np.float32)

    # 为每个实例应用对应的颜色
    for segment in segments_info:
        segment_id = segment['id']
        color = gt_color_mapping.get(segment_id, (128, 128, 128))  # 默认灰色

        # 创建当前实例的mask
        instance_mask = (segmentation_map == segment_id)

        # 将颜色应用到mask区域
        for c in range(3):
            overlay_image[instance_mask, c] = (
                    (1 - alpha) * overlay_image[instance_mask, c] + alpha * color[c]
            )

    return np.clip(overlay_image, 0, 255).astype(np.uint8)


def convert_model_a_to_json_format(
        predicted_instance_maps: List[Dict],
        image_names: List[str],
        save_dir: str,
        original_sizes: Optional[List[Tuple[int, int]]] = None
):
    """
    将模型A的预测结果转换为统一的JSON格式并保存

    Args:
        predicted_instance_maps: 模型A的预测结果列表
        image_names: 图像名称列表（不含扩展名）
        save_dir: 保存JSON文件的目录
        original_sizes: 原始图像尺寸列表 [(height, width), ...]，如果为None则从分割图中获取
    """
    import json
    from pycocotools import mask as coco_mask

    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    for idx, (prediction, image_name) in enumerate(tqdm(
            zip(predicted_instance_maps, image_names),
            desc="转换模型A输出为JSON格式"
    )):
        try:
            json_data = _convert_single_prediction_to_json(
                prediction,
                original_sizes[idx] if original_sizes else None
            )

            # 保存JSON文件
            json_file_path = save_path / f"{image_name}.json"
            with open(json_file_path, 'w') as f:
                json.dump(json_data, f, indent=2)

            print(f"已保存: {json_file_path}")

        except Exception as e:
            print(f"处理图像 {image_name} 时出错: {e}")
            continue


def _convert_single_prediction_to_json(prediction: Dict, original_size: Optional[Tuple[int, int]] = None) -> Dict:
    """
    将单个预测结果转换为JSON格式

    Args:
        prediction: 单个图像的预测结果，包含 'segmentation' 和 'segments_info'
        original_size: 原始图像尺寸 (height, width)

    Returns:
        JSON格式的预测结果字典
    """
    from pycocotools import mask as coco_mask

    # 提取分割图和段信息
    segmentation_tensor = prediction['segmentation']
    segments_info = prediction['segments_info']

    # 转换张量为numpy数组
    if isinstance(segmentation_tensor, torch.Tensor):
        segmentation_map = segmentation_tensor.cpu().numpy()
    else:
        segmentation_map = segmentation_tensor

    # 获取图像尺寸
    if original_size is not None:
        height, width = original_size
    else:
        height, width = segmentation_map.shape[:2]

    # 初始化输出列表
    labels = []
    scores = []
    bboxes = []
    masks = []

    # 处理每个实例
    for segment in segments_info:
        segment_id = segment['id']
        label_id = segment['label_id']
        score = segment.get('score', 1.0)  # 如果没有score，默认为1.0

        # 创建该实例的二值掩码
        binary_mask = (segmentation_map == segment_id).astype(np.uint8)

        # 检查掩码是否为空
        if np.sum(binary_mask) == 0:
            print(f"警告: 实例ID {segment_id} 的掩码为空，跳过")
            continue

        # 计算边界框 [x, y, width, height]
        bbox = _calculate_bbox_from_mask(binary_mask)
        if bbox is None:
            print(f"警告: 无法为实例ID {segment_id} 计算边界框，跳过")
            continue

        # 将二值掩码转换为RLE格式
        # pycocotools期望的格式是Fortran order (column-major)
        binary_mask_fortran = np.asfortranarray(binary_mask)
        rle = coco_mask.encode(binary_mask_fortran)

        # 确保RLE格式正确
        if isinstance(rle['counts'], bytes):
            rle['counts'] = rle['counts'].decode('utf-8')

        # 添加到输出列表
        labels.append(int(label_id))
        scores.append(float(score))
        bboxes.append([float(x) for x in bbox])
        masks.append({
            'size': [int(height), int(width)],
            'counts': rle['counts']
        })

    # 构建最终的JSON格式数据
    json_data = {
        'labels': labels,
        'scores': scores,
        'bboxes': bboxes,
        'masks': masks
    }

    return json_data


def _calculate_bbox_from_mask(binary_mask: np.ndarray) -> Optional[List[float]]:
    """
    从二值掩码计算边界框

    Args:
        binary_mask: 二值掩码，形状为 (H, W)

    Returns:
        边界框 [x, y, width, height]，如果计算失败返回None
    """
    # 找到非零像素的坐标
    y_coords, x_coords = np.where(binary_mask > 0)

    if len(y_coords) == 0 or len(x_coords) == 0:
        return None

    # 计算边界框
    x_min = np.min(x_coords)
    x_max = np.max(x_coords)
    y_min = np.min(y_coords)
    y_max = np.max(y_coords)

    # 转换为 [x, y, width, height] 格式
    bbox = [
        float(x_min),
        float(y_min),
        float(x_max - x_min + 1),
        float(y_max - y_min + 1)
    ]

    return bbox


def convert_gt_labels_to_json_format(
        label_data: List[List[np.ndarray]],
        image_names: List[str],
        save_dir: str,
        original_sizes: Optional[List[Tuple[int, int]]] = None
):
    """
    将GT标签数据转换为统一的JSON格式并保存

    Args:
        label_data: GT标签数据列表，每个元素包含 [masks, ids]
        image_names: 图像名称列表（不含扩展名）
        save_dir: 保存JSON文件的目录
        original_sizes: 原始图像尺寸列表 [(height, width), ...]
    """
    import json
    from pycocotools import mask as coco_mask

    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    for idx, (label_info, image_name) in enumerate(tqdm(
            zip(label_data, image_names),
            desc="转换GT标签为JSON格式"
    )):
        try:
            json_data = _convert_single_gt_label_to_json(
                label_info,
                original_sizes[idx] if original_sizes else None
            )

            # 保存JSON文件
            json_file_path = save_path / f"{image_name}.json"
            with open(json_file_path, 'w') as f:
                json.dump(json_data, f, indent=2)

            print(f"已保存GT JSON: {json_file_path}")

        except Exception as e:
            print(f"处理GT标签 {image_name} 时出错: {e}")
            continue


def _convert_single_gt_label_to_json(label_info: List[np.ndarray],
                                     original_size: Optional[Tuple[int, int]] = None) -> Dict:
    """
    将单个GT标签转换为JSON格式

    Args:
        label_info: GT标签信息，格式为 [masks, ids]
        original_size: 原始图像尺寸 (height, width)

    Returns:
        JSON格式的GT标签字典
    """
    from pycocotools import mask as coco_mask

    masks, ids = label_info

    # 初始化输出列表
    labels = []
    scores = []
    bboxes = []
    masks_rle = []

    # 确定图像尺寸
    if original_size is not None:
        height, width = original_size
    else:
        if masks.ndim == 3:  # 多个mask (N, H, W)
            height, width = masks.shape[1:]
        elif masks.ndim == 2:  # 单个分割图 (H, W)
            height, width = masks.shape
        else:
            raise ValueError(f"不支持的masks维度: {masks.shape}")

    if masks.ndim == 3:  # 多个mask (N, H, W)
        # 处理每个单独的mask
        for i in range(masks.shape[0]):
            mask = masks[i]
            if hasattr(ids, '__len__'):
                instance_id = ids[i]
            else:
                instance_id = ids

            if instance_id <= 0:  # 跳过背景或无效ID
                continue

            # 确保mask是二值的
            binary_mask = (mask > 0).astype(np.uint8)

            # 检查掩码是否为空
            if np.sum(binary_mask) == 0:
                continue

            # 计算边界框
            bbox = _calculate_bbox_from_mask(binary_mask)
            if bbox is None:
                continue

            # 将二值掩码转换为RLE格式
            binary_mask_fortran = np.asfortranarray(binary_mask)
            rle = coco_mask.encode(binary_mask_fortran)

            # 确保RLE格式正确
            if isinstance(rle['counts'], bytes):
                rle['counts'] = rle['counts'].decode('utf-8')

            # 添加到输出列表
            labels.append(int(instance_id))
            scores.append(1.0)  # GT标签的置信度设为1.0
            bboxes.append([float(x) for x in bbox])
            masks_rle.append({
                'size': [int(height), int(width)],
                'counts': rle['counts']
            })

    elif masks.ndim == 2:  # 单个分割图 (H, W)，每个像素值代表实例ID
        # 获取所有唯一的实例ID（排除背景0）
        unique_ids = np.unique(masks)
        unique_ids = unique_ids[unique_ids > 0]  # 排除背景

        for instance_id in unique_ids:
            # 创建当前实例的二值掩码
            binary_mask = (masks == instance_id).astype(np.uint8)

            # 检查掩码是否为空
            if np.sum(binary_mask) == 0:
                continue

            # 计算边界框
            bbox = _calculate_bbox_from_mask(binary_mask)
            if bbox is None:
                continue

            # 将二值掩码转换为RLE格式
            binary_mask_fortran = np.asfortranarray(binary_mask)
            rle = coco_mask.encode(binary_mask_fortran)

            # 确保RLE格式正确
            if isinstance(rle['counts'], bytes):
                rle['counts'] = rle['counts'].decode('utf-8')

            # 添加到输出列表
            labels.append(int(instance_id))
            scores.append(1.0)  # GT标签的置信度设为1.0
            bboxes.append([float(x) for x in bbox])
            masks_rle.append({
                'size': [int(height), int(width)],
                'counts': rle['counts']
            })

    # 构建最终的JSON格式数据
    json_data = {
        'labels': labels,
        'scores': scores,
        'bboxes': bboxes,
        'masks': masks_rle
    }

    return json_data


def process_prediction(result, image_processor, test_dataset, version, save_dir=None,
                       save_json_format=False, json_save_dir=None,
                       save_gt_json=False, gt_json_save_dir=None):
    """
    处理预测结果并可视化

    Args:
        result: 模型A的预测结果
        image_processor: 图像处理器
        test_dataset: 测试数据集
        version: 版本号
        save_dir: 保存目录
        save_json_format: 是否保存模型A预测结果为JSON格式
        json_save_dir: 模型A JSON文件保存目录，如果为None则使用 save_dir + "_json"
        save_gt_json: 是否保存GT标签为JSON格式
        gt_json_save_dir: GT JSON文件保存目录，如果为None则使用 save_dir + "_gt_json"
    """
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

    if save_dir is not None:
        # 保存常规可视化结果
        save_comparison_visualization(
            image_list=image_list,
            predicted_instance_maps=pred_instance_map,
            label_data=resized_label_ids,
            save_dir=save_dir,
            layout='horizontal',
            show_titles=True,
            show_info_text=True,
            image_names=name_list
        )

    # 如果需要，将模型A的结果转换为JSON格式并保存
    if save_json_format:
        if json_save_dir is None:
            json_save_dir = save_dir + "_json"

        print(f"\n正在将模型A的预测结果转换为JSON格式...")
        convert_model_a_to_json_format(
            predicted_instance_maps=pred_instance_map,
            image_names=name_list,
            save_dir=json_save_dir,
            original_sizes=original_sizes
        )
        print(f"模型A JSON格式结果已保存到: {json_save_dir}")

    # 如果需要，将GT标签转换为JSON格式并保存
    if save_gt_json:
        if gt_json_save_dir is None:
            gt_json_save_dir = save_dir + "_gt_json"

        print(f"\n正在将GT标签转换为JSON格式...")
        convert_gt_labels_to_json_format(
            label_data=resized_label_ids,
            image_names=name_list,
            save_dir=gt_json_save_dir,
            original_sizes=original_sizes
        )
        print(f"GT JSON格式结果已保存到: {gt_json_save_dir}")


def visualize_multi_model_json_results(
        image_paths: List[str],
        gt_json_paths: List[str],
        model_json_paths: List[List[str]],
        save_dir: str,
        model_names: Optional[List[str]] = None,
        alpha: float = 0.6,
        class_names: Optional[Dict[int, str]] = None,
        save_format: str = 'png',
        dpi: int = 300,
        show_titles: bool = True,
        show_info_text: bool = False,
        max_models_per_row: int = 3,
        iou_threshold: float = 0.5,  # 新增参数
        color_seed: int = 42  # 新增参数
):
    """
    可视化多模型JSON格式的预测结果对比图

    Args:
        image_paths: 原始图像路径列表
        gt_json_paths: GT标签JSON文件路径列表
        model_json_paths: 多个模型的JSON预测文件路径列表，格式为 [[model1_files], [model2_files], ...]
        save_dir: 保存可视化结果的目录
        model_names: 模型名称列表，如果为None则使用 "Model 1", "Model 2" 等
        alpha: 叠加透明度
        class_names: 类别ID到名称的映射
        save_format: 保存格式
        dpi: 分辨率
        show_titles: 是否显示标题
        show_info_text: 是否显示详细信息文本
        max_models_per_row: 每行最多显示的模型数量
    """
    import json
    from pycocotools import mask as coco_mask

    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    # 检查输入数据长度一致性
    num_images = len(image_paths)
    if len(gt_json_paths) != num_images:
        raise ValueError("图像数量与GT JSON文件数量不匹配")

    num_models = len(model_json_paths)
    for i, model_files in enumerate(model_json_paths):
        if len(model_files) != num_images:
            raise ValueError(f"模型{i + 1}的JSON文件数量与图像数量不匹配")

    # 设置默认模型名称
    if model_names is None:
        model_names = [f"Model {i + 1}" for i in range(num_models)]
    elif len(model_names) != num_models:
        raise ValueError("模型名称数量与模型数量不匹配")

    # 处理每张图像
    for idx in tqdm(range(num_images), desc="生成多模型对比可视化"):
        image_path = image_paths[idx]
        gt_json_path = gt_json_paths[idx]
        current_model_files = [model_files[idx] for model_files in model_json_paths]

        # 获取图像名称
        image_name = Path(image_path).stem

        try:
            _create_and_save_multi_model_json_comparison(
                image_path=image_path,
                gt_json_path=gt_json_path,
                model_json_paths=current_model_files,
                model_names=model_names,
                save_path=save_path / f"{image_name}_multi_model_comparison.{save_format}",
                alpha=alpha,
                class_names=class_names,
                dpi=dpi,
                show_titles=show_titles,
                show_info_text=show_info_text,
                max_models_per_row=max_models_per_row,
                iou_threshold=iou_threshold,  # 新增
                color_seed=color_seed  # 新增
            )
        except Exception as e:
            print(f"处理图像 {image_name} 时出错: {e}")
            continue


def _create_and_save_multi_model_json_comparison(
        image_path: str,
        gt_json_path: str,
        model_json_paths: List[str],
        model_names: List[str],
        save_path: Path,
        alpha: float = 0.6,
        class_names: Optional[Dict[int, str]] = None,
        dpi: int = 300,
        show_titles: bool = True,
        show_info_text: bool = False,
        max_models_per_row: int = 3,
        iou_threshold: float = 0.5,  # 新增参数
        color_seed: int = 42  # 新增参数
):
    """创建并保存多模型JSON对比图像（使用一致的颜色映射）"""

    # 1. 加载原始图像
    try:
        pil_image = Image.open(image_path).convert("RGB")
        original_image = np.array(pil_image)
    except Exception as e:
        print(f"加载图像 {image_path} 失败: {e}")
        return

    # 确保图像数据类型正确
    if original_image.dtype != np.uint8:
        if original_image.max() <= 1.0:
            original_image = (original_image * 255).astype(np.uint8)
        else:
            original_image = original_image.astype(np.uint8)

    # 2. 加载GT标签
    gt_prediction = None
    try:
        gt_prediction = _load_json_prediction_data(gt_json_path, original_image.shape[:2])
    except Exception as e:
        print(f"加载GT JSON {gt_json_path} 失败: {e}")

    # 3. 生成GT颜色映射
    gt_color_mapping = {}
    if gt_prediction is not None:
        gt_instance_ids = [segment['id'] for segment in gt_prediction['segments_info']]
        if gt_instance_ids:
            consistent_colors = generate_consistent_colors(len(gt_instance_ids), seed=color_seed)
            gt_color_mapping = dict(zip(gt_instance_ids, consistent_colors))

    # 4. 创建GT叠加图像
    if gt_prediction is not None:
        gt_overlay = _create_gt_overlay_with_consistent_colors(
            original_image, gt_prediction, gt_color_mapping, alpha
        )
    else:
        gt_overlay = original_image.copy()

    # 5. 加载各模型预测结果并创建一致颜色的叠加图像
    model_overlays = []
    for model_json_path, model_name in zip(model_json_paths, model_names):
        try:
            model_prediction = _load_json_prediction_data(model_json_path, original_image.shape[:2])
            if model_prediction is not None:
                model_overlay = _create_consistent_prediction_overlay(
                    original_image=original_image,
                    prediction=model_prediction,
                    gt_color_mapping=gt_color_mapping,
                    gt_prediction=gt_prediction,
                    alpha=alpha,
                    iou_threshold=iou_threshold,
                    class_names=class_names
                )
            else:
                model_overlay = original_image.copy()
        except Exception as e:
            print(f"加载模型 {model_name} JSON {model_json_path} 失败: {e}")
            model_overlay = original_image.copy()

        model_overlays.append((model_overlay, model_name))

    # 6. 计算布局（与原来相同）
    num_models = len(model_overlays)
    total_subplots = 2 + num_models  # 原图 + GT + 各模型

    # 计算网格布局
    cols = min(max_models_per_row + 2, total_subplots)
    rows = (total_subplots + cols - 1) // cols

    # 7. 创建可视化图像
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 6, rows * 5))

    # 如果只有一行，确保axes是二维数组
    if rows == 1:
        axes = axes.reshape(1, -1)
    elif cols == 1:
        axes = axes.reshape(-1, 1)

    # 展平axes数组以便索引
    axes_flat = axes.flatten()

    current_idx = 0

    # 显示原始图像
    axes_flat[current_idx].imshow(original_image)
    if show_titles:
        axes_flat[current_idx].set_title('Original Image', fontsize=14, fontweight='bold')
    axes_flat[current_idx].axis('off')
    current_idx += 1

    # 显示GT（使用一致颜色）
    axes_flat[current_idx].imshow(gt_overlay)
    if show_titles:
        axes_flat[current_idx].set_title('Ground Truth', fontsize=14, fontweight='bold', color='green')
    axes_flat[current_idx].axis('off')

    if show_info_text and gt_prediction is not None:
        _add_info_text(axes_flat[current_idx], gt_prediction['segments_info'], class_names, text_type='gt')

    current_idx += 1

    # 显示各模型预测结果（使用一致颜色）
    for model_overlay, model_name in model_overlays:
        axes_flat[current_idx].imshow(model_overlay)
        if show_titles:
            axes_flat[current_idx].set_title(model_name, fontsize=14, fontweight='bold')
        axes_flat[current_idx].axis('off')
        current_idx += 1

    # 隐藏多余的子图
    for i in range(current_idx, len(axes_flat)):
        axes_flat[i].axis('off')

    # 添加颜色一致性说明
    if gt_color_mapping:
        fig.suptitle('颜色一致性对比 - 相同颜色表示同一目标，红色表示误检',
                     fontsize=16, fontweight='bold', y=0.02)

    plt.tight_layout()
    plt.savefig(save_path, dpi=dpi, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print(f"已保存一致颜色多模型对比图像: {save_path}")


def _load_json_prediction_data(json_file_path: str, image_shape: Tuple[int, int]) -> Optional[Dict]:
    """
    加载JSON预测文件并转换为内部格式

    Args:
        json_file_path: JSON文件路径
        image_shape: 图像形状 (height, width)

    Returns:
        预测结果字典，包含 'segmentation' 和 'segments_info'
    """
    import json
    from pycocotools import mask as coco_mask

    try:
        with open(json_file_path, 'r') as f:
            prediction_data = json.load(f)

        # 检查必需的键
        required_keys = ['masks', 'labels', 'scores']
        if not all(k in prediction_data for k in required_keys):
            print(f"警告: JSON文件 {json_file_path} 缺少必需的键")
            return None

        rle_masks_data = prediction_data['masks']
        labels = prediction_data['labels']
        scores = prediction_data['scores']

        if not rle_masks_data:
            return None

        # 从RLE掩码获取尺寸
        mask_height, mask_width = rle_masks_data[0]['size']

        # 创建实例分割图
        instance_segmentation_map = np.zeros((mask_height, mask_width), dtype=np.int32)
        segments_info = []

        for i, rle_obj in enumerate(rle_masks_data):
            if i >= len(labels) or i >= len(scores):
                continue

            current_instance_id = i + 1
            label_id = labels[i]
            score = scores[i]

            try:
                # 构造完整的RLE字典
                rle_dict = {
                    'size': rle_obj['size'],
                    'counts': rle_obj['counts']
                }

                # 解码RLE掩码
                binary_mask_decoded = coco_mask.decode(rle_dict)
                if binary_mask_decoded.ndim == 3:
                    binary_mask = np.sum(binary_mask_decoded, axis=2) > 0
                else:
                    binary_mask = binary_mask_decoded > 0

                # 检查掩码和图像尺寸是否匹配
                if binary_mask.shape != image_shape:
                    # 如果尺寸不匹配，调整掩码大小
                    from PIL import Image as PILImage
                    mask_pil = PILImage.fromarray(binary_mask.astype(np.uint8) * 255)
                    mask_resized = mask_pil.resize((image_shape[1], image_shape[0]), PILImage.NEAREST)
                    binary_mask = np.array(mask_resized) > 0

                # 如果掩码不为空，添加到分割图中
                if np.sum(binary_mask) > 0:
                    instance_segmentation_map[binary_mask] = current_instance_id
                    segments_info.append({
                        'id': current_instance_id,
                        'label_id': label_id,
                        'score': score
                    })

            except Exception as e:
                print(f"错误: 解码RLE掩码失败: {e}")
                continue

        if not segments_info:
            return None

        return {
            'segmentation': instance_segmentation_map,
            'segments_info': segments_info
        }

    except Exception as e:
        print(f"错误: 加载JSON文件 {json_file_path} 失败: {e}")
        return None


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
    # image_path = "/Users/theobald/Documents/code_lib/python_lib/huggingface_data_process/datasets/local/360.jpg"
    # model_path = "/Users/theobald/Documents/code_lib/python_lib/huggingface_data_process/checkpoint/local/checkpoint-868"
    # predictor(image_path, model_path)

    image_dir = "/Users/theobald/Documents/code_lib/python_lib/shrimpDetection/mask2former/checkpoints/local/test22322dd2h"
    gt_dir = "/Users/theobald/Documents/code_lib/python_lib/shrimpDetection/mask2former/checkpoints/remote/benchmark/NYU/48label/plot/GT"
    m1_dir = "/Users/theobald/Documents/code_lib/python_lib/shrimpDetection/mask2former/checkpoints/remote/benchmark/NYU/48label/plot/vis_0.0.0"

    image_path_list = [os.path.join(image_dir, i) for i in os.listdir(image_dir) if i != '.DS_Store']
    gt_path_list = [os.path.join(gt_dir, i) for i in os.listdir(gt_dir) if i != '.DS_Store']
    m1_path_list = [os.path.join(m1_dir, i) for i in os.listdir(m1_dir) if i != '.DS_Store']

    # 调用可视化函数
    visualize_multi_model_json_results(
        image_paths=image_path_list,
        gt_json_paths=gt_path_list,
        model_json_paths=[m1_path_list, m1_path_list, m1_path_list],
        save_dir="/Users/theobald/Documents/code_lib/python_lib/shrimpDetection/mask2former/checkpoints/local/test22322dd2hdd",
        model_names=['m1', 'm1_1', 'm1_2'],
        alpha=0.6,
        show_titles=True,
        show_info_text=False,
        max_models_per_row=3
    )



if __name__ == '__main__':
    main()