"""
Date: 2024/11/29
Author: Renjie Zhou
Email: nikibandit200814@gmail.com
"""
import json

import torch
from PIL import Image
import shutil
import random
import string
import PIL.ImageOps
from tqdm import tqdm
from datasets import Dataset, DatasetDict, Image as datasets_Image
import json
import os
import cv2
import numpy as np
from PIL import Image
from typing import Dict, Tuple, List
import matplotlib.pyplot as plt
from collections import defaultdict


def get_unique_colors(image_path):
    """
    从给定图像路径读取图像，并返回图像中所有的独特颜色。

    :param image_path: 图像文件路径
    :return: 包含图像中所有独特颜色的列表
    """
    # 读取图像
    image = cv2.imread(image_path)

    # 确保图像是3通道（BGR）
    if len(image.shape) != 3 or image.shape[2] != 3:
        raise ValueError("Input image must be a 3-channel BGR image.")

    # 将图像转换为一维数组
    reshaped_image = image.reshape(-1, 3)

    # 使用NumPy的unique函数找出所有独特的颜色
    unique_colors = np.unique(reshaped_image, axis=0)

    return [tuple(color) for color in unique_colors]


def convert_mask2grayscale(mask_path):
    """
    将给定的语义分割mask图像转换成指定灰度值的灰度图像。

    :param mask_path: 语义分割mask图像路径
    :param output_path: 输出文件路径
    """
    # 读取图像
    mask_image = cv2.imread(mask_path)

    # 确保图像是3通道（BGR）
    if len(mask_image.shape) != 3 or mask_image.shape[2] != 3:
        raise ValueError("Input image must be a 3-channel BGR image.")

    # 获取图像中的所有独特颜色
    unique_colors = get_unique_colors(mask_path)
    print(f"该图像的颜色有：{unique_colors}")

    # 创建颜色到灰度值的映射
    color_to_gray_map = {color: i for i, color in enumerate(unique_colors, start=0)}

    # 黑色背景保持不变，用0表示
    color_to_gray_map[(0, 0, 0)] = 0

    # 创建一个查找表 (LUT) 来快速映射颜色
    lut = np.zeros((256, 256, 256), dtype=np.uint8)
    # for color, gray_value in color_to_gray_map.items():
    #     lut[color] = gray_value

    colors = np.array(list(color_to_gray_map.keys()))
    gray_values = np.array(list(color_to_gray_map.values()), dtype=np.uint8)
    lut[colors[:, 0], colors[:, 1], colors[:, 2]] = gray_values

    # 将图像转换为灰度图像
    grayscale_image = lut[mask_image[:, :, 0], mask_image[:, :, 1], mask_image[:, :, 2]]

    # colorful_grayscale_image = cv2.cvtColor(grayscale_image, cv2.COLOR_GRAY2BGR)
    # # 将图像转换为一维数组
    # reshaped_image = colorful_grayscale_image.reshape(-1, 3)
    #
    # # 使用NumPy的unique函数找出所有独特的颜色
    # unique_colors = np.unique(reshaped_image, axis=0)
    #
    # print(f"该图像的颜色有：{[tuple(color) for color in unique_colors]}")

    # # 保存结果
    # cv2.imwrite(output_path, grayscale_image)

    return grayscale_image


def combine_sematic_instance_mask(sematic_mask_path, instance_mask_path):
    """
    拼接sematic mask和instance mask
    :param sematic_mask_path: sematic_mask_path
    :param instance_mask_path: instance_mask_path
    :return: mask shape = （h, w, 3）
    """
    sematic_mask = convert_mask2grayscale(sematic_mask_path)
    instance_mask = convert_mask2grayscale(instance_mask_path)
    _ = np.zeros((sematic_mask.shape[0], sematic_mask.shape[1]))
    assert sematic_mask.shape == instance_mask.shape == _.shape, "the shape of sematic mask and instance mask and _ must be equal"
    # =============================================重要说明===========================================================
    # mask2former要求的annotation格式为3通道的图片: (h, w, 3)
    # 其中，第一层为sematic mask；第二层为instance mask；第三层会在annotation加载时舍丢弃
    # annotation在加载时使用PIL.Image读取，然后使用np.array(image)进行转化。因此，图像数据是按照RGB格式进行解析
    # 该函数(combine_sematic_instance_mask)使用的是opencv对图像数据进行解析，因此默认使用BGR格式
    # 为了对齐模型所采用的RGB加载格式，且优化图像格式之间的转化，因此，此处mask的拼接顺序有所不同（_, instance_mask, sematic_mask）
    #================================================================================================================
    mask = np.dstack((_, instance_mask, sematic_mask))
    assert mask.shape == (sematic_mask.shape[0], sematic_mask.shape[1], 3)

    return mask


def get_image_name_list(path: str) -> list:
    """
    Get image name
    :param path: path
    :return: 该文件夹下的图片名
    """
    assert os.path.isdir(path), f"{path} 不是一个文件夹"
    image_name_list = [x for x in os.listdir(path) if not os.path.isdir(os.path.join(path, x)) and (x.lower().endswith(".jpg") or x.lower().endswith(".png"))]

    return sorted(image_name_list)


def image_sort(path: str):
    """
    图片重定向（强烈建议在图片打标签之前完成重定向，否则后期处理很麻烦）
    :param path: path
    """
    assert os.path.isdir(path), f"{path} 不是一个文件夹"

    image_name_list = get_image_name_list(path)
    image_cache_path = os.path.join(path, "image_cache")
    if os.path.exists(image_cache_path) and os.path.isdir(image_cache_path):
        shutil.rmtree(image_cache_path)
        print("旧缓存已清除")

    os.mkdir(image_cache_path)
    print(f"{image_cache_path} 已创建")

    for i in range(len(image_name_list)):
        image_path = os.path.join(path, image_name_list[i])
        os.rename(image_path, os.path.join(image_cache_path, str(i) + ".jpg"))

    image_name_list = get_image_name_list(image_cache_path)
    for image_name in image_name_list:
        image_path = os.path.join(image_cache_path, image_name)
        os.rename(image_path, os.path.join(path, image_name))

    os.rmdir(image_cache_path)


def generate_random_color():
    """生成一个随机的 RGB 颜色"""
    return (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))


def label_check(image_path_list, mask_path_list):
    """
    标签检查
    :param image_path_list: image_path_list
    :param mask_path_list: mask_path_list
    """
    for image_path, mask_path in zip(image_path_list, mask_path_list):
        # 读取图像和掩码
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        mask = cv2.imread(mask_path, cv2.IMREAD_COLOR)

        # 提取语义掩码和实例掩码
        semantic_mask = mask[..., 2]  # 语义掩码在第三通道
        instance_mask = mask[..., 1]  # 实例掩码在第二通道

        # 处理语义掩码（不同语义区域用不同颜色）
        unique_semantics = np.unique(semantic_mask)  # 获取所有唯一的语义区域 ID
        unique_semantics = unique_semantics[unique_semantics != 0]  # 排除背景（0）

        semantic_colored = np.zeros_like(image, dtype=np.uint8)
        for semantic_id in unique_semantics:
            color = generate_random_color()  # 为每个语义区域生成一个随机颜色
            semantic_colored[semantic_mask == semantic_id] = color  # 用该颜色填充语义区域

        alpha = 0.5  # 透明度
        semantic_colored = cv2.addWeighted(semantic_colored, alpha, np.zeros_like(semantic_colored), 1 - alpha, 0)
        semantic_image = cv2.addWeighted(image, 1, semantic_colored, 1, 0)

        # 处理实例掩码（不同实例用不同颜色）
        unique_instances = np.unique(instance_mask)  # 获取所有唯一的实例 ID
        unique_instances = unique_instances[unique_instances != 0]  # 排除背景（0）

        instance_colored = np.zeros_like(image, dtype=np.uint8)
        for instance_id in unique_instances:
            color = generate_random_color()  # 为每个实例生成一个随机颜色
            instance_colored[instance_mask == instance_id] = color  # 用该颜色填充实例区域

        instance_colored = cv2.addWeighted(instance_colored, alpha, np.zeros_like(instance_colored), 1 - alpha, 0)
        instance_image = cv2.addWeighted(image, 1, instance_colored, 1, 0)

        # 处理掩码显示
        semantic_mask_display = np.where(semantic_mask == 0, 255, semantic_mask)
        instance_mask_display = np.where(instance_mask == 0, 255, instance_mask)
        semantic_mask_display = np.dstack((semantic_mask_display, semantic_mask_display, semantic_mask_display))
        instance_mask_display = np.dstack((instance_mask_display, instance_mask_display, instance_mask_display))

        # 确保所有图像的形状一致
        assert image.shape == semantic_mask_display.shape == instance_mask_display.shape

        # 将结果拼接并显示
        row = cv2.hconcat([semantic_image, instance_image, semantic_mask_display, instance_mask_display])
        cv2.imshow("image & sematic & instance", row)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def is_image(file_path):
    """检查给定路径的文件是否为图片"""
    try:
        with Image.open(file_path) as img:
            return True
    except IOError:
        return False


def rename_with_random_string(length=10):
    """生成指定长度的随机字符串"""
    letters = string.ascii_lowercase
    return ''.join(random.choice(letters) for i in range(length))


def process_directory(directory):
    """遍历目录，删除非图片文件并重命名图片文件(混洗图片)"""
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)

        # 检查是否为文件
        if not os.path.isfile(file_path):
            continue

        # 检查是否为图片
        if not is_image(file_path):
            print(f"Deleting non-image file: {file_path}")
            os.remove(file_path)
        else:
            # 生成新的文件名
            new_filename = rename_with_random_string() + ".jpg"
            new_file_path = os.path.join(directory, new_filename)
            print(f"Renaming image file: {filename} -> {new_filename}")
            os.rename(file_path, new_file_path)


def exif_transpose(img):
    """
    根据exif数据，调整图片姿态
    :param img: img
    :return: img
    """
    if not img:
        return img

    exif_orientation_tag = 274

    # Check for EXIF data (only present on some files)
    if hasattr(img, "_getexif") and isinstance(img._getexif(), dict) and exif_orientation_tag in img._getexif():
        exif_data = img._getexif()
        orientation = exif_data[exif_orientation_tag]

        # Handle EXIF Orientation
        if orientation == 1:
            # Normal image - nothing to do!
            pass
        elif orientation == 2:
            # Mirrored left to right
            img = img.transpose(PIL.Image.FLIP_LEFT_RIGHT)
        elif orientation == 3:
            # Rotated 180 degrees
            img = img.rotate(180)
        elif orientation == 4:
            # Mirrored top to bottom
            img = img.rotate(180).transpose(PIL.Image.FLIP_LEFT_RIGHT)
        elif orientation == 5:
            # Mirrored along top-left diagonal
            img = img.rotate(-90, expand=True).transpose(PIL.Image.FLIP_LEFT_RIGHT)
        elif orientation == 6:
            # Rotated 90 degrees
            img = img.rotate(-90, expand=True)
        elif orientation == 7:
            # Mirrored along top-right diagonal
            img = img.rotate(90, expand=True).transpose(PIL.Image.FLIP_LEFT_RIGHT)
        elif orientation == 8:
            # Rotated 270 degrees
            img = img.rotate(90, expand=True)

    return img


def load_image_file(file, mode='RGB'):
    """
    加载不包括exif数据的图片
    :param file: img path
    :param mode: mode
    :return: img
    """
    # Load the image with PIL
    img = PIL.Image.open(file)

    if hasattr(PIL.ImageOps, 'exif_transpose'):
        # Very recent versions of PIL can do exit transpose internally
        img = PIL.ImageOps.exif_transpose(img)
    else:
        # Otherwise, do the exif transpose ourselves
        img = exif_transpose(img)

    img = img.convert(mode)

    return img


def extract_frames(video_path, output_folder, frame_interval):
    """
    视频抽帧并保存
    :param video_path: 视频路径
    :param output_folder: 输出文件夹
    :param frame_interval: 抽帧间隔
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    save_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % frame_interval == 0:
            filename = os.path.join(output_folder, rename_with_random_string() + ".jpg")
            cv2.imwrite(filename, frame)
            print(f"Saved {filename}")
            save_count += 1
        frame_count += 1

    cap.release()


def get_label2id(json_path):
    """
    加载label2id文件
    :json_path: json_path
    :return: label2id
    """
    assert os.path.isfile(json_path) and os.path.splitext(json_path)[1] == '.json', f"{json_path}不是一个json配置文件"
    with open(json_path, 'r') as f:
        label2id = json.load(f)
    if not isinstance(label2id, dict):
        raise ValueError(f"JSON 文件 {json_path} 的内容不是有效的字典格式")

    return label2id


def split2train_and_valid(image_path_list, mask_path_list, depth_path_list=None, depth_expand_list_dict = None, valid_rate=0.3):
    """
    按照比例将数据集划分为训练集和验证集
    :param image_path_list: image_path_list
    :param mask_path_list: mask_path_list
    :valid_rate: 0.3
    :return: train_image_path_list, train_mask_path_list, valid_image_path_list, valid_mask_path_list
    """
    size = len(image_path_list)
    train_size = int(size * (1 - valid_rate))
    train_image_path_list = image_path_list[:train_size]
    train_mask_path_list = mask_path_list[:train_size]
    valid_image_path_list = image_path_list[train_size:]
    valid_mask_path_list = mask_path_list[train_size:]

    if depth_path_list is not None:
        train_depth_path_list = depth_path_list[:train_size]
        valid_depth_path_list = depth_path_list[train_size:]
        return train_image_path_list, train_mask_path_list, train_depth_path_list, valid_image_path_list, valid_mask_path_list, valid_depth_path_list
    elif depth_expand_list_dict is not None:
        train_depth_expand_list_dict = {}
        valid_depth_expand_list_dict = {}
        for i in depth_expand_list_dict:
            train_depth_expand_list_dict[i] = depth_expand_list_dict[i][:train_size]
            valid_depth_expand_list_dict[i] = depth_expand_list_dict[i][train_size:]
        return train_image_path_list, train_mask_path_list, train_depth_expand_list_dict, valid_image_path_list, valid_mask_path_list, valid_depth_expand_list_dict
    else:
        return train_image_path_list, train_mask_path_list, None, valid_image_path_list, valid_mask_path_list, None


def generate_meta_file(train_image_path_list, train_mask_path_list, valid_image_path_list, valid_mask_path_list, output_dir,
                       train_depth_path_list=None, train_depth_expand_list_dict=None,  valid_depth_path_list=None, valid_depth_expand_list_dict=None,
                       semantic_class_to_id=None):
    """
    生成dataset的元数据文件（json格式，train和valid分开存放）
    :param train_image_path_list: train_image_path_list
    :param train_mask_path_list: train_mask_path_list
    :param valid_image_path_list: valid_image_path_list
    :param valid_mask_path_list: valid_mask_path_list
    :param output_dir: the output dir of meta file
    :semantic_class_to_id: {"background": 0, "shrimp": 1}
    """
    if semantic_class_to_id is None:
        semantic_class_to_id = {"background": 0, "organ": 1, "shrimp": 2}

    def single_meta_data_unit(image_path_list, mask_path_list):
        data = []
        for i in tqdm(range(len(image_path_list))):
            data.append({
                "image": image_path_list[i],
                "annotation": mask_path_list[i],
                "semantic_class_to_id": semantic_class_to_id
            })

        return data

    def multi_meta_data_unit(image_path_list, mask_path_list, depth_path_list):
        data = []
        for i in tqdm(range(len(image_path_list))):
            data.append({
                "image": [image_path_list[i], depth_path_list[i]],
                "annotation": mask_path_list[i],
                "semantic_class_to_id": semantic_class_to_id
            })

        return data

    def ultra_meta_data_unit(image_path_list, mask_path_list, depth_expand_list_dict):
        data = []
        for i in tqdm(range(len(image_path_list))):
            data.append({
                "image":[
                    image_path_list[i],
                    depth_expand_list_dict['decimation_depth'][i],
                    depth_expand_list_dict['depth_colormap_by_rs'][i],
                    depth_expand_list_dict['spatial_depth'][i],
                    depth_expand_list_dict['hole_filling_depth'][i],
                    depth_expand_list_dict['ahe_depth'][i],
                    depth_expand_list_dict['laplace_depth'][i],
                    depth_expand_list_dict['gaussian_depth'][i],
                    depth_expand_list_dict['eq_depth'][i],
                    depth_expand_list_dict['lt_depth'][i]
                ],
                "annotation": mask_path_list[i],
                "semantic_class_to_id": semantic_class_to_id
            })

        return data

    if train_depth_expand_list_dict is not None and valid_depth_expand_list_dict is not None: # 30 channel
        train_data = ultra_meta_data_unit(train_image_path_list, train_mask_path_list, train_depth_expand_list_dict)
        valid_data = ultra_meta_data_unit(valid_image_path_list, valid_mask_path_list, valid_depth_expand_list_dict)

        # Write JSON files
        train_json_path = os.path.join(output_dir, "train_u.json")
        valid_json_path = os.path.join(output_dir, "valid_u.json")
    elif train_depth_path_list is not None and valid_depth_path_list is not None: # 6 channel
        train_data = multi_meta_data_unit(train_image_path_list, train_mask_path_list, train_depth_path_list)
        valid_data = multi_meta_data_unit(valid_image_path_list, valid_mask_path_list, valid_depth_path_list)

        # Write JSON files
        train_json_path = os.path.join(output_dir, "train_m.json")
        valid_json_path = os.path.join(output_dir, "valid_m.json")
    else: # 3 channel
        train_data = single_meta_data_unit(train_image_path_list, train_mask_path_list)
        valid_data = single_meta_data_unit(valid_image_path_list, valid_mask_path_list)

        # Write JSON files
        train_json_path = os.path.join(output_dir, "train_s.json")
        valid_json_path = os.path.join(output_dir, "valid_s.json")

    with open(train_json_path, "w") as train_file:
        json.dump(train_data, train_file, indent=4)

    with open(valid_json_path, "w") as valid_file:
        json.dump(valid_data, valid_file, indent=4)

    print(f"JSON files generated:\n  Train: {train_json_path}\n  Validation: {valid_json_path}")


def load_coco_annotations(json_path: str) -> Tuple[Dict, Dict]:
    """加载并组织COCO格式的标注数据"""
    with open(json_path, 'r') as f:
        data = json.load(f)

    images = {img['id']: img for img in data['images']}
    annotations = {}
    for ann in data['annotations']:
        img_id = ann['image_id']
        annotations.setdefault(img_id, []).append(ann)

    return images, annotations


def create_mask_from_polygon(polygon: List[float], image_size: Tuple[int, int]) -> np.ndarray:
    """将多边形坐标转换为二值掩码"""
    mask = np.zeros(image_size[::-1], dtype=np.uint8)
    pts = np.array(polygon, np.int32).reshape((-1, 1, 2))
    cv2.fillPoly(mask, [pts], color=1)
    return mask


def generate_combined_masks(
        images: Dict,
        annotations: Dict,
        output_dir: str,
        image_dir: str,
        max_instances: int = 255
) -> None:
    """
    生成并保存三通道标签图像（第二、第三通道相同）

    参数说明：
    max_instances - 单图最大允许实例数（默认255）
    """
    os.makedirs(output_dir, exist_ok=True)

    for img_id, img_info in tqdm(images.items()):
        # 获取图像尺寸
        img_path = os.path.join(image_dir, img_info['file_name'])
        if os.path.exists(img_path):
            img = Image.open(img_path)
            width, height = img.size
        else:
            width, height = img_info['width'], img_info['height']

        # 初始化标签矩阵
        semantic_mask = np.zeros((height, width), dtype=np.uint8)
        instance_mask = np.zeros((height, width), dtype=np.uint8)

        # 处理实例标注
        instance_id = 1
        for ann in annotations.get(img_id, []):
            # 检查实例ID是否超出限制
            if instance_id > max_instances:
                raise ValueError(f"实例数量超过最大限制 {max_instances}（图像：{img_info['file_name']}）")

            # 处理语义标签（假设单类别）
            for polygon in ann['segmentation']:
                poly_mask = create_mask_from_polygon(polygon, (width, height))
                semantic_mask = np.where(poly_mask, ann['category_id'], semantic_mask)

            # 处理实例标签
            instance_poly = ann['segmentation'][0]
            instance_region = create_mask_from_polygon(instance_poly, (width, height))
            instance_mask = np.where(instance_region, instance_id, instance_mask)

            instance_id += 1

        # 构建三通道图像（第二、第三通道相同）
        combined = np.stack([
            semantic_mask,
            instance_mask,
            instance_mask  # 第三通道复制第二通道
        ], axis=-1)

        # 保存结果
        image_name = img_info['file_name'].split("_")
        output_path = os.path.join(
            output_dir,
            image_name[0] + "_" + image_name[1] + ".png"
        )
        Image.fromarray(combined).save(output_path)


def dataset_constructor(image_dir, mask_dir, output_dir, mask_check=True, data_form='cvat', semantic_dir=None, instance_dir=None, coco_json_path=None, depth_dir=None):
    assert os.path.isdir(image_dir), f"{image_dir} 不存在"
    assert os.path.isdir(mask_dir), f"{mask_dir} 不存在"
    assert os.path.isdir(output_dir), f"{output_dir} 不存在"
    if data_form == 'cvat': # cvat segmentation
        assert os.path.isdir(semantic_dir), f"{semantic_dir} 不存在"
        assert os.path.isdir(instance_dir), f"{instance_dir} 不存在"
        assert len(os.listdir(mask_dir)) == 0, f"{mask_dir} 不为空，妨碍mask存储"

        semantic_name_list = get_image_name_list(semantic_dir)
        instance_name_list = get_image_name_list(instance_dir)
        assert semantic_name_list == instance_name_list, "sematic mask 和 instance mask不匹配"
        for mask_name in tqdm(semantic_name_list):
            sematic_path = os.path.join(semantic_dir, mask_name)
            instance_path = os.path.join(instance_dir, mask_name)
            mask = combine_sematic_instance_mask(sematic_path, instance_path)
            save_path = os.path.join(mask_dir, mask_name)
            cv2.imwrite(save_path, mask)

    else: # roboflow segmentation
        assert os.path.isfile(coco_json_path), f"{coco_json_path} 不存在"
        img_data, ann_data = load_coco_annotations(coco_json_path)
        generate_combined_masks(img_data, ann_data, mask_dir, image_dir)
        for image_name in get_image_name_list(image_dir): # 对原始图片文件重命名
            new_image_name = image_name.split("_")
            new_image_name = new_image_name[0] + "_" + new_image_name[1] + ".png"
            os.renames(os.path.join(image_dir, image_name), os.path.join(image_dir, new_image_name))

    mask_path_list = [os.path.join(mask_dir, mask_name) for mask_name in get_image_name_list(mask_dir)]
    image_path_list = [os.path.join(image_dir, image_name) for image_name in get_image_name_list(image_dir)]
    depth_path_list = None
    assert all(os.path.splitext(os.path.basename(image_path))[0] == os.path.splitext(os.path.basename(mask_path))[0]
               for image_path, mask_path in zip(image_path_list, mask_path_list)), "image 和 mask 不匹配"
    if depth_dir is not None:
        depth_path_list = [os.path.join(depth_dir, image_name) for image_name in get_image_name_list(image_dir)]
        assert all(os.path.splitext(os.path.basename(image_path))[0] == os.path.splitext(os.path.basename(depth_path))[0]
                   for image_path, depth_path in zip(image_path_list, depth_path_list)), "image和depth 不匹配"

    if mask_check:
        label_check(image_path_list, mask_path_list)

    train_image_path_list, train_mask_path_list, train_depth_path_list, valid_image_path_list, valid_mask_path_list, valid_depth_path_list = split2train_and_valid(
        image_path_list, mask_path_list, depth_path_list)

    generate_meta_file(train_image_path_list=train_image_path_list,
                       train_mask_path_list=train_mask_path_list,
                       valid_image_path_list=valid_image_path_list,
                       valid_mask_path_list=valid_mask_path_list,
                       output_dir=output_dir,
                       train_depth_path_list=train_depth_path_list,
                       valid_depth_path_list=valid_depth_path_list)


def calculate_depth_histogram(depth_map, bins=512, value_range=None):
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
        value_range = (np.nanmin(depth_map), np.nanmax(depth_map)) # 忽略 NaN 值

    hist, bin_edges = np.histogram(depth_map.flatten(), bins=bins, range=value_range, density=False)
    return hist, bin_edges


def select_depth_distribution_modes(hist, bin_edges, num_modes=3, prominence_threshold=0.01):
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
    peaks_indices, _ = find_peaks(hist, prominence=prominence_threshold * np.max(hist)) # 使用显著性阈值

    if not peaks_indices.size: # 如果没有找到峰值
        return []

    # 获取峰值的高度和位置 (近似中心值)
    peak_heights = hist[peaks_indices]
    peak_centers = bin_edges[:-1][peaks_indices] + np.diff(bin_edges)[peaks_indices] / 2.0 # 近似中心值

    # 将峰值按照高度降序排序
    peak_data = sorted(zip(peak_heights, peak_centers), reverse=True)

    selected_modes = [center for _, center in peak_data[:num_modes]] # 选择前 num_modes 个峰值中心

    return selected_modes


def define_depth_interval_windows(depth_modes, window_size_ratio=0.1):
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
        lower_bound = max(0, mode_center - window_half_width) # 保证下界不小于0，假设深度值非负
        upper_bound = mode_center + window_half_width
        interval_windows.append((lower_bound, upper_bound))
    return interval_windows


def generate_depth_region_masks(depth_map, interval_windows):
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
    combined_mask = np.zeros_like(depth_map, dtype=bool) # 用于记录已覆盖的区域

    for lower_bound, upper_bound in interval_windows:
        mask = (depth_map >= lower_bound) & (depth_map <= upper_bound)
        region_masks.append(mask)
        combined_mask |= mask # 累积覆盖区域

    # 生成剩余区域掩码 (深度值不在任何定义的窗口内的区域)
    remaining_mask = ~combined_mask
    region_masks.append(remaining_mask)

    return region_masks


def histogram_viewer(hist, bin_edges):
    # 可视化直方图
    plt.figure(figsize=(8, 4))
    plt.bar(bin_edges[:-1], hist, width=np.diff(bin_edges), align="edge")
    plt.title("Depth Histogram")
    plt.xlabel("Depth Value")
    plt.ylabel("Frequency")
    plt.show()


def depth_region_viewer(interval_windows, region_masks):
    # 可视化深度区域掩码 (前3个区域 + 剩余区域)
    plt.figure(figsize=(12, 3))
    titles = [f"Region Mask {i+1}" for i in range(len(interval_windows))] + ["Remaining Region Mask"]
    for i, mask in enumerate(region_masks):
        plt.subplot(1, len(region_masks), i+1)
        plt.imshow(mask, cmap='gray')
        plt.title(titles[i])
        plt.axis('off')
    plt.suptitle("Depth Region Masks Visualization")
    plt.tight_layout()
    plt.show()


def cosine_similarity(image_A, image_B):
    """
    计算图像像素级别的余弦相似度图，**向量化版本，效率更高。**
    兼容 NumPy array 和 PyTorch Tensor 输入。
    特殊处理两个像素向量都为零向量的情况，返回相似度 1.0。
    使用 float32 数据类型进行计算，避免 uint8 溢出问题。

    参数:
    image_A (numpy.ndarray or torch.Tensor): 图像 A, 形状 (H, W, C) 或 (H, W)  (灰度图), dtype 可以是 uint8 或其他。
    image_B (numpy.ndarray or torch.Tensor): 图像 B, 形状 (H, W, C') 或 (H, W) (灰度图)。
                                图像 A 和 图像 B 需要 resize 到相同的 Height 和 Width。

    返回值:
    numpy.ndarray: 像素级别的余弦相似度图, 形状 (H, W), dtype=float32。
                   每个像素值表示对应位置的余弦相似度得分，范围在 [-1, 1] 之间。
                   **返回 NumPy array 格式的相似度图。**
    """

    # 1. 检查输入类型并转换为 NumPy array (保持与之前版本一致)
    if isinstance(image_A, torch.Tensor):
        image_A_np = image_A.cpu().numpy()
    else:
        image_A_np = image_A

    if isinstance(image_B, torch.Tensor):
        image_B_np = image_B.cpu().numpy()
    else:
        image_B_np = image_B

    # 2. 转换为 float32 数据类型 (向量化操作的关键)
    image_A_float = image_A_np.astype(np.double)
    image_B_float = image_B_np.astype(np.double)

    # 3. 向量化计算点积 (pixel-wise dot product)
    # 如果是彩色图像 (H, W, C)，则沿着通道维度 (axis=-1) 求和，得到 (H, W) 的点积图
    # 如果是灰度图像 (H, W)，则直接 element-wise 乘法，得到 (H, W) 的点积图
    dot_product_map = np.sum(image_A_float * image_B_float, axis=-1, keepdims=False)  # keepdims=False 去除维度为 1 的轴

    # 4. 向量化计算 L2 范数 (pixel-wise norm)
    # 同样，沿着通道维度 (axis=-1) 计算范数，得到 (H, W) 的范数图
    norm_A_map = np.linalg.norm(image_A_float, axis=-1, keepdims=False)
    norm_B_map = np.linalg.norm(image_B_float, axis=-1, keepdims=False)

    # 5. 向量化计算余弦相似度 (避免除以零)
    # 初始化相似度图为 0 (处理分母为零的情况)
    similarity_map_np = np.zeros_like(dot_product_map, dtype=np.double)

    # 找到分母不为零的位置 (即 norm_A * norm_B != 0 的位置)
    valid_denominator_mask = (norm_A_map * norm_B_map) != 0

    # 在分母不为零的位置，计算余弦相似度
    similarity_map_np[valid_denominator_mask] = (
        dot_product_map[valid_denominator_mask] / (norm_A_map[valid_denominator_mask] * norm_B_map[valid_denominator_mask])
    )

    # **[可选] 特殊处理：两个像素向量都为零向量的情况，设置为 1.0 (如果需要)**
    zero_vector_mask = (norm_A_map == 0) & (norm_B_map == 0)
    similarity_map_np[zero_vector_mask] = 1.0  #  向量化设置为 1.0

    return similarity_map_np  # 返回 NumPy array 格式的相似度图


def cosine_similarity_fuse_v3(original_images, check=None):
    """
    Implements the Cosine Similarity Fuse (CSF) algorithm to fuse multiple images.
    Includes a check parameter to collect intermediate data for visualization.

    Args:
        original_images (list of numpy.ndarray): A list of N original images (O_N).
                                                Images should have the same height and width.
        check (bool or function, optional): If True or a function is provided, intermediate
                                            data will be collected and passed to the function.
                                            Defaults to None.

    Returns:
        numpy.ndarray: The fused image.
    """
    num_images = len(original_images)
    if num_images <= 1:
        return original_images[0] if original_images else None  # Handle cases with 0 or 1 image

    visualization_data = { # Initialize the dictionary to store visualization data
        'similarity_score_matrices_rounds': [],
        'contributing_pixel_counts_rounds': [],
        'round_result_images': [],
        'final_scores_and_weights': {}
    }

    round_result_images = []
    original_image_scores = {i: 0 for i in range(num_images)} # Initialize scores for each original image

    for k_standard_index in range(num_images):
        standard_image = original_images[k_standard_index]
        similarity_score_matrices = []
        compared_image_indices = [i for i in range(num_images) if i != k_standard_index]

        # 1. Generate Similarity Score Matrices (for round k)
        current_round_similarity_matrices = [] # Store similarity matrices for current round
        for compared_index in compared_image_indices:
            compared_image = original_images[compared_index]
            similarity_matrix = cosine_similarity(standard_image, compared_image)
            similarity_score_matrices.append(similarity_matrix)
            current_round_similarity_matrices.append(similarity_matrix) # Append to round list
        visualization_data['similarity_score_matrices_rounds'].append(current_round_similarity_matrices) # Store for visualization data

        # 2. Generate Round Image (B_k) and Original Image Scores
        round_result_image_Bk = np.zeros_like(standard_image, dtype=np.float32) # Initialize B_k
        contributing_image_counts = {i: 0 for i in compared_image_indices} # Count pixel contributions

        height, width = standard_image.shape[:2]
        for h in range(height):
            for w in range(width):
                max_similarity = -float('inf')
                best_source_image_index = -1

                for i, sim_matrix in enumerate(similarity_score_matrices):
                    current_similarity = sim_matrix[h, w]
                    if current_similarity > max_similarity:
                        max_similarity = current_similarity
                        best_source_image_index = compared_image_indices[i]

                if best_source_image_index != -1: # Should always be true in this logic but good to check
                    source_image = original_images[best_source_image_index]
                    round_result_image_Bk[h, w] = source_image[h, w]
                    contributing_image_counts[best_source_image_index] += 1

        round_result_images.append(round_result_image_Bk)
        visualization_data['round_result_images'].append(round_result_image_Bk) # Store round result image

        visualization_data['contributing_pixel_counts_rounds'].append(contributing_image_counts) # Store contributing counts for round

        # Find image C with most contribution and update score
        max_contribution_count = -1
        image_C_index = -1
        for index, count in contributing_image_counts.items():
            if count > max_contribution_count:
                max_contribution_count = count
                image_C_index = index

        if image_C_index != -1:
            original_image_scores[image_C_index] += max_contribution_count

    # 4. Generate Fused Image
    total_score = sum(original_image_scores.values())
    if total_score == 0:
        weights_normalized = [1.0 / num_images] * num_images # Default uniform weights if all scores are zero
    else:
        weights = [original_image_scores[i] for i in range(num_images)]
        weights_normalized = [w / total_score for w in weights] # Normalize scores to weights

    visualization_data['final_scores_and_weights']['original_image_scores'] = original_image_scores # Store final scores
    visualization_data['final_scores_and_weights']['weights_normalized'] = weights_normalized # Store normalized weights


    fused_image = np.zeros_like(original_images[0], dtype=np.float32)
    for i in range(num_images):
        fused_image += weights_normalized[i] * round_result_images[i]

    if check: # Check if check is True or a function is provided
        if callable(check):
            check(visualization_data) # Call the injected function with visualization data
        elif check == True:
            pass # If check=True and no function, you can add default data printing here if needed

    return fused_image.astype(original_images[0].dtype) # Return fused image with original dtype


def csf_viewer_v2(visualization_data):
    """
    Visualizes the intermediate data from cosine_similarity_fuse_v1.
    (Corrected version to avoid IndexError)
    Adds round result images to the rightmost column.
    """
    similarity_score_matrices_rounds = visualization_data['similarity_score_matrices_rounds']
    contributing_pixel_counts_rounds = visualization_data['contributing_pixel_counts_rounds']
    round_result_images = visualization_data['round_result_images'] # Retrieve round result images
    num_rounds = len(similarity_score_matrices_rounds)
    num_images = num_rounds # N images, N rounds

    fig, axes = plt.subplots(num_rounds, num_images + 1, figsize=((num_images + 1) * 5, num_rounds * 4)) # +1 column for round result images
    fig.suptitle('Cosine Similarity Fuse - Intermediate Visualization', fontsize=16)

    for round_index in range(num_rounds):
        current_round_matrices = similarity_score_matrices_rounds[round_index]
        current_round_contributions = contributing_pixel_counts_rounds[round_index]
        current_round_result_image = round_result_images[round_index] # Get round result image
        standard_image_index = round_index

        compared_image_indices_in_round = [i for i in range(num_images) if i != standard_image_index]

        col_index_counter = 0 # Counter for column index in visualization grid

        for image_index in range(num_images): # Iterate through all image indices for columns
            ax = axes[round_index, image_index]

            if image_index == standard_image_index:
                # Position for standard image - leave blank
                ax.set_xticks([])
                ax.set_yticks([])
                ax.set_title(f'Image {standard_image_index + 1} (Standard)')
                ax.set_xlabel('Blank (Standard Image Position)')
                col_index_counter += 1 # Increment column counter, even for blank space
                continue # Skip to next image_index

            # Check if current image_index is in compared_image_indices_in_round
            if image_index in compared_image_indices_in_round:
                # Find the index of image_index within compared_image_indices_in_round
                compared_image_current_index = compared_image_indices_in_round.index(image_index)

                sim_matrix = current_round_matrices[compared_image_current_index] # Now should be safe
                contribution_count = current_round_contributions.get(image_index, 0)

                im = ax.imshow(sim_matrix, cmap='jet', vmin=-1, vmax=1)
                ax.set_title(f'vs Image {standard_image_index + 1}')
                ax.set_xlabel(f'Image {image_index + 1}\nContribution: {contribution_count} pixels')

                # No colorbar in last original image column, colorbar will be in the round result image column

            else:
                # This should ideally not be reached, but for safety, handle it:
                ax.set_xticks([])
                ax.set_yticks([])
                ax.set_title(f'Image {image_index + 1}')
                ax.set_xlabel('No Sim Matrix')
                print(f"Warning: No similarity matrix expected for Round {round_index + 1}, Image {image_index + 1} (Standard Image?)")

            col_index_counter += 1 # Increment column counter for compared images as well

        # --- Add Round Result Image to the last column of each row ---
        ax_round_result = axes[round_index, num_images] # Get the axes for the last column

        print(f"Round {round_index + 1} Result Image:") # 调试信息
        print(f"  Data type: {current_round_result_image.dtype}") # 打印数据类型
        print(f"  Min value: {current_round_result_image.min()}") # 打印最小值
        print(f"  Max value: {current_round_result_image.max()}") # 打印最大值
        # 关键修改： 归一化数据到 [0, 1] 范围 (针对 float32 和 float64)
        if current_round_result_image.dtype == np.float32 or current_round_result_image.dtype == np.float64:
            image_to_display = current_round_result_image / 255.0  # 归一化到 [0, 1]
        else:
            image_to_display = current_round_result_image  # 其他数据类型 (如 uint8) 保持不变

        ax_round_result.imshow(image_to_display) # Display round result image (assuming grayscale)
        ax_round_result.set_xticks([]) # Remove ticks
        ax_round_result.set_yticks([]) # Remove ticks
        ax_round_result.set_title(f'Round {round_index + 1}\nResult Image') # Set title for round result image

        # if round_index == 0: # Add colorbar to the first round result image column only for demonstration
        fig.colorbar(im, ax=ax_round_result) # Add colorbar in the round result image column, using the last `im` created

        axes[round_index, 0].set_ylabel(f'Round {round_index + 1}') # Row label


    # Column labels (for the first row)
    for image_index in range(num_images):
        axes[0, image_index].set_title(f'Image {image_index + 1}')
        if image_index == standard_image_index:
            axes[0, image_index].set_title(f'Image {image_index + 1} (Standard)')
    axes[0, num_images].set_title('Round Result\nImages') # Title for the new round result image column


    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()


def to_grayscale(image_data):
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


def compute_depth_gradient(depth_map: np.ndarray) -> np.ndarray:
    """
    Computes the gradient magnitude map from a single-channel depth map.

    Args:
        depth_map (np.ndarray): Input depth map as a NumPy array.
                                Expected to be single-channel (H, W).
                                Can be integer (e.g., uint16) or float.

    Returns:
        np.ndarray: The gradient magnitude map (H, W), float type.
                    Returns an empty array if input is invalid.
    """
    if depth_map is None or depth_map.ndim != 2:
        print("Error: Input depth_map must be a single-channel NumPy array.")
        return np.array([])

    # Convert depth map to float32 for gradient calculation
    # Sobel works on various types, but float is generally safer for derivatives
    depth_map_float = depth_map.astype(np.float32)

    # Calculate gradients in x and y directions using Sobel operator
    # Output depth CV_64F to avoid clipping negative gradients or large values
    grad_x = cv2.Sobel(depth_map_float, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(depth_map_float, cv2.CV_64F, 0, 1, ksize=3)

    # Calculate the gradient magnitude
    gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)

    # --- Visualization ---
    # Normalize the magnitude for display (scale to 0-255)
    # cv2.NORM_MINMAX scales the array to the specified range [0, 255]
    gradient_magnitude_display = cv2.normalize(gradient_magnitude, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)

    # # Display the gradient magnitude map
    # cv2.imshow("Depth Gradient Magnitude", gradient_magnitude_display)

    return gradient_magnitude # Return the float magnitude map


def compute_surface_normals(depth_map: np.ndarray, invalid_depth_value: float = 0.0) -> np.ndarray:
    """
    Computes the surface normal map from a single-channel depth map.
    Assumes a camera coordinate system where +Z is depth, +X is right, +Y is down.
    Normals are computed based on gradients (-Gx, -Gy, 1) and then normalized.

    Args:
        depth_map (np.ndarray): Input depth map as a NumPy array (H, W).
                                Can be integer (e.g., uint16) or float.
        invalid_depth_value (float): The value used to represent invalid depth pixels.
                                     These pixels will have zero normals in the output.
                                     Defaults to 0.0.

    Returns:
        np.ndarray: The surface normal map (H, W, 3), float type, with values in [-1, 1].
                    Returns an empty array if input is invalid.
    """
    if depth_map is None or depth_map.ndim != 2:
        print("Error: Input depth_map must be a single-channel NumPy array.")
        return np.array([])

    # Mask out invalid depth values
    valid_mask = (depth_map != invalid_depth_value)

    # Convert depth map to float32 for gradient calculation
    depth_map_float = depth_map.astype(np.float32)

    # Calculate gradients (∂Z/∂u, ∂Z/∂v) using Sobel operator
    # Output depth CV_64F
    grad_x = cv2.Sobel(depth_map_float, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(depth_map_float, cv2.CV_64F, 0, 1, ksize=3)

    # Calculate normal vectors proportional to (-Gx, -Gy, 1)
    # The Z component is 1 in this simplified model (assuming unit aspect ratio and focal length)
    z_component = np.ones_like(grad_x) # Shape (H, W), dtype CV_64F

    # Stack components to get normal vectors (H, W, 3)
    # The order here is [-Gx, -Gy, 1]
    normals = np.stack([-grad_x, -grad_y, z_component], axis=-1) # Shape (H, W, 3), dtype CV_64F

    # Calculate magnitude for normalization
    magnitude = np.linalg.norm(normals, axis=-1, keepdims=True) # Shape (H, W, 1)

    # Avoid division by zero where magnitude is zero (e.g., perfectly flat areas)
    # Add a small epsilon or handle explicitly
    magnitude[magnitude == 0] = 1e-6 # Prevent division by zero

    # Normalize the normal vectors to unit length
    unit_normals = normals / magnitude

    # Apply the valid mask: set normals to [0, 0, 0] for invalid pixels
    unit_normals[~valid_mask] = 0 # Set invalid normals to zero vector

    # --- Visualization ---
    # Map unit normals (-1 to 1) to RGB (0 to 255) for display
    # Formula: (N + 1) / 2 * 255
    # Ensure values are in [0, 1] range before scaling to [0, 255]
    # Clip values just in case numerical issues cause slight deviations
    unit_normals_display = ((unit_normals + 1) / 2.0 * 255.0).clip(0, 255).astype(np.uint8)

    # OpenCV expects BGR order for color images.
    # A common visualization mapping is:
    # X component -> Red channel
    # Y component -> Green channel
    # Z component -> Blue channel
    # Our unit_normals are in order [Nx, Ny, Nz] (where Nx=-Gx, Ny=-Gy, Nz=1)
    # For OpenCV BGR, we need [Nz, Ny, Nx]
    unit_normals_display_bgr = unit_normals_display[:, :, [2, 1, 0]] # Rearrange channels

    # Display the surface normal map
    # cv2.imshow("Surface Normals", unit_normals_display_bgr)

    return unit_normals_display_bgr # Return the float unit normal map


def calculate_gradient_features(depth_np: np.ndarray, invalid_depth_value: float = 0.0):
    """
    Calculates depth gradient magnitude and Gx/Gy from a depth map.
    Handles invalid depth values by masking or setting gradients to 0.

    Args:
        depth_np (np.ndarray): Input depth map as a float32 NumPy array (H, W).
        invalid_depth_value (float): The value representing invalid depth in depth_np.

    Returns:
        tuple: (gradient_magnitude_np, grad_x_np, grad_y_np, valid_gradient_mask_np)
               All are float32 NumPy arrays (H, W).
               valid_gradient_mask_np is 1.0 for valid gradients, 0.0 otherwise.
    """
    # Ensure depth is float32
    depth_np = depth_np.astype(np.float32)

    # Create a mask for valid depth values
    valid_depth_mask = (depth_np != invalid_depth_value) & (~np.isnan(depth_np))

    # Calculate gradients using Sobel operator
    # Use CV_32F for float output
    grad_x = cv2.Sobel(depth_np, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(depth_np, cv2.CV_32F, 0, 1, ksize=3)

    # Calculate magnitude
    magnitude = np.sqrt(grad_x**2 + grad_y**2)

    # Mask out gradients where original depth was invalid
    # Set to 0 where depth was invalid
    grad_x[~valid_depth_mask] = 0
    grad_y[~valid_depth_mask] = 0
    magnitude[~valid_depth_mask] = 0

    # Generate a mask for valid *gradients* (e.g., where magnitude > 0)
    # This mask indicates potential boundary regions
    valid_gradient_mask = (magnitude > 0).astype(np.float32)

    # Optional: Normalize magnitude (e.g., to [0, 1] based on non-zero values)
    # This normalization is different from standard image normalization
    valid_magnitudes = magnitude[valid_gradient_mask > 0]
    if valid_magnitudes.size > 0:
        min_val = np.min(valid_magnitudes)
        max_val = np.max(magnitude) # Use max of all values, including 0
        if max_val > min_val:
             normalized_magnitude = (magnitude - min_val) / (max_val - min_val)
        else: # Handle case where all valid gradients are the same or only 0
             normalized_magnitude = np.zeros_like(magnitude, dtype=np.float32)
    else: # Handle case with no valid gradients
        normalized_magnitude = np.zeros_like(magnitude, dtype=np.float32)

    # Add channel dimension
    # normalized_magnitude = np.expand_dims(normalized_magnitude, axis=-1) # (H, W, 1)
    # grad_x = np.expand_dims(grad_x, axis=-1) # (H, W, 1)
    # grad_y = np.expand_dims(grad_y, axis=-1) # (H, W, 1)
    # valid_gradient_mask = np.expand_dims(valid_gradient_mask, axis=-1) # (H, W, 1)


    return normalized_magnitude, grad_x, grad_y, valid_gradient_mask


def calculate_surface_normals(depth_np: np.ndarray, camera_intrinsics: dict = None, invalid_depth_value: float = 0.0):
    """
    Calculates surface normals from a depth map.
    Uses camera intrinsics if provided for accurate normals.
    If intrinsics are NOT provided, uses a gradient-based approximation.
    Handles invalid depth values.

    Args:
        depth_np (np.ndarray): Input depth map as a float32 NumPy array (H, W).
        camera_intrinsics (dict | None): Dictionary with keys 'fx', 'fy', 'cx', 'cy'.
                                         If None, a gradient-based approximation is used.
        invalid_depth_value (float): The value representing invalid depth pixels.

    Returns:
        tuple: (surface_normals_np, valid_normal_mask_np)
               surface_normals_np is float32 NumPy array (H, W, 3) with values in [-1, 1].
               valid_normal_mask_np is 1.0 for valid normals, 0.0 otherwise.
    """
    # Ensure depth is float32
    depth_np = depth_np.astype(np.float32)
    H, W = depth_np.shape

    # Create a mask for valid depth values
    valid_depth_mask = (depth_np != invalid_depth_value) & (~np.isnan(depth_np))

    if camera_intrinsics is not None:
        # --- Method 1: Standard 3D Point Cross Product (Requires Intrinsics) ---
        fx, fy, cx, cy = camera_intrinsics['fx'], camera_intrinsics['fy'], camera_intrinsics['cx'], camera_intrinsics['cy']

        # Convert depth to 3D points in camera coordinates
        v, u = np.indices((H, W))
        Z = depth_np.copy()
        Z[~valid_depth_mask] = np.nan # Use NaN for invalid points for safer calculations

        X = (u - cx) * Z / fx
        Y = (v - cy) * Z / fy

        points_3d = np.stack([X, Y, Z], axis=-1) # Shape (H, W, 3)

        # Calculate gradients of 3D points
        dP_du = np.gradient(points_3d, axis=1) # Gradient along width (u)
        dP_dv = np.gradient(points_3d, axis=0) # Gradient along height (v)

        # Calculate cross product (normal vector)
        normals = np.cross(dP_du.reshape(-1, 3), dP_dv.reshape(-1, 3)).reshape(H, W, 3)

        # Normalize normals
        norm = np.linalg.norm(normals, axis=-1, keepdims=True)
        # Avoid division by zero for invalid normals (where norm is 0 or NaN)
        norm[norm == 0] = 1e-6
        norm[np.isnan(norm)] = 1e-6

        unit_normals = normals / norm

        # Handle invalid points: set normals to (0,0,0) where original depth was invalid
        # Also set to (0,0,0) where the calculated normal is NaN (e.g., from NaN gradients)
        invalid_normal_mask = ~valid_depth_mask | np.isnan(unit_normals).any(axis=-1)
        unit_normals[invalid_normal_mask] = 0

        # Generate a mask for valid normals (where normal vector is not (0,0,0))
        valid_normal_mask = (np.linalg.norm(unit_normals, axis=-1) > 1e-5).astype(np.float32)

    else:
        # --- Method 2: Gradient-Based Approximation (No Intrinsics) ---
        print("Camera intrinsics not provided. Using gradient-based normal approximation.")

        # Calculate gradients using Sobel operator
        # Use CV_32F for float output
        grad_x = cv2.Sobel(depth_np, cv2.CV_32F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(depth_np, cv2.CV_32F, 0, 1, ksize=3)

        # Mask out gradients where original depth was invalid
        grad_x[~valid_depth_mask] = 0
        grad_y[~valid_depth_mask] = 0

        # Construct approximate normal vector (-Gx, -Gy, 1)
        # The Z component is fixed at 1
        z_component = np.ones_like(grad_x) # Shape (H, W), dtype float32

        # Stack components to get approximate normal vectors (H, W, 3)
        # Order: [-Gx, -Gy, 1]
        approx_normals = np.stack([-grad_x, -grad_y, z_component], axis=-1) # Shape (H, W, 3), dtype float32

        # Calculate magnitude for normalization
        magnitude = np.linalg.norm(approx_normals, axis=-1, keepdims=True) # Shape (H, W, 1)

        # Avoid division by zero where magnitude is zero (e.g., perfectly flat areas)
        # Add a small epsilon or handle explicitly
        magnitude[magnitude == 0] = 1e-6 # Prevent division by zero

        # Normalize the approximate normal vectors to unit length
        unit_normals = approx_normals / magnitude

        # Apply the valid depth mask: set normals to [0, 0, 0] for invalid pixels
        # Also handle potential NaNs from division if magnitude was NaN
        invalid_normal_mask = ~valid_depth_mask | np.isnan(unit_normals).any(axis=-1)
        unit_normals[invalid_normal_mask] = 0 # Set invalid normals to zero vector

        # Generate a mask for valid normals (where normal vector is not (0,0,0))
        # This mask indicates points where a normal could be successfully calculated
        valid_normal_mask = (np.linalg.norm(unit_normals, axis=-1) > 1e-5).astype(np.float32) # Use a small threshold


    # Add channel dimension to mask
    # valid_normal_mask = np.expand_dims(valid_normal_mask, axis=-1) # (H, W, 1)

    return unit_normals, valid_normal_mask

def extract_image_paths_from_json(json_file_path):
    """
    Scans a JSON file with the specified structure and extracts all image paths.

    The expected JSON structure is a list of objects, where each object
    contains an "image" key with the image file path as its value.

    Args:
        json_file_path (str): The path to the input JSON file.

    Returns:
        list: A list of strings, where each string is an image file path.
              Returns an empty list if the file is not found, invalid JSON,
              or has an unexpected structure.
    """
    if not os.path.exists(json_file_path):
        print(f"Error: JSON file not found at {json_file_path}")
        return []

    try:
        with open(json_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON from {json_file_path}: {e}")
        return []
    except Exception as e: # Catch other potential file reading errors
        print(f"Error reading file {json_file_path}: {e}")
        return []


    if not isinstance(data, list):
        print(f"Error: JSON data is not a list in {json_file_path}")
        return []

    image_paths = []
    for i, item in enumerate(data):
        # Check if the item is a dictionary and contains the "image" key
        if isinstance(item, dict) and "annotation" in item:
            # Check if the value associated with "image" is a string
            if isinstance(item["annotation"], str):
                image_paths.append(item["annotation"])
            else:
                print(f"Warning: Item {i} in {json_file_path} has 'image' value that is not a string: {item['annotation']}. Skipping.")
        else:
            # Optionally print a warning for items that don't match the expected structure
            # print(f"Warning: Skipping item {i} with unexpected structure in {json_file_path}. Expected a dictionary with an 'image' key.")
            pass # Silently skip items that don't have the expected structure

    return image_paths

def extract_image_paths_from_json2(json_file_path):
    """
    Scans a JSON file with the specified structure and extracts all image paths.

    The expected JSON structure is a list of objects, where each object
    contains an "image" key with the image file path as its value.

    Args:
        json_file_path (str): The path to the input JSON file.

    Returns:
        list: A list of strings, where each string is an image file path.
              Returns an empty list if the file is not found, invalid JSON,
              or has an unexpected structure.
    """
    if not os.path.exists(json_file_path):
        print(f"Error: JSON file not found at {json_file_path}")
        return []

    try:
        with open(json_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON from {json_file_path}: {e}")
        return []
    except Exception as e: # Catch other potential file reading errors
        print(f"Error reading file {json_file_path}: {e}")
        return []


    if not isinstance(data, list):
        print(f"Error: JSON data is not a list in {json_file_path}")
        return []

    image_paths = []
    for i, item in enumerate(data):
        # Check if the item is a dictionary and contains the "image" key
        if isinstance(item, dict) and "image" in item:
            # Check if the value associated with "image" is a string
            if isinstance(item["image"], str):
                image_paths.append(item["image"])
            else:
                print(f"Warning: Item {i} in {json_file_path} has 'image' value that is not a string: {item['image']}. Skipping.")
        else:
            # Optionally print a warning for items that don't match the expected structure
            # print(f"Warning: Skipping item {i} with unexpected structure in {json_file_path}. Expected a dictionary with an 'image' key.")
            pass # Silently skip items that don't have the expected structure

    return image_paths


def main():
    image_dir = "dataset/local/coco82/color"
    output_dir = "dataset/local/coco82"
    coco_json_path = "dataset/local/coco82/_annotations.coco.json"
    depth_dir = "dataset/local/processed_realsense/ahe_depth/png"
    mask_dir = "dataset/local/coco82/mask"

    semantic_dir = "/Users/theobald/Documents/code_lib/python_lib/shrimpDetection/dataset/local/backup/test/cvat/semantic"
    instance_dir = "/Users/theobald/Documents/code_lib/python_lib/shrimpDetection/dataset/local/backup/test/cvat/instance"



    dataset_constructor(image_dir, mask_dir, output_dir,
                        mask_check=False,
                        data_form='coco',
                        semantic_dir=semantic_dir,
                        instance_dir=instance_dir,
                        coco_json_path=coco_json_path,
                        depth_dir=depth_dir)


if __name__ == '__main__':
    main()
