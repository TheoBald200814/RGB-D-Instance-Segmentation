from typing import List, Dict, Tuple
import numpy as np
import random
from mask2former.utils.data_process import (label_check,
                                            split2train_and_valid,
                                            generate_meta_file,
                                            get_image_name_list,
                                            load_coco_annotations,
                                            create_mask_from_polygon)
import shutil
import json
import os
from tqdm import tqdm
from PIL import Image  # Optional, for getting image size if not in JSON


def convert_labelme_to_coco_instance_segmentation(image_path_list, label_path_list, output_coco_json_file):
    """
    将 LabelMe 格式的 JSON 标签文件转换为 COCO 实例分割格式的 JSON 文件。
    增加错误处理，更健壮地处理不同结构的 'points' 数据。

    Args:
        image_path_list (list): 图像文件路径列表.
        label_path_list (list): 对应的 LabelMe JSON 标签文件路径列表.
        output_coco_coco_json_file (str): 输出 COCO 格式 JSON 文件的路径.

    Raises:
        ValueError: 如果 image_path_list 和 label_path_list 列表长度不一致.
    """

    if len(image_path_list) != len(label_path_list):
        raise ValueError("图像文件列表和标签文件列表的长度必须一致.")

    coco_output = {
        "images": [],
        "annotations": [],
        "categories": []
    }

    categories_list = []
    category_name_to_id = {}
    category_id_counter = 1  # COCO category IDs start from 1

    image_id_counter = 1
    annotation_id_counter = 1

    for image_path, label_path in tqdm(zip(image_path_list, label_path_list), total=len(image_path_list)):
        try:
            with open(label_path, 'r', encoding='utf-8') as f: # Specify encoding for robustness
                label_data = json.load(f)
        except Exception as e:
            print(f"Error loading JSON file: {label_path}, error: {e}")
            continue # Skip to the next file if loading fails

        image_filename = os.path.basename(image_path)
        try:
            image_height = label_data['imageHeight']
            image_width = label_data['imageWidth']
        except KeyError:
            try:
                img = Image.open(image_path) # Fallback to reading image size from file if not in JSON
                image_width, image_height = img.size
            except Exception as e:
                print(f"Warning: Could not get image size for {image_path} from JSON or image file. Skipping image. Error: {e}")
                continue
            print(f"Warning: imageHeight/imageWidth not found in {label_path}, getting size from image file.")


        coco_image = {
            "id": image_id_counter,
            "file_name": image_filename,
            "height": image_height,
            "width": image_width,
        }
        coco_output["images"].append(coco_image)

        for shape in label_data['shapes']:
            label_name = shape['label']
            if label_name not in category_name_to_id:
                category_name_to_id[label_name] = category_id_counter
                categories_list.append({'id': category_id_counter, 'name': label_name, 'supercategory': 'object'}) # You can adjust supercategory
                category_id_counter += 1

            category_id = category_name_to_id[label_name]
            shape_type = shape['shape_type']
            points = shape['points']

            if shape_type == 'polygon':
                segmentation = []
                valid_points = True # Flag to track if points are valid

                if not isinstance(points, list): # Check if points is a list
                    print(f"Warning: 'points' is not a list in {label_path}, shape label: {label_name}. Skipping annotation.")
                    valid_points = False
                else:
                    for point in points:
                        if not isinstance(point, list) or len(point) != 2: # Check if each point is a list of 2 coords
                            print(f"Warning: Invalid point format in {label_path}, shape label: {label_name}, point: {point}. Skipping annotation.")
                            valid_points = False
                            break # Exit inner loop if invalid point found
                        try:
                            segmentation.extend(list(map(float, point))) # Convert coords to float and extend
                        except TypeError as e:
                            print(f"Warning: TypeError converting point coordinates to float in {label_path}, shape label: {label_name}, point: {point}. Skipping annotation. Error: {e}")
                            valid_points = False
                            break # Exit inner loop if type error in conversion

                if not valid_points:
                    continue # Skip to next shape if points are invalid

                segmentation = [segmentation] # COCO segmentation is a list of polygons, even if only one


                # Calculate area (using shoelace formula)
                area = 0.0
                for i in range(len(points)):
                    x1, y1 = points[i]
                    x2, y2 = points[(i + 1) % len(points)]
                    area += (x1 * y2 - x2 * y1)
                area = abs(area) / 2.0

                # Calculate bounding box (xywh)
                x_coords = [point[0] for point in points]
                y_coords = [point[1] for point in points]
                x_min = min(x_coords)
                y_min = min(y_coords)
                x_max = max(x_coords)
                y_max = max(y_coords)
                bbox = [x_min, y_min, x_max - x_min, y_max - y_min]


                coco_annotation = {
                    "id": annotation_id_counter,
                    "image_id": image_id_counter,
                    "category_id": category_id,
                    "segmentation": segmentation,
                    "area": area,
                    "bbox": bbox,
                    "iscrowd": 0 # 0 for instance segmentation, 1 for crowd (if applicable)
                }
                coco_output["annotations"].append(coco_annotation)
                annotation_id_counter += 1
            else:
                print(f"Warning: Shape type '{shape_type}' in {label_path} is not 'polygon'. Skipping annotation.")

        image_id_counter += 1

    coco_output["categories"] = list(categories_list) # Ensure categories are in list format

    with open(output_coco_json_file, 'w') as outfile:
        json.dump(coco_output, outfile, indent=4) # Indent for better readability

    print(f"COCO instance segmentation JSON file saved to: {output_coco_json_file}")


def generate_random_color():
    """生成一个随机的 RGB 颜色"""
    return (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))


def coco_category_id_constructor(coco_path, output_dir):
    with open(coco_path, 'r') as f:
        coco_data = json.load(f)

    categories_array = coco_data['categories']
    category_name2id = {}  # 创建一个字典来存储 category_id 到 name 的映射

    for category_info in categories_array:
        category_id = category_info['id']
        category_name = category_info['name']
        category_name2id[category_name] = category_id

    output_json_file_path = os.path.join(output_dir, 'label2id.json')

    with open(output_json_file_path, 'w') as f:
        json.dump(category_name2id, f, indent=4)

    print(f"字典已保存到 JSON 文件: {output_json_file_path}")

    return category_name2id


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
        image_name = img_info['file_name'].split(".")
        output_path = os.path.join(
            output_dir,
            image_name[0] + ".png"
        )
        Image.fromarray(combined).save(output_path)


def dataset_constructor(image_dir, label_dir, mask_dir, coco_path, output_dir, check=False, depth_dir=None):
    """
    数据集构造器
    :param image_dir: RGB图像文件夹路径
    :param label_dir: 官方json标签文件夹路径
    :param coco_path: 希望生成coco.json文件的路径
    :param mask_dir: 希望生成mask的文件夹路径
    :param check: 构造过程中是否需要可视化标签检查
    :param output_dir: 希望生成train.json & valid.json的文件夹路径
    :param depth_dir: 深度图像文件夹路径
    """
    image_path_list = [os.path.join(image_dir, i) for i in os.listdir(image_dir) if i != '.DS_Store']
    label_path_list = [os.path.join(label_dir, i) for i in os.listdir(label_dir) if i != '.DS_Store']
    image_path_list = sorted(image_path_list, key=lambda x: os.path.basename(x))
    label_path_list = sorted(label_path_list, key=lambda x: os.path.basename(x))

    # convert_labelme_to_coco_instance_segmentation(image_path_list, label_path_list, coco_path)

    img_data, ann_data = load_coco_annotations(coco_path)
    # generate_combined_masks(img_data, ann_data, mask_dir, image_dir)

    mask_path_list = [os.path.join(mask_dir, i) for i in os.listdir(mask_dir) if i != '.DS_Store']
    mask_path_list = sorted(mask_path_list, key=lambda x: os.path.basename(x))

    assert all(os.path.basename(image_path).split('.')[0] == os.path.basename(mask_path).split('.')[0]
               for image_path, mask_path in zip(image_path_list, mask_path_list)), "image 和 mask 不匹配"
    depth_path_list = None
    if depth_dir is not None:
        depth_path_list = [os.path.join(depth_dir, image_name) for image_name in get_image_name_list(image_dir)]
        assert all(os.path.basename(image_path).split('.')[0] == os.path.basename(depth_path).split('.')[0]
                   for image_path, depth_path in zip(image_path_list, depth_path_list)), "image和depth 不匹配"

    if check:
        label_check(image_path_list[:50], mask_path_list[:50])

    train_image_path_list, train_mask_path_list, train_depth_path_list, valid_image_path_list, valid_mask_path_list, valid_depth_path_list = split2train_and_valid(
        image_path_list, mask_path_list, depth_path_list)
    generate_meta_file(train_image_path_list, train_mask_path_list, train_depth_path_list,
                       valid_image_path_list, valid_mask_path_list, valid_depth_path_list,
                       output_dir)
    coco_category_id_constructor(coco_path, output_dir)


def reload_original_data_structure(src_dir, aim_json_dir):
    json_path_list = [os.path.join(src_dir, i) for i in os.listdir(src_dir) if os.path.splitext(i)[1] == '.json']
    os.makedirs(aim_json_dir, exist_ok=True)
    for json_path in json_path_list:
        shutil.move(json_path, os.path.join(aim_json_dir, os.path.basename(json_path)))


def main():
    image_dir = "dataset/local/archive/multi_set/color"
    label_dir = "/Users/theobald/Downloads/archive/multi_set/json"
    mask_dir = "dataset/local/archive/multi_set/mask"
    depth_dir = "dataset/local/archive/multi_set/height"
    coco_save_path = "dataset/local/archive/multi_set/coco.json"
    output_path = "dataset/local/archive/multi_set"

    dataset_constructor(image_dir=image_dir,
                        label_dir=label_dir,
                        mask_dir=mask_dir,
                        coco_path=coco_save_path,
                        check=False,
                        output_dir=output_path,
                        depth_dir=depth_dir)


if __name__ == '__main__':
    # main()
    coco82v2_dir = "dataset/local/coco82v2"
    img_name_list = [i for i in os.listdir(os.path.join(coco82v2_dir, "eq")) if i != '.DS_Store']
    img_path_list = [os.path.join(coco82v2_dir + '/color', i) for i in img_name_list]
    mask_path_list = [os.path.join(coco82v2_dir + '/mask', i) for i in img_name_list]
    output_path ="/Users/theobald/Documents/code_lib/python_lib/shrimpDetection/dataset/local/coco82v2"
    depth_path_list = [os.path.join(coco82v2_dir + '/ahe', i) for i in img_name_list]
    # [RGB, DECIMATION, RS, SPATIAL, HOLE_FILLING, AHE, LAPLACE, GAUSSIAN, EQ, LT]
    depth_expand_list_dict = {
        "decimation_depth": [os.path.join(coco82v2_dir + '/decimation', i) for i in img_name_list],
        "depth_colormap_by_rs": [os.path.join(coco82v2_dir + '/rs', i) for i in img_name_list],
        "spatial_depth": [os.path.join(coco82v2_dir + '/spatial', i) for i in img_name_list],
        "hole_filling_depth": [os.path.join(coco82v2_dir + '/hole_filling', i) for i in img_name_list],
        "ahe_depth": [os.path.join(coco82v2_dir + '/ahe', i) for i in img_name_list],
        "laplace_depth": [os.path.join(coco82v2_dir + '/laplace', i) for i in img_name_list],
        "gaussian_depth": [os.path.join(coco82v2_dir + '/gaussian', i) for i in img_name_list],
        "eq_depth": [os.path.join(coco82v2_dir + '/eq', i) for i in img_name_list],
        "lt_depth": [os.path.join(coco82v2_dir + '/lt', i) for i in img_name_list],
    }
    (train_image_path_list, train_mask_path_list, train_depth_expand_list_dict,
     valid_image_path_list, valid_mask_path_list, valid_depth_expand_list_dict) = split2train_and_valid(img_path_list, mask_path_list, depth_expand_list_dict=depth_expand_list_dict)
    generate_meta_file(train_image_path_list, train_mask_path_list, None, train_depth_expand_list_dict,
                       valid_image_path_list, valid_mask_path_list, None, valid_depth_expand_list_dict,
                       output_path)
