"""
Date: 2024/11/29
Author: Renjie Zhou
Email: nikibandit200814@gmail.com
"""
import json

from PIL import Image
import shutil
import numpy as np
import cv2
import random
import string
import PIL.ImageOps
from tqdm import tqdm
from datasets import Dataset, DatasetDict, Image as datasets_Image
import os
import json


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
    for color, gray_value in color_to_gray_map.items():
        lut[color] = gray_value

    # 将图像转换为灰度图像
    grayscale_image = lut[mask_image[:, :, 0], mask_image[:, :, 1], mask_image[:, :, 2]]

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


def label_check(image_path_list, mask_path_list):
    """
    标签检查
    :param image_path_list: image_path_list
    :param mask_path_list: mask_path_list
    """
    for image_path, mask_path in zip(image_path_list, mask_path_list):
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        mask = cv2.imread(mask_path, cv2.IMREAD_COLOR)
        # 基于opencv打开并处理图像数据，因此sematic mask在第三层；instance mask在第二层
        semantic_mask = mask[..., 2]
        instance_mask = mask[..., 1]
        yellow_mask = np.zeros_like(image, dtype=np.uint8)
        yellow_mask[semantic_mask != 0] = (0, 255, 255)  # 黄色
        blue_mask = np.zeros_like(image, dtype=np.uint8)
        blue_mask[instance_mask !=0] = (0, 0, 255)
        alpha = 0.5  # 透明度
        yellow_mask = cv2.addWeighted(yellow_mask, alpha, np.zeros_like(yellow_mask), 1 - alpha, 0)
        blue_mask = cv2.addWeighted(blue_mask, alpha, np.zeros_like(blue_mask), 1 - alpha, 0)
        image = cv2.addWeighted(image, 1, yellow_mask, 1, 0)
        image = cv2.addWeighted(image, 1, blue_mask, 1, 0)

        semantic_mask = np.where(semantic_mask == 0, 255, semantic_mask)
        instance_mask = np.where(instance_mask == 0, 255, instance_mask)
        semantic_mask = np.dstack((semantic_mask, semantic_mask, semantic_mask))
        instance_mask = np.dstack((instance_mask, instance_mask, instance_mask))

        assert image.shape == semantic_mask.shape == instance_mask.shape
        row = cv2.hconcat([image, semantic_mask, instance_mask])
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


def split2train_and_valid(image_path_list, mask_path_list, valid_rate=0.3):
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

    return train_image_path_list, train_mask_path_list, valid_image_path_list, valid_mask_path_list


def generate_meta_file(train_image_path_list, train_mask_path_list,
                       valid_image_path_list, valid_mask_path_list,
                       output_dir,
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
        semantic_class_to_id = {"background": 0, "shrimp": 1}

    def meta_data_unit(image_path_list, mask_path_list):
        data = []
        for i in range(len(image_path_list)):
            data.append({
                "image": image_path_list[i],
                "annotation": mask_path_list[i],
                "semantic_class_to_id": semantic_class_to_id
            })

        return data

    train_data = meta_data_unit(train_image_path_list, train_mask_path_list)
    valid_data = meta_data_unit(valid_image_path_list, valid_mask_path_list)

    # Write JSON files
    train_json_path = os.path.join(output_dir, "train.json")
    valid_json_path = os.path.join(output_dir, "valid.json")

    with open(train_json_path, "w") as train_file:
        json.dump(train_data, train_file, indent=4)

    with open(valid_json_path, "w") as valid_file:
        json.dump(valid_data, valid_file, indent=4)

    print(f"JSON files generated:\n  Train: {train_json_path}\n  Validation: {valid_json_path}")


def dataset_constructor(image_dir, semantic_dir, instance_dir, mask_dir, output_dir, mask_check=False):
    """
    数据集构造器
    :param image_dir: iamge_dir
    :param semantic_dir: semantic_dir
    :param instance_dir: instance_dir
    :param mask_dir: mask_dir
    :param output_dir: output_dir
    :param mask_check: 是否需要做标签可视化检查
    """
    assert os.path.isdir(image_dir), f"{image_dir} 不存在"
    assert os.path.isdir(semantic_dir), f"{semantic_dir} 不存在"
    assert os.path.isdir(instance_dir), f"{mask_dir} 不存在"
    assert os.path.isdir(output_dir), f"{output_dir} 不存在"
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

    mask_path_list = [os.path.join(mask_dir, mask_name) for mask_name in get_image_name_list(mask_dir)]
    image_path_list = [os.path.join(image_dir, image_name) for image_name in get_image_name_list(image_dir)]
    assert all(os.path.splitext(os.path.basename(image_path))[0] == os.path.splitext(os.path.basename(mask_path))[0]
               for image_path, mask_path in zip(image_path_list, mask_path_list)), "image 和 mask 不匹配"
    if mask_check:
        label_check(image_path_list, mask_path_list)

    train_image_path_list, train_mask_path_list, valid_image_path_list, valid_mask_path_list = split2train_and_valid(image_path_list, mask_path_list)
    generate_meta_file(train_image_path_list, train_mask_path_list, valid_image_path_list, valid_mask_path_list, output_dir)


def main():
    image_dir = "/Users/theobald/Documents/code_lib/python_lib/shrimpDetection/dataset/local/shrimp_test/JPEGImages"
    semantic_dir = "/Users/theobald/Documents/code_lib/python_lib/shrimpDetection/dataset/local/shrimp_test/SegmentationClass"
    instance_dir = "/Users/theobald/Documents/code_lib/python_lib/shrimpDetection/dataset/local/shrimp_test/SegmentationObject"
    mask_dir = "/Users/theobald/Documents/code_lib/python_lib/shrimpDetection/dataset/local/test_mask"
    output_dir = "/Users/theobald/Documents/code_lib/python_lib/shrimpDetection/dataset/local/test_config"
    dataset_constructor(image_dir, semantic_dir, instance_dir, mask_dir, output_dir, mask_check=True)


if __name__ == '__main__':
    main()
