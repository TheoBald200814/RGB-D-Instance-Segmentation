from PIL import Image
import shutil
import numpy as np
import os
import cv2
import os
import random
import string
import PIL.Image
import PIL.ImageOps
from tqdm import tqdm

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


def label_check(images_path: str, labels_path: str):
    """
    标签检查
    :param images_path: images path
    :param labels_path: labels path
    """
    image_name_list = get_image_name_list(images_path)
    label_name_list = get_image_name_list(labels_path)
    assert len(image_name_list) == len(label_name_list), "图片与标签图片规模不一致"
    for i in range(len(image_name_list)):
        image_path = os.path.join(images_path, image_name_list[i])
        label_path = os.path.join(labels_path, label_name_list[i])
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        mask = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)
        assert image.shape[:2] == mask.shape, "图片与标签图片尺寸不一致"

        # 创建一个黄色的掩码
        yellow_mask = np.zeros_like(image, dtype=np.uint8)
        yellow_mask[mask != 255] = (0, 255, 255)  # 黄色

        # 将黄色掩码设置为半透明
        alpha = 0.5  # 透明度
        yellow_mask = cv2.addWeighted(yellow_mask, alpha, np.zeros_like(yellow_mask), 1 - alpha, 0)

        # 将黄色掩码叠加到原始图片上
        composite_image = cv2.addWeighted(image, 1, yellow_mask, 1, 0)

        # 显示图像
        cv2.imshow('Composite Image', composite_image)
        cv2.waitKey(0)  # 等待按键事件
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
    # 确保输出文件夹存在
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 读取视频
    cap = cv2.VideoCapture(video_path)

    frame_count = 0
    save_count = 0
    while True:
        # 读取下一帧
        ret, frame = cap.read()

        # 如果正确读取到了帧，则ret为True
        if not ret:
            break

        # 检查是否达到保存帧的条件
        if frame_count % frame_interval == 0:
            # 构造输出文件名
            filename = os.path.join(output_folder, rename_with_random_string() + ".jpg")

            # 保存当前帧为图片
            cv2.imwrite(filename, frame)

            print(f"Saved {filename}")
            save_count += 1

        frame_count += 1

    # 释放视频捕获对象
    cap.release()


# if __name__ == "__main__":
    # directory_to_process = "/Users/theobald/Documents/code_lib/python_lib/huggingface_data_process/datasets/local/24_11_19/val"
    # if os.path.isdir(directory_to_process):
    #     image_name_list = get_image_name_list(directory_to_process)
    #     for image_name in tqdm(image_name_list):
    #         image_path = os.path.join(directory_to_process, image_name)
    #         img = load_image_file(image_path)
    #         img.save(image_path)
    # else:
    #     print("提供的路径不是一个有效的目录")

def main():
    sematic_mask_dir = "/Users/theobald/Documents/code_lib/python_lib/huggingface_data_process/datasets/local/job_1_dataset_2024_11_21_07_32_42_segmentation mask 1.1/SegmentationClass"
    instance_mask_dir = "/Users/theobald/Documents/code_lib/python_lib/huggingface_data_process/datasets/local/job_1_dataset_2024_11_21_07_32_42_segmentation mask 1.1/SegmentationObject"
    mask_save_dir = "/Users/theobald/Documents/code_lib/python_lib/huggingface_data_process/datasets/local/job_1_dataset_2024_11_21_07_32_42_segmentation mask 1.1/mask"

    sematic_mask_image_name_list = get_image_name_list(sematic_mask_dir)
    instance_mask_image_name_list = get_image_name_list(instance_mask_dir)

    assert len(sematic_mask_image_name_list) == len(instance_mask_image_name_list)
    for i in tqdm(range(len(sematic_mask_image_name_list))):
        assert sematic_mask_image_name_list[i] == instance_mask_image_name_list[i]
        sematic_mask_path = os.path.join(sematic_mask_dir, sematic_mask_image_name_list[i])
        instance_mask_path = os.path.join(instance_mask_dir, instance_mask_image_name_list[i])
        mask = combine_sematic_instance_mask(sematic_mask_path, instance_mask_path)
        save_path = os.path.join(mask_save_dir, sematic_mask_image_name_list[i])

        cv2.imwrite(save_path, mask)


if __name__ == '__main__':
    main()
