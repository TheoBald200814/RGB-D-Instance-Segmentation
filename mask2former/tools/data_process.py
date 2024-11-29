from PIL import Image
import shutil
import numpy as np
import cv2
import os
import random
import string
import PIL.ImageOps
from tqdm import tqdm
from datasets import Dataset, DatasetDict, Image as datasets_Image


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


def label_check(image_dir, mask_dir, image_name_list):
    """
    标签检查
    :param image_dir: image_dir
    :param mask_dir: mask_dir
    :param image_name_list: image_name_list
    """
    for image_name in image_name_list:
        image_path = os.path.join(image_dir, image_name)
        mask_path = os.path.join(mask_dir, os.path.splitext(image_name)[0] + '.png')
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        mask = cv2.imread(mask_path, cv2.IMREAD_COLOR)
        # 基于opencv打开并处理图像数据，因此sematic mask在第三层；instance mask在第二层
        sematic_mask = mask[..., 2]
        instance_mask = mask[..., 1]
        sematic_mask = np.where(sematic_mask == 0, 255, sematic_mask)
        instance_mask = np.where(instance_mask == 0, 255, instance_mask)
        sematic_mask = np.dstack((sematic_mask, sematic_mask, sematic_mask))
        instance_mask = np.dstack((instance_mask, instance_mask, instance_mask))
        assert image.shape == sematic_mask.shape == instance_mask.shape
        row = cv2.hconcat([image, sematic_mask, instance_mask])
        cv2.imshow("image & sematic & instance", row)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def old_label_check(images_path: str, labels_path: str):
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


def create_dataset(image_paths, label_paths):
    """
    create dataset
    :param image_paths: the list of image path
    :param label_paths: the list of label path
    :return: dataset
    """
    dataset = Dataset.from_dict({"image": sorted(image_paths),
                                 "annotation": sorted(label_paths),
                                 "semantic_class_to_id": [{"shrimp": 0}] * len(image_paths)
                                 })
    dataset = dataset.cast_column("image", datasets_Image())
    dataset = dataset.cast_column("annotation", datasets_Image())

    return dataset


def local_dataset_constructor(image_dir: str, label_dir: str):
    """
    本地数据集构造器
    :param image_dir: iamge_dir
    :param label_dir: label_dir
    :return: dataset
    """
    image_name_list = get_image_name_list(image_dir)
    label_name_list = get_image_name_list(label_dir)
    image_paths = [os.path.join(image_dir, x) for x in image_name_list]
    label_paths = [os.path.join(label_dir, x) for x in label_name_list]
    dataset = create_dataset(image_paths, label_paths)
    dataset = DatasetDict({
        "train": dataset,
        "validation": dataset
    })

    return dataset


def ready2training(image_dir='', mask_dir='', sematic_dir='', instance_dir='', do_mask=True, check=False):
    """
    准备训练数据，构造dataset
    :param check: 是否需要标签可视化检查
    :param image_dir: image_dir
    :param do_mask: 是否需要构造mask
    :param mask_dir: mask_dir
    :param sematic_dir: sematic_dir
    :param instance_dir: instance_dir
    :return: dataset
    """
    assert os.path.isdir(image_dir), f"{image_dir} 不存在"
    assert os.path.isdir(mask_dir), f"{mask_dir} 不存在"
    assert (not do_mask or (os.path.isdir(sematic_dir) and os.path.isdir(instance_dir))), "数据缺失，无法准备训练数据"
    image_name_list = get_image_name_list(image_dir)
    if do_mask:
        assert len(os.listdir(mask_dir)) == 0, f"{mask_dir} 不为空，妨碍mask存储"
        sematic_name_list = get_image_name_list(sematic_dir)
        instance_name_list = get_image_name_list(instance_dir)
        assert sematic_name_list == instance_name_list, "sematic mask 和 instance mask不匹配"
        for mask_name in tqdm(sematic_name_list):
            sematic_path = os.path.join(sematic_dir, mask_name)
            instance_path = os.path.join(instance_dir, mask_name)
            mask = combine_sematic_instance_mask(sematic_path, instance_path)
            save_path = os.path.join(mask_dir, mask_name)
            cv2.imwrite(save_path, mask)

    mask_name_list = get_image_name_list(mask_dir)
    assert all(os.path.splitext(image_name)[0] == os.path.splitext(mask_name)[0] for image_name, mask_name in zip(image_name_list, mask_name_list)), "image 和 mask 不匹配"
    if check:
        label_check(image_dir, mask_dir, image_name_list)
    dataset = local_dataset_constructor(image_dir, mask_dir)

    return dataset


def main():
    image_dir = "/Users/theobald/Documents/code_lib/python_lib/shrimpDetection/dataset/local/shrimp_test/JPEGImages"
    mask_dir = "/Users/theobald/Documents/code_lib/python_lib/shrimpDetection/dataset/local/shrimp_test/mask"
    sematic_dir = "/Users/theobald/Documents/code_lib/python_lib/shrimpDetection/dataset/local/shrimp_test/SegmentationClass"
    instance_dir = "/Users/theobald/Documents/code_lib/python_lib/shrimpDetection/dataset/local/shrimp_test/SegmentationObject"
    ready2training(image_dir, mask_dir, sematic_dir, instance_dir,do_mask=False, check=True)


if __name__ == '__main__':
    main()
