import pyrealsense2 as rs
import numpy as np
import cv2
import string
import random
import os
import copy


def init(file_path: str):
    """
    初始化pipeline
    :param file_path: 文件路径
    :return: pipeline
    """
    pipeline = rs.pipeline()
    config = rs.config()
    rs.config.enable_device_from_file(config, file_path, repeat_playback=False)
    pipeline.start(config)
    return pipeline


def histogram_equalization(image):
    """
    直方图均衡化
    :param image: image
    :return: image
    """
    # image = uint8_convert(image)
    return cv2.equalizeHist(image)


def adaptive_histogram_equalization(image, clipLimit=2.0, tileGridSize=(8,8)):
    """
    自适应直方图均衡化
    :param image: image
    :param clipLimit:
    :param tileGridSize:
    :return: image
    """
    clahe = cv2.createCLAHE(clipLimit=clipLimit, tileGridSize=tileGridSize)
    return clahe.apply(image)


def linear_transformation(image, a=1.5, b=-50):
    """
    线性变换
    :param image: image
    :param a:
    :param b:
    :return: image
    """
    # 确保像素值在 [0, 255] 范围内
    return np.clip(a * image + b, 0, 255).astype(np.uint8)


def gamma_correction(image, gamma=0.5):
    """
    Gamma 校正
    :param image: image
    :param gamma:
    :return: image
    """
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    # image = uint8_convert(image)
    return cv2.LUT(image, table)


def laplacian_sharpening(image):
    """
    拉普拉斯锐化
    :param image: image
    :return: image
    """
    kernel = np.array([[0, -1, 0],
                       [-1, 5,-1],
                       [0, -1, 0]])
    return cv2.filter2D(image, -1, kernel)


def gaussian_blur_and_subtraction(image, sigma=1):
    """
    高斯模糊和减法
    :param image: image
    :param sigma:
    :return: image
    """
    blurred = cv2.GaussianBlur(image, (0, 0), sigmaX=sigma, sigmaY=sigma)
    return cv2.addWeighted(image, 1.5, blurred, -0.5, 0)


def uint8_convert(depth_image):
    """
    转换为8位无符号整数类型的单通道图像
    :param depth_image:image
    :return:image
    """
    if depth_image.dtype != np.uint8 or len(depth_image.shape) != 2:
        depth_image = cv2.convertScaleAbs(depth_image, alpha=0.1)
    return depth_image


def do_depth_image_process(depth_image):
    """
    深度图像数据处理（自适应直方图均衡化、线性变换、拉普拉斯锐化、高斯模糊和减法、直方图均衡化、Gamma 校正）
    :param depth_image: depth_image
    :return: process images list
    """
    uint8_depth_image = uint8_convert(depth_image)  # 转换为8位无符号整数类型的单通道图像

    ahe_depth = adaptive_histogram_equalization(uint8_depth_image) # 自适应直方图均衡化
    lt_depth = linear_transformation(uint8_depth_image) # 线性变换
    laplace_depth = laplacian_sharpening(uint8_depth_image) # 拉普拉斯锐化
    gaussian_depth = gaussian_blur_and_subtraction(uint8_depth_image) # 高斯模糊和减法

    eq_depth = histogram_equalization(uint8_depth_image) # 直方图均衡化
    gamma_depth = gamma_correction(uint8_depth_image) # Gamma 校正

    return [ahe_depth, lt_depth, laplace_depth, gaussian_depth, eq_depth, gamma_depth]


def do_depth_image_filter(depth_frame, colorizer):
    decimation_depth_frame = depth_image_filter(decimation_filter(), depth_frame)
    decimation_depth_color_image = np.asanyarray(colorizer.colorize(decimation_depth_frame).get_data())
    decimation_depth_gray_image = np.asarray(decimation_depth_frame.get_data())

    spatial_depth_frame = depth_image_filter(spatial_filter(), depth_frame)
    spatial_depth_color_image = np.asanyarray(colorizer.colorize(spatial_depth_frame).get_data())
    spatial_depth_gray_image = np.asarray(spatial_depth_frame.get_data())

    hole_filling_depth_frame = depth_image_filter(hole_filling_filter(), depth_frame)
    hole_filling_depth_color_image = np.asanyarray(colorizer.colorize(hole_filling_depth_frame).get_data())
    hole_filling_depth_gray_image = np.asarray(hole_filling_depth_frame.get_data())

    return [[decimation_depth_color_image, spatial_depth_color_image, hole_filling_depth_color_image],
            [decimation_depth_gray_image, spatial_depth_gray_image, hole_filling_depth_gray_image]]


def depth2color(depth_image):
    """
    深度图像数据转颜色图像数据
    """
    return cv2.cvtColor(depth_image, cv2.COLOR_GRAY2BGR)


def depth_image_filter(func, depth_image):
    """
    filter接口
    :param func: 过滤函数
    :param depth_image: depth image
    :return: depth image
    """
    return func.process(depth_image)


def decimation_filter(level=4):
    """
    抽取过滤器
    :return:decimation
    """
    decimation = rs.decimation_filter()
    decimation.set_option(rs.option.filter_magnitude, level) # 设置降采样的程度，通常取值范围为1-8。较高的值会导致更多的下采样。
    return decimation


def spatial_filter(level=5, smooth_alpha=1, smooth_delta=50):
    """
    空间过滤器
    :return:spatial
    """
    spatial = rs.spatial_filter()
    # 可以通过增加smooth_alpha和smooth_delta选项来强调滤镜的效果
    spatial.set_option(rs.option.filter_magnitude, level) # 设置滤波器的强度（较高的值会影响更多像素）。
    spatial.set_option(rs.option.filter_smooth_alpha, smooth_alpha) # 影响平滑程度的系数，值越小，平滑程度越大。
    spatial.set_option(rs.option.filter_smooth_delta, smooth_delta) # 控制平滑过程中深度值的变化范围。

    # 该过滤器还提供一些基本的空间孔填充功能：
    spatial.set_option(rs.option.holes_fill, 1)
    return spatial


def hole_filling_filter():
    """
    孔填充过滤器
    :return:hole_filling
    """
    hole_filling = rs.hole_filling_filter()
    return hole_filling


def resize2match(*images: np.ndarray, interpolation=cv2.INTER_LINEAR) -> tuple:
    """
    将任意数量的图像调整为相同的尺寸，以第一个图像的尺寸为标准。

    参数:
        *images: 任意数量的图像，类型为 numpy 数组。
        interpolation: 调整图像大小时使用的插值方法，默认为 INTER_LINEAR。

    返回:
        tuple: 返回一个元组，包含调整后的所有图像。
    """
    # 确保所有输入都是有效的图像
    for img in images:
        if not isinstance(img, np.ndarray):
            raise ValueError("每个输入参数必须是 numpy 数组类型的图像")

    # 获取第一个图像的尺寸作为目标尺寸
    target_height, target_width = images[0].shape[:2]

    resized_images = []

    # 遍历每个图像，调整尺寸
    for img in images:
        # 如果尺寸不匹配，调整当前图像的尺寸
        if img.shape[:2] != (target_height, target_width):
            print(f"尺寸不匹配：调整图像尺寸从 {img.shape[:2]} 到 ({target_height}, {target_width})")
            img = cv2.resize(img, (target_width, target_height), interpolation=interpolation)
        resized_images.append(img)

    return tuple(resized_images)


def preload_frames(file_path: str, timeout_ms=10000):
    """
    frames预加载，序列化图像数据（color and depth）
    :param file_path: bag文件路径
    :param timeout_ms: timeout_ms
    :return: image_list
    """
    # 创建一个管道对象
    pipeline = init(file_path)
    image_list = []
    # 预先加载所有帧
    t = 0
    try:
        print("Preloading all frames...")
        while True:
            frame = pipeline.wait_for_frames(timeout_ms=timeout_ms)
            color_frame = frame.get_color_frame()
            depth_frame = frame.get_depth_frame()

            colorizer = rs.colorizer()
            t += 1
            print(t)

            # # 将帧转换为numpy数组
            color_image = np.asanyarray(color_frame.get_data())
            depth_image = np.asanyarray(depth_frame.get_data())
            depth_colormap_by_cv = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.1),
                                                     cv2.COLORMAP_JET)  # 伪彩色图像变换（cv）
            depth_colormap_by_rs = np.asanyarray(colorizer.colorize(depth_frame).get_data())

            # 三种filter处理过后的图像集（filter_depth_image_list中包括colorizer处理过后的颜色图和灰度图）
            filter_depth_image_list = do_depth_image_filter(depth_frame, colorizer)

            # 六种图像处理手段处理过后的图像集
            processed_depth_image_list = [depth2color(image) for image in do_depth_image_process(depth_image)]

            (color, depth_colormap_by_cv, depth_colormap_by_rs,
             decimation_depth, spatial_depth, hole_filling_depth,
             ahe_depth, lt_depth, laplace_depth,
             gaussian_depth, eq_depth, gamma_depth) = resize2match(color_image, depth_colormap_by_cv,
                                                                   depth_colormap_by_rs,
                                                                   filter_depth_image_list[0][0],
                                                                   filter_depth_image_list[0][1],
                                                                   filter_depth_image_list[0][2],
                                                                   processed_depth_image_list[0],
                                                                   processed_depth_image_list[1],
                                                                   processed_depth_image_list[2],
                                                                   processed_depth_image_list[3],
                                                                   processed_depth_image_list[4],
                                                                   processed_depth_image_list[5])

            image = {
                "color": color,
                "depth_colormap_by_cv": depth_colormap_by_cv,
                "depth_colormap_by_rs": depth_colormap_by_rs,
                "decimation_depth": decimation_depth,
                "spatial_depth": spatial_depth,
                "hole_filling_depth": hole_filling_depth,
                "ahe_depth": ahe_depth,
                "lt_depth": lt_depth,
                "laplace_depth": laplace_depth,
                "gaussian_depth": gaussian_depth,
                "eq_depth": eq_depth,
                "gamma_depth": gamma_depth,
            }
            copy_image = copy.deepcopy(image)
            del image
            image_list.append(copy_image)

    except Exception as e:
        print(e)
        print(f"All frames preloaded. Total frames: {len(image_list)}")
        pipeline.stop()

    return image_list


def checkout(file_path: str, save_dir: str):
    """
    数据检查
    :param file_path: bag文件路径
    """
    image_list = preload_frames(file_path)
    current_frame_index = 0  # 当前帧索引
    try:
        while True:
            # 确保当前帧索引在合法范围内
            current_frame_index = max(0, min(current_frame_index, len(image_list) - 1))
            image = image_list[current_frame_index]
            show(image)

            # 处理键盘输入
            key = cv2.waitKey(0)  # 等待按键输入
            if key & 0xFF == ord('q') or key == 27:  # 按 esc 或 'q' 退出
                cv2.destroyAllWindows()
                break
            elif key & 0xFF == ord('a'):  # 播放上一帧
                current_frame_index -= 1
            elif key & 0xFF == ord('d'):  # 播放下一帧
                current_frame_index += 1
            elif key & 0xFF == ord('s'): # 保存该帧数据
                enum_image_type = ("color",
                                   "depth_colormap_by_cv",
                                   "depth_colormap_by_rs",
                                   "decimation_depth",
                                   "spatial_depth",
                                   "hole_filling_depth",
                                   "ahe_depth",
                                   "lt_depth",
                                   "laplace_depth",
                                   "gaussian_depth",
                                   "eq_depth",
                                   "gamma_depth")
                init_save_dir(save_dir, enum_image_type)
                save_frame(image, current_frame_index, save_dir, enum_image_type)
    finally:
        cv2.destroyAllWindows()


def show(image):
    """
    数据展示
    # ====================================================
    # -------------------原始预览--------------------------
    # 原始RGB图像
    # cv color映射的深度图像
    # rs color映射的深度图像
    # -------------------过滤预览--------------------------
    # decimation_filter过滤的深度图像（基于rs color显示）
    # spatial_filter过滤的深度图像（基于rs color显示）
    # hole_filling_filter过滤的深度图像（基于rs color显示）
    # -------------------变换预览--------------------------
    # 自适应直方图均衡化
    # 线性变换
    # 拉普拉斯锐化
    # 高斯模糊和减法
    # 直方图均衡化
    # Gamma 校正
    # ====================================================
    :param image: 序列化image_list
    """
    color = image["color"]
    depth_colormap_by_cv = image["depth_colormap_by_cv"]
    depth_colormap_by_rs = image["depth_colormap_by_rs"]
    decimation_depth = image["decimation_depth"]
    spatial_depth = image["spatial_depth"]
    hole_filling_depth = image["hole_filling_depth"]
    ahe_depth = image["ahe_depth"]
    lt_depth = image["lt_depth"]
    laplace_depth = image["laplace_depth"]
    gaussian_depth = image["gaussian_depth"]
    eq_depth = image["eq_depth"]
    gamma_depth = image["gamma_depth"]
    row1 = cv2.hconcat([color, depth_colormap_by_cv, depth_colormap_by_rs])
    row2 = cv2.hconcat([decimation_depth, spatial_depth, hole_filling_depth])
    row3 = cv2.hconcat([ahe_depth, lt_depth, laplace_depth])
    row4 = cv2.hconcat([gaussian_depth, eq_depth, gamma_depth])
    row = cv2.vconcat([row1, row2, row3, row4])
    cv2.imshow("display", row)


def generate_random_string(length=10):
    """生成指定长度的随机字符串"""
    letters = string.ascii_lowercase

    return ''.join(random.choice(letters) for i in range(length))


def init_save_dir(save_path: str, enum_image_type: tuple):
    """
    构建数据存储目录：
        ｜- color
        ｜｜- png
        ｜｜- npy
        ｜- depth_colormap_by_cv
        ｜｜- png
        ｜｜- npy
        ｜...
    :param save_path: save_path
    """
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    if len(os.listdir(save_path)) == 0:
        enum_file_type = ("png", "npy")
        try:
            for image_type in enum_image_type:
                os.mkdir(os.path.join(save_path, image_type))
                for file_type in enum_file_type:
                    os.mkdir(os.path.join(save_path, image_type, file_type))
        except Exception as e:
            print(e)
    else:
        print(f"检查{save_path}下的文件结构")


def save_frame(image: dict, frame_index: int, save_directory: str, enum_image_type: tuple):
    """
    保存当前帧数据为 PNG 和 NumPy 格式
    :param image: 当前帧图像数据 (NumPy 数组)
    :param frame_index: 当前帧索引
    :param save_directory: 保存目录
    """
    random_str = generate_random_string()

    for image_type in enum_image_type:
        data = image[image_type]
        png_filename = os.path.join(save_directory, image_type, "png", f"{random_str}_" + str(frame_index) + ".png")
        npy_filename = os.path.join(save_directory, image_type, "npy", f"{random_str}_" + str(frame_index) + ".npy")

        # 保存为 PNG 格式
        cv2.imwrite(png_filename, data)
        print(f"Saved PNG: {png_filename}")

        # 保存为 NumPy 格式
        np.save(npy_filename, data)
        print(f"Saved NumPy: {npy_filename}")


def main():
    file_path = "/Users/theobald/Library/Mobile Documents/com~apple~CloudDocs/datasets/instance_seg_shrimp/intel_realsense/24_12_03/recording_9392_20241203.bag"
    save_dir = "/Users/theobald/Documents/code_lib/python_lib/shrimpDetection/dataset/local/processed_realsense"
    checkout(file_path, save_dir)
    # 已经处理的数据：'/Users/theobald/Library/Mobile Documents/com~apple~CloudDocs/datasets/instance_seg_shrimp/intel_realsense/24_11_26'
    #             ：'/Users/theobald/Library/Mobile Documents/com~apple~CloudDocs/datasets/instance_seg_shrimp/intel_realsense/24_12_03'


if __name__ == "__main__":
    main()