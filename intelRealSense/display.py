import pyrealsense2 as rs
import numpy as np
import cv2


def init(file_path: str):
    """
    初始化pipeline
    :param file_path: 文件路径
    :return: pipeline
    """
    pipeline = rs.pipeline()
    config = rs.config()
    rs.config.enable_device_from_file(config, file_path, repeat_playback=True)
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


def depth_image_process(depth_image):
    """
    深度图像数据处理（自适应直方图均衡化、线性变换、拉普拉斯锐化、高斯模糊和减法、直方图均衡化、Gamma 校正）
    :param depth_image: depth_image
    :return: process images list
    """
    uint8_depth_image = uint8_convert(depth_image)  # 转换为8位无符号整数类型的单通道图像
    depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.1), cv2.COLORMAP_JET) # 伪彩色图像变换

    ahe_depth = adaptive_histogram_equalization(uint8_depth_image) # 自适应直方图均衡化
    lt_depth = linear_transformation(uint8_depth_image) # 线性变换
    laplace_depth = laplacian_sharpening(uint8_depth_image) # 拉普拉斯锐化
    gaussian_depth = gaussian_blur_and_subtraction(uint8_depth_image) # 高斯模糊和减法

    eq_depth = histogram_equalization(uint8_depth_image) # 直方图均衡化
    gamma_depth = gamma_correction(uint8_depth_image) # Gamma 校正

    return [[depth_colormap], [ahe_depth, lt_depth, laplace_depth, gaussian_depth, eq_depth, gamma_depth]]


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


def decimation_filter():
    """
    抽取过滤器
    :return:decimation
    """
    decimation = rs.decimation_filter()
    decimation.set_option(rs.option.filter_magnitude, 4)
    return decimation


def spatial_filter():
    """
    空间过滤器
    :return:spatial
    """
    spatial = rs.spatial_filter()
    # 可以通过增加smooth_alpha和smooth_delta选项来强调滤镜的效果
    spatial.set_option(rs.option.filter_magnitude, 5)
    spatial.set_option(rs.option.filter_smooth_alpha, 1)
    spatial.set_option(rs.option.filter_smooth_delta, 50)

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


def resize2match(image1, image2):
    """
    图像尺寸重构
    """
    height1, width1 = image1.shape[:2]
    height2, width2 = image2.shape[:2]

    if (height1, width1) != (height2, width2):
        # 如果尺寸不匹配，调整 image2 的尺寸以匹配 image1
        image2 = cv2.resize(image2, (width1, height1), interpolation=cv2.INTER_LINEAR)

    return image1, image2


def display(file_path: str):
    # 创建一个管道对象
    pipeline = init(file_path)
    alpha = 0.1

    try:
        while True:
            # 等待一帧数据
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            depth_frame = frames.get_depth_frame()
            if not color_frame or not depth_frame:
                continue

            colorizer = rs.colorizer()

            # 将帧转换为numpy数组
            color_image = np.asanyarray(color_frame.get_data())
            depth_image = np.asanyarray(depth_frame.get_data())

            # color_image = np.asanyarray(colorizer.colorize(color_frame).get_data())
            # depth_image = np.asanyarray(colorizer.colorize(depth_frame).get_data())

            # decimation_depth_frame = depth_image_filter(decimation_filter(), depth_frame)
            # decimation_depth_image = np.asanyarray(colorizer.colorize(decimation_depth_frame).get_data())
            #
            # spatial_depth_frame = depth_image_filter(spatial_filter(), depth_frame)
            # spatial_depth_image = np.asanyarray(colorizer.colorize(spatial_depth_frame).get_data())
            #
            # hole_filling_depth_frame = depth_image_filter(hole_filling_filter(), depth_frame)
            # hole_filling_depth_image = np.asanyarray(colorizer.colorize(hole_filling_depth_frame).get_data())
            #
            # depth_image, decimation_depth_image = resize_to_match(depth_image, decimation_depth_image)
            # depth_image, spatial_depth_image = resize_to_match(depth_image, spatial_depth_image)
            # depth_image, hole_filling_image = resize_to_match(depth_image, hole_filling_depth_image)
            #
            # row_1 = cv2.hconcat([depth_image, decimation_depth_image, spatial_depth_image, hole_filling_image])
            #
            # cv2.imshow("depth", row_1)


            depth_image_list = depth_image_process(depth_image)

            row1 = cv2.hconcat([color_image, depth_image_list[0][0]])

            row2 = cv2.hconcat([depth2color(depth_image_list[1][0]),
                                depth2color(depth_image_list[1][1]),
                                depth2color(depth_image_list[1][2])])
            row3 = cv2.hconcat([depth2color(depth_image_list[1][3]),
                                depth2color(depth_image_list[1][4]),
                                depth2color(depth_image_list[1][5])])

            combined_image = cv2.vconcat([row2, row3])
            cv2.imshow("color & depth", row1)
            cv2.imshow("depth_process", combined_image)

            key = cv2.waitKey(1)
            if key & 0xFF == ord('q') or key == 27:  # Press esc or 'q' to close the image window
                cv2.destroyAllWindows()
                break
            elif key & 0xFF == ord('w'):  # Increase min_clipping_distance
                alpha += 0.01
                print(f"alpha: {alpha}")
            elif key & 0xFF == ord('s'):  # Decrease min_clipping_distance
                alpha -= 0.01
                print(f"alpha: {alpha}")

    finally:
        # 停止流
        pipeline.stop()
        cv2.destroyAllWindows()


def main():
    cgy = "/Volumes/NO NAME/recorder/24_12_03/recording_5067_20241203.bag"
    me = "/Users/theobald/datasets/instance_seg_shrimp/intel_realsense/24_11_26/recording_2561_20241126.bag"
    file_path = me
    display(file_path)


if __name__ == "__main__":
    main()