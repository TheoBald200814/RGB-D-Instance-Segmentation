import pyrealsense2 as rs
import cv2
import numpy as np
import time
import os
import random
from datetime import datetime


def init_file_name(save_dir):
    """
    初始化文件名: 使用日期 + 随机字符构成
    """
    random_number = random.randint(1000, 9999)
    today_date = datetime.now().strftime("%Y%m%d")
    file_name = os.path.join(save_dir, f"recording_{random_number}_{today_date}.bag")

    return file_name


def recorder(save_dir: str, interval: int):
    # 创建保存目录
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # 创建一个管道对象
    pipeline = rs.pipeline()

    # 配置流
    config = rs.config()
    config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)

    # 初始化变量
    start_time = time.time()
    file_index = 0

    try:
        # 生成第一个文件名并开始录制

        # file_name = os.path.join(save_dir, f"recording_{file_index}.bag")
        file_name = init_file_name(save_dir)
        config.enable_record_to_file(file_name)
        pipeline.start(config)

        print(f"Started recording to: {file_name}")

        while True:
            # 计算当前时间
            current_time = time.time()

            # 检查是否需要创建新的录制文件
            if current_time - start_time >= interval:
                # 停止当前录制
                pipeline.stop()
                print("pipeline stop success")

                # 生成新的文件名
                file_index += 1
                # file_name = os.path.join(save_dir, f"recording_{file_index}.bag")
                file_name = init_file_name(save_dir)
                # 重新配置录制
                config.enable_record_to_file(file_name)
                pipeline.start(config)
                print(f"Started recording to: {file_name}")

                # 更新开始时间
                start_time = current_time

            # 等待一帧数据
            frames = pipeline.wait_for_frames(timeout_ms=10000)
            depth_frame = frames.get_depth_frame()
            color_frame = frames.get_color_frame()

            if not depth_frame or not color_frame:
                continue

            # 将帧转换为numpy数组
            depth_image = np.asanyarray(depth_frame.get_data())
            color_image = np.asanyarray(color_frame.get_data())

            # 将深度图像转换为伪彩色图像
            depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.1), cv2.COLORMAP_JET)

            # 获取深度和彩色图像的尺寸
            depth_colormap_dim = depth_colormap.shape
            color_colormap_dim = color_image.shape

            # 如果深度和彩色图像的分辨率不同，调整彩色图像的大小以匹配深度图像
            if depth_colormap_dim != color_colormap_dim:
                resized_color_image = cv2.resize(color_image, dsize=(depth_colormap_dim[1], depth_colormap_dim[0]),
                                                 interpolation=cv2.INTER_AREA)
                images = np.hstack((resized_color_image, depth_colormap))
            else:
                images = np.hstack((color_image, depth_colormap))

            # 显示图像
            cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
            cv2.imshow('RealSense', images)

            # 按 'q' 键退出
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        # 停止流
        pipeline.stop()
        # cv2.destroyAllWindows()


def main():
    record_dir =    me = "/Volumes/ESD-USB/24_12_03"
    recorder(record_dir, 30)


if __name__ == '__main__':
    main()