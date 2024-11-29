import numpy as np
import os
import cv2
from transformers import AutoImageProcessor, Mask2FormerForUniversalSegmentation
import torch
from PIL import Image


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


def main():
    image_path = "/Users/theobald/Documents/code_lib/python_lib/huggingface_data_process/datasets/local/360.jpg"
    model_path = "/Users/theobald/Documents/code_lib/python_lib/huggingface_data_process/checkpoint/local/checkpoint-868"
    predictor(image_path, model_path)


if __name__ == '__main__':
    main()