from functools import partial

import cv2
import numpy as np
import torch
import os
import albumentations as A
from datasets import load_dataset, Image as IMG, Value
from .data_process import get_label2id
from PIL import Image
from typing import Any, Dict, List, Mapping, Optional
from transformers.image_processing_utils import BatchFeature
from transformers import (
    AutoImageProcessor,
)
from mask2former.utils.data_process import cosine_similarity_fuse_v3, to_grayscale, compute_depth_gradient, compute_surface_normals, calculate_gradient_features


no_augment_and_transform = A.Compose([A.NoOp()],)

# Input: RGB(3 channel) = 3 channel
# Output: RGB(3 channel) = 3 channel
def map_3channel(example, transform, image_processor):
    mask = cv2.imread(example["annotation"], cv2.IMREAD_UNCHANGED)
    semantic_and_instance_masks = mask[..., 1:]
    image = np.array(example["image"])
    output = transform(image=image, mask=semantic_and_instance_masks)
    aug_image = output["image"]
    aug_semantic_and_instance_masks = output["mask"]
    aug_instance_mask = aug_semantic_and_instance_masks[..., 0]

    # Create mapping from instance id to semantic id
    unique_instance_id_semantic_id_pairs = np.unique(aug_semantic_and_instance_masks.reshape(-1, 2), axis=0)
    instance_id_to_semantic_id = {
        instance_id: semantic_id for instance_id, semantic_id in unique_instance_id_semantic_id_pairs
    }

    model_inputs = image_processor(
        images=[aug_image],
        segmentation_maps=[aug_instance_mask],
        instance_id_to_semantic_id=instance_id_to_semantic_id,
        return_tensors="pt",
    )

    example["pixel_values"] = model_inputs.pixel_values[0].tolist()
    example["mask_labels"] = model_inputs.mask_labels[0].tolist()
    example["class_labels"] = model_inputs.class_labels[0]

    return example

# Input: RGB(3 channel) + Any-Depth(3 channel) = 6 channel
# Output: RGB(3 channel) + Depth/3Gradient-Depth/Surface-Normal-Depth(3 channel) = 6 channel
def map_6channel(example, transform, image_processor):
    assert len(example["image"]) >= 2, "the dataset not include multi-modal image"
    example["image"] = [example["image"][0], example["image"][1].convert('RGB')]

    mask = cv2.imread(example["annotation"], cv2.IMREAD_UNCHANGED)
    semantic_and_instance_masks = mask[..., 1:]
    image = np.array(example["image"])
    image = image.transpose(1, 2, 0, 3).reshape(image.shape[1], image.shape[2], -1)
    output = transform(image=image, mask=semantic_and_instance_masks)
    aug_image = output["image"]
    aug_semantic_and_instance_masks = output["mask"]
    aug_instance_mask = aug_semantic_and_instance_masks[..., 0]

    # Create mapping from instance id to semantic id
    unique_instance_id_semantic_id_pairs = np.unique(aug_semantic_and_instance_masks.reshape(-1, 2), axis=0)
    instance_id_to_semantic_id = {
        instance_id: semantic_id for instance_id, semantic_id in unique_instance_id_semantic_id_pairs
    }

    model_inputs = image_processor(
        images=[aug_image[..., :3], aug_image[..., 3:6]],
        segmentation_maps=[aug_instance_mask, aug_instance_mask],
        instance_id_to_semantic_id=instance_id_to_semantic_id,
        return_tensors="pt",
    )

    image = model_inputs.pixel_values
    example["pixel_values"] = image.reshape(-1, image.shape[2], image.shape[3]).tolist()
    example["mask_labels"] = model_inputs.mask_labels[0].tolist()
    example["class_labels"] = model_inputs.class_labels[0]

    return example

# Input: RGB(3 channel) + Augmentation-Depth(27 channel) = 30 channel
# TODO: Verify the Output channel
def map_30channel(example, transform, image_processor):
    assert len(example["image"]) >= 2, "the dataset not include multi-modal image"
    example["image"] = [i.convert('RGB') for i in example["image"]]

    image = np.array(example["image"])
    image = image.transpose(1, 2, 0, 3).reshape(image.shape[1], image.shape[2], -1)
    mask = cv2.imread(example["annotation"], cv2.IMREAD_UNCHANGED)
    semantic_and_instance_masks = mask[..., 1:]
    output = transform(image=image, mask=semantic_and_instance_masks)
    aug_image = output["image"]
    aug_semantic_and_instance_masks = output["mask"]
    aug_instance_mask = aug_semantic_and_instance_masks[..., 0]

    # Create mapping from instance id to semantic id
    unique_instance_id_semantic_id_pairs = np.unique(aug_semantic_and_instance_masks.reshape(-1, 2), axis=0)
    instance_id_to_semantic_id = {
        instance_id: semantic_id for instance_id, semantic_id in unique_instance_id_semantic_id_pairs
    }

    # Insert ICSFer
    # aug_fused_img_1, aug_fused_img_2, depth_input = rgbd_ultra_preprocess(aug_image)
    aug_fused_img, depth_input = nyu_ultra_preprocess(aug_image)
    color = image[..., 0:3]
    ahe = image[..., 15:18]
    laplace = image[..., 18:21]
    gaussian = image[..., 21:24]

    model_inputs = image_processor(
        images=[color, aug_fused_img, depth_input],
        segmentation_maps=[aug_instance_mask, aug_instance_mask, aug_instance_mask],
        instance_id_to_semantic_id=instance_id_to_semantic_id,
        return_tensors="pt",
    )

    image = model_inputs.pixel_values

    # example["pixel_values"] : [aug_fused_img_1, aug_fused_img_2, depth_input, ahe, laplace, gaussian]
    example["pixel_values"] = image.reshape(-1, image.shape[2], image.shape[3]).tolist()
    example["mask_labels"] = model_inputs.mask_labels[0].tolist()
    example["class_labels"] = model_inputs.class_labels[0]

    return example

# Input: RGB(3 channel) + Depth(3 channel) = 6 channel
# Output: RGB(3 channel) + Gradient-Depth(3 channel) + Gradient-Depth-Mask(1 channel) = 7 channel
def map_7channel_g(example, transform, image_processor):
    assert len(example["image"]) >= 2, "the dataset not include multi-modal image"
    color = np.array(example["image"][0]) # (H, W, 3)
    depth = np.array(example["image"][1].convert('L')) # (H, W)
    mask = cv2.imread(example["annotation"], cv2.IMREAD_UNCHANGED) # (H, W, 3)

    semantic_and_instance_masks = mask[..., 1:]
    output = transform(image=color, mask=semantic_and_instance_masks)
    aug_color = output["image"]
    aug_semantic_and_instance_masks = output["mask"]
    aug_instance_mask = aug_semantic_and_instance_masks[..., 0]

    # Create mapping from instance id to semantic id
    unique_instance_id_semantic_id_pairs = np.unique(aug_semantic_and_instance_masks.reshape(-1, 2), axis=0)
    instance_id_to_semantic_id = {
        instance_id: semantic_id for instance_id, semantic_id in unique_instance_id_semantic_id_pairs
    }
    gradient_depth = compute_depth_gradient(depth).astype(np.uint8)
    colorful_gradient_depth = np.stack([gradient_depth, gradient_depth, gradient_depth], axis=2)

    model_inputs = image_processor(
        images=[aug_color, colorful_gradient_depth],
        segmentation_maps=[aug_instance_mask, aug_instance_mask],
        instance_id_to_semantic_id=instance_id_to_semantic_id,
        return_tensors="pt",
    )

    h = model_inputs.pixel_values.shape[2]
    w = model_inputs.pixel_values.shape[3]
    resized_depth = cv2.resize(colorful_gradient_depth, (h, w), interpolation=cv2.INTER_LINEAR)
    gradient_mask = np.any(resized_depth > 50, axis=-1).tolist()

    image = model_inputs.pixel_values
    example["pixel_values"] = image.reshape(-1, image.shape[2], image.shape[3]).tolist()
    example["pixel_values"].append(gradient_mask)
    example["mask_labels"] = model_inputs.mask_labels[0].tolist()
    example["class_labels"] = model_inputs.class_labels[0]

    return example

# Input: RGB(3 channel) + Depth(3 channel) = 6 channel
# Output: RGB(3 channel) + Gradient-Depth(3 channel) + Gradient-Depth-Mask(1 channel) = 7 channel
def map_7channel_g2(example, transform, image_processor):
    assert len(example["image"]) >= 2, "the dataset not include multi-modal image"
    color = np.array(example["image"][0])  # (H, W, 3)
    depth = np.array(example["image"][1].convert('L'))  # (H, W)
    mask = cv2.imread(example["annotation"], cv2.IMREAD_UNCHANGED)  # (H, W, 3)

    semantic_and_instance_masks = mask[..., 1:]
    output = transform(image=color, mask=semantic_and_instance_masks)
    aug_color = output["image"]
    aug_semantic_and_instance_masks = output["mask"]
    aug_instance_mask = aug_semantic_and_instance_masks[..., 0]

    # Create mapping from instance id to semantic id
    unique_instance_id_semantic_id_pairs = np.unique(aug_semantic_and_instance_masks.reshape(-1, 2), axis=0)
    instance_id_to_semantic_id = {
        instance_id: semantic_id for instance_id, semantic_id in unique_instance_id_semantic_id_pairs
    }

    model_inputs = image_processor(
        images=[aug_color],
        segmentation_maps=[aug_instance_mask],
        instance_id_to_semantic_id=instance_id_to_semantic_id,
        return_tensors="pt",
    )

    h = model_inputs.pixel_values.shape[2]
    w = model_inputs.pixel_values.shape[3]
    resized_depth = cv2.resize(depth, (h, w), interpolation=cv2.INTER_LINEAR)
    normalized_magnitude, grad_x, grad_y, valid_gradient_mask = calculate_gradient_features(resized_depth)
    colorful_gradient_depth = np.stack([normalized_magnitude, normalized_magnitude, normalized_magnitude], axis=0).tolist()

    image = model_inputs.pixel_values
    example["pixel_values"] = image.reshape(-1, image.shape[2], image.shape[3]).tolist()
    example["pixel_values"] += colorful_gradient_depth
    example["pixel_values"].append(valid_gradient_mask.tolist())
    example["mask_labels"] = model_inputs.mask_labels[0].tolist()
    example["class_labels"] = model_inputs.class_labels[0]

    return example

# Input: RGB(3 channel) + 3Gradient-Depth(3 channel) = 6 channel
# Output: RGB(3 channel) + 3Gradient-Depth(3 channel) + 3Gradient-Depth-Mask(1 channel) = 7 channel
def map_7channel_tmp(example, transform, image_processor):
    assert len(example["image"]) >= 2, "the dataset not include multi-modal image"
    example["image"] = [example["image"][0], example["image"][1].convert('RGB')]

    mask = cv2.imread(example["annotation"], cv2.IMREAD_UNCHANGED)
    semantic_and_instance_masks = mask[..., 1:]
    image = np.array(example["image"])
    image = image.transpose(1, 2, 0, 3).reshape(image.shape[1], image.shape[2], -1)
    output = transform(image=image, mask=semantic_and_instance_masks)
    aug_image = output["image"]
    aug_semantic_and_instance_masks = output["mask"]
    aug_instance_mask = aug_semantic_and_instance_masks[..., 0]

    # Create mapping from instance id to semantic id
    unique_instance_id_semantic_id_pairs = np.unique(aug_semantic_and_instance_masks.reshape(-1, 2), axis=0)
    instance_id_to_semantic_id = {
        instance_id: semantic_id for instance_id, semantic_id in unique_instance_id_semantic_id_pairs
    }

    model_inputs = image_processor(
        images=[aug_image[..., :3], aug_image[..., 3:6]],
        segmentation_maps=[aug_instance_mask, aug_instance_mask],
        instance_id_to_semantic_id=instance_id_to_semantic_id,
        return_tensors="pt",
    )

    h = model_inputs.pixel_values.shape[2]
    w = model_inputs.pixel_values.shape[3]
    resized_depth = cv2.resize(aug_image[..., 3:6], (h, w), interpolation=cv2.INTER_LINEAR)
    gradient_mask = np.any(resized_depth > 50, axis=-1).tolist()

    image = model_inputs.pixel_values
    example["pixel_values"] = image.reshape(-1, image.shape[2], image.shape[3]).tolist()
    example["pixel_values"].append(gradient_mask)
    example["mask_labels"] = model_inputs.mask_labels[0].tolist()
    example["class_labels"] = model_inputs.class_labels[0]

    return example

# Input: RGB(3 channel) + Surface-Normal-Depth(3 channel) = 6 channel
# Output: RGB(3 channel) + Surface-Normal-Depth(3 channel) + Surface-Normal-Depth-Mask(1 channel) = 7 channel
def map_7channel_s(example, transform, image_processor):
    pass

# Input: RGB(3 channel) + Depth(3 channel) = 6 channel
# Output: RGB(3 channel) + Gradient-Depth(1 channel) + Surface-Normal-Depth(3 channel) + Gradient-Depth-Mask(1 channel) + Surface-Normal-Depth-Mask(1 channel) = 9 channel
def map_9channel(example, transform, image_processor):
    pass





register = {
    "0.0.0": {
        "map": map_3channel,
        "trans": no_augment_and_transform,
        "feature": IMG(),
        "num_proc": 4,
        "writer_batch_size": 50
    },
    "0.0.1": {
        "map": map_6channel,
        "trans": no_augment_and_transform,
        "feature": [IMG()],
        "num_proc": 4,
        "writer_batch_size": 50
    },
    "0.0.2": {
        "map": map_7channel_tmp,
        "trans": no_augment_and_transform,
        "feature": [IMG()],
        "num_proc": 4,
        "writer_batch_size": 50
    },
    "0.0.3": {
        "map": map_7channel_tmp,
        "trans": no_augment_and_transform,
        "feature": [IMG()],
        "num_proc": 4,
        "writer_batch_size": 50
    },
    "0.0.4": {
        "map": map_7channel_g,
        "trans": no_augment_and_transform,
        "feature": [IMG()],
        "num_proc": 1,
        "writer_batch_size": 50
    },
    "0.0.5": {
        "map": map_7channel_g2,
        "trans": no_augment_and_transform,
        "feature": [IMG()],
        "num_proc": 1,
        "writer_batch_size": 50
    },
    "0.1.0": {
        "map": map_6channel,
        "trans": no_augment_and_transform,
        "feature": [IMG()],
        "num_proc": 4,
        "writer_batch_size": 50
    },
    "0.1.1": {
        "map": map_6channel,
        "trans": no_augment_and_transform,
        "feature": [IMG()],
        "num_proc": 4,
        "writer_batch_size": 50
    },
    "0.1.2": {
        "map": map_6channel,
        "trans": no_augment_and_transform,
        "feature": [IMG()],
        "num_proc": 4,
        "writer_batch_size": 50
    },
    "0.1.3": {
        "map": map_6channel,
        "trans": no_augment_and_transform,
        "feature": [IMG()],
        "num_proc": 4,
        "writer_batch_size": 50
    },
    "0.2.0": {
        "map": map_30channel,
        "trans": no_augment_and_transform,
        "feature": [IMG()],
        "num_proc": 4,
        "writer_batch_size": 50
    }
}


def dataloader(args, image_processor):
    label2id = get_label2id(os.path.join(args.root_path, args.label2id_path))
    if args.do_reduce_labels:
        label2id = {name: idx for name, idx in label2id.items() if idx != 0}  # remove background class
        label2id = {name: idx - 1 for name, idx in label2id.items()}  # shift class indices by -1
    id2label = {v: k for k, v in label2id.items()}

    data_files = {
        "train": os.path.join(args.root_path, args.train_json_path),
        "validation": os.path.join(args.root_path, args.valid_json_path),
    }
    dataset = load_dataset("json", data_files=data_files)
    dataset = dataset.cast_column("annotation", Value("string"))

    loading_info = register[args.version] # 根据版本号加载注册信息
    map_func = partial(
        loading_info["map"], transform=loading_info["trans"], image_processor=image_processor
    )
    dataset = dataset.cast_column("image", loading_info["feature"])
    dataset["train"] = dataset["train"].map(map_func, num_proc=loading_info["num_proc"], writer_batch_size=loading_info["writer_batch_size"])
    dataset["validation"] = dataset["validation"].map(map_func, num_proc=loading_info["num_proc"], writer_batch_size=loading_info["writer_batch_size"])

    dataset["train"].set_format(type="torch")
    dataset["validation"].set_format(type="torch")

    return dataset, label2id, id2label


def augment_and_transform_batch(
    examples: Mapping[str, Any], image_processor: AutoImageProcessor
) -> BatchFeature:
    batch = {
        "pixel_values": [],
        "mask_labels": [],
        "class_labels": [],
    }
    for pixel_values, mask_labels, class_labels in zip(examples["pixel_values"], examples["mask_labels"], examples["class_labels"]):
        pixel_values = torch.tensor(pixel_values)
        mask_labels = torch.tensor(mask_labels)
        class_labels = torch.tensor(class_labels)


        batch["pixel_values"].append(pixel_values)
        batch["mask_labels"].append(mask_labels)
        batch["class_labels"].append(class_labels)

    return batch

def rgb_aug_and_trans(example, transform, image_processor):
    # Resize image
    # size = (256, 256)
    # example["image"] = example["image"].resize(size, Image.BILINEAR)
    # example["annotation"] = example["annotation"].resize(size, Image.NEAREST)

    mask = cv2.imread(example["annotation"], cv2.IMREAD_UNCHANGED)
    semantic_and_instance_masks = mask[..., 1:]
    image = np.array(example["image"])
    output = transform(image=image, mask=semantic_and_instance_masks)
    aug_image = output["image"]
    aug_semantic_and_instance_masks = output["mask"]
    aug_instance_mask = aug_semantic_and_instance_masks[..., 0]
    aug_semantic_mask = aug_semantic_and_instance_masks[..., 1]
    # in1 = np.unique(aug_instance_mask)
    # se1 = np.unique(aug_semantic_mask)

    # Create mapping from instance id to semantic id
    unique_instance_id_semantic_id_pairs = np.unique(aug_semantic_and_instance_masks.reshape(-1, 2), axis=0)
    instance_id_to_semantic_id = {
        instance_id: semantic_id for instance_id, semantic_id in unique_instance_id_semantic_id_pairs
    }

    model_inputs = image_processor(
        images=[aug_image],
        segmentation_maps=[aug_instance_mask],
        instance_id_to_semantic_id=instance_id_to_semantic_id,
        return_tensors="pt",
    )

    example["pixel_values"] = model_inputs.pixel_values[0].tolist()
    example["mask_labels"] = model_inputs.mask_labels[0].tolist()
    example["class_labels"] = model_inputs.class_labels[0]

    return example


def rgbd_aug_and_trans(example, transform, image_processor):
    # Resize image
    # size = (256, 256)
    assert len(example["image"]) >= 2, "the dataset not include multi-modal image, but the param of rgb_d in config.json was multi/ultra"
    example["image"] = [example["image"][0], example["image"][1].convert('RGB')]
    # example["image"] = [image.resize(size, Image.BILINEAR) for image in example["image"]]
    # Resize annotation (use NEAREST to preserve label values)
    # example["annotation"] = example["annotation"].resize(size, Image.NEAREST)

    mask = cv2.imread(example["annotation"], cv2.IMREAD_UNCHANGED)
    semantic_and_instance_masks = mask[..., 1:]
    image = np.array(example["image"])
    image = image.transpose(1, 2, 0, 3).reshape(image.shape[1], image.shape[2], -1)
    # semantic_and_instance_masks = np.array(example["annotation"])[..., :2]
    output = transform(image=image, mask=semantic_and_instance_masks)
    aug_image = output["image"]
    aug_semantic_and_instance_masks = output["mask"]
    aug_instance_mask = aug_semantic_and_instance_masks[..., 0]
    aug_semantic_mask = aug_semantic_and_instance_masks[..., 1]

    # Create mapping from instance id to semantic id
    unique_instance_id_semantic_id_pairs = np.unique(aug_semantic_and_instance_masks.reshape(-1, 2), axis=0)
    instance_id_to_semantic_id = {
        instance_id: semantic_id for instance_id, semantic_id in unique_instance_id_semantic_id_pairs
    }

    model_inputs = image_processor(
        images=[aug_image[..., :3], aug_image[..., 3:6]],
        segmentation_maps=[aug_instance_mask, aug_instance_mask],
        instance_id_to_semantic_id=instance_id_to_semantic_id,
        return_tensors="pt",
    )

    image = model_inputs.pixel_values
    example["pixel_values"] = image.reshape(-1, image.shape[2], image.shape[3]).tolist()
    example["mask_labels"] = model_inputs.mask_labels[0].tolist()
    example["class_labels"] = model_inputs.class_labels[0]

    return example


def rgbd_ultra_aug_and_trans(example, transform, image_processor):
    # Resize image
    # size = (256, 256)
    assert len(example["image"]) >= 2, "the dataset not include multi-modal image, but the param of rgb_d in config.json was multi/ultra"
    # example["image"] = [example["image"][0], example["image"][1].convert('RGB')]
    example["image"] = [i.convert('RGB') for i in example["image"]]
    # example["image"] = [image.resize(size, Image.BILINEAR) for image in example["image"]]
    # Resize annotation (use NEAREST to preserve label values)
    # example["annotation"] = example["annotation"].resize(size, Image.NEAREST)

    image = np.array(example["image"])
    image = image.transpose(1, 2, 0, 3).reshape(image.shape[1], image.shape[2], -1)
    mask = cv2.imread(example["annotation"], cv2.IMREAD_UNCHANGED)
    semantic_and_instance_masks = mask[..., 1:]
    output = transform(image=image, mask=semantic_and_instance_masks)
    aug_image = output["image"]
    aug_semantic_and_instance_masks = output["mask"]
    aug_instance_mask = aug_semantic_and_instance_masks[..., 0]
    aug_semantic_mask = aug_semantic_and_instance_masks[..., 1]

    # Create mapping from instance id to semantic id
    unique_instance_id_semantic_id_pairs = np.unique(aug_semantic_and_instance_masks.reshape(-1, 2), axis=0)
    instance_id_to_semantic_id = {
        instance_id: semantic_id for instance_id, semantic_id in unique_instance_id_semantic_id_pairs
    }

    # Insert ICSFer
    # aug_fused_img_1, aug_fused_img_2, depth_input = rgbd_ultra_preprocess(aug_image)
    aug_fused_img, depth_input = nyu_ultra_preprocess(aug_image)
    color = image[..., 0:3]
    ahe = image[..., 15:18]
    laplace = image[..., 18:21]
    gaussian = image[..., 21:24]

    model_inputs = image_processor(
        images=[color, aug_fused_img, depth_input],
        segmentation_maps=[aug_instance_mask, aug_instance_mask, aug_instance_mask],
        instance_id_to_semantic_id=instance_id_to_semantic_id,
        return_tensors="pt",
    )

    image = model_inputs.pixel_values

    # example["pixel_values"] : [aug_fused_img_1, aug_fused_img_2, depth_input, ahe, laplace, gaussian]
    example["pixel_values"] = image.reshape(-1, image.shape[2], image.shape[3]).tolist()
    example["mask_labels"] = model_inputs.mask_labels[0].tolist()
    example["class_labels"] = model_inputs.class_labels[0]

    return example


def rgbd_ultra_preprocess(image):
    # (batch, 30, H, W) [RGB, DECIMATION, RS, SPATIAL, HOLE_FILLING, AHE, LAPLACE, GAUSSIAN, EQ, LT]

    decimation = image[..., 3:6]
    rs = image[..., 6:9]
    spatial = image[..., 9:12]
    hole_filling = image[..., 12:15]
    ahe = image[..., 15:18]
    laplace = image[..., 18:21]
    gaussian = image[..., 21:24]
    eq = image[..., 24:27]
    lt = image[..., 27:30]

    # ICSFer
    fused_img_1 = cosine_similarity_fuse_v3([ahe, laplace, gaussian], check=None)
    fused_img_2 = cosine_similarity_fuse_v3([decimation, rs, spatial, hole_filling], check=None)

    # depth backbone
    eq = to_grayscale(eq)
    lt = to_grayscale(lt)

    fused_img_gray = to_grayscale(fused_img_1)
    depth_input = np.stack([eq, lt, fused_img_gray], axis=2)

    return fused_img_1, fused_img_2, depth_input

def nyu_ultra_preprocess(image):
    # (batch, 30, H, W) [RGB, DEPTH, AUG1, AUG2, AUG3, AUG4, AUG5, AUG6, AUG7, AUG8]

    depth = image[..., 3:6]
    aug1 = image[..., 6:9]
    aug2 = image[..., 9:12]
    aug3 = image[..., 12:15]
    aug4 = image[..., 15:18]
    aug5 = image[..., 18:21]
    aug6 = image[..., 21:24]
    aug7 = image[..., 24:27]
    aug8 = image[..., 27:30]

    # ICSFer
    fused_img = cosine_similarity_fuse_v3([aug1, aug2, aug3, aug4, aug5, aug6, aug7, aug8], check=None)

    return fused_img, depth


def collate_fn(examples):
    batch = {}
    batch["pixel_values"] = torch.stack([example["pixel_values"] for example in examples])
    batch["class_labels"] = [example["class_labels"] for example in examples]
    batch["mask_labels"] = [example["mask_labels"] for example in examples]
    if "pixel_mask" in examples[0]:
        batch["pixel_mask"] = torch.stack([example["pixel_mask"] for example in examples])
    return batch


def collate_fn_v2(examples): # 修改后的 collate_fn
    batch = {}
    batch["pixel_values"] = torch.stack([example["pixel_values"] for example in examples])
    batch["class_labels"] = [example["class_labels"] for example in examples]
    batch["mask_labels"] = [example["mask_labels"] for example in examples] # 可以选择 stack 或保持 list of tensors

    if "pixel_mask" in examples[0]:
        batch["pixel_mask"] = torch.stack([example["pixel_mask"] for example in examples])

    return batch