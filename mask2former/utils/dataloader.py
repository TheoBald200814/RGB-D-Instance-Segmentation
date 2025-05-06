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
from mask2former.utils.data_process import cosine_similarity_fuse_v3, to_grayscale, csf_viewer_v2


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
    train_augment_and_transform = A.Compose(
        [
            # A.HorizontalFlip(p=0.5),
            # A.RandomBrightnessContrast(p=0.5),
            # A.HueSaturationValue(p=0.1),
            A.NoOp()
        ],
    )
    validation_transform = A.Compose(
        [A.NoOp()],
    )
    if args.version == '0.0.0': # Only 3 channel RGB as input
        transform_v0 = partial(
            rgb_aug_and_trans, transform=train_augment_and_transform, image_processor=image_processor
        )
        dataset = dataset.cast_column("image", IMG())
        dataset["train"] = dataset["train"].map(transform_v0, num_proc=4, writer_batch_size=50)
        dataset["validation"] = dataset["validation"].map(transform_v0, num_proc=4, writer_batch_size=50)

    elif args.version == '0.1.0' or args.version == '0.1.1': # 6 channel of RGB and Depth as input
        transform_v1 = partial(
            rgbd_aug_and_trans, transform=train_augment_and_transform, image_processor=image_processor
        )
        dataset = dataset.cast_column("image", [IMG()])
        dataset["train"] = dataset["train"].map(transform_v1, num_proc=4, writer_batch_size=50)
        dataset["validation"] = dataset["validation"].map(transform_v1, num_proc=4, writer_batch_size=50)

    elif args.version == '0.2.0': # 30 channel of RGB and expand Depth as input
        transform_v2 = partial(
            rgbd_ultra_aug_and_trans, transform=train_augment_and_transform, image_processor=image_processor
        )
        dataset = dataset.cast_column("image", [IMG()])
        dataset["train"] = dataset["train"].map(transform_v2, num_proc=4, writer_batch_size=50)
        dataset["validation"] = dataset["validation"].map(transform_v2, num_proc=4, writer_batch_size=50)

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