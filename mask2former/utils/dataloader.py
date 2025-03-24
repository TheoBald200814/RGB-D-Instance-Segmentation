from functools import partial

import numpy as np
import torch
import os
import albumentations as A
from datasets import load_dataset, Image as IMG
from .data_process import get_label2id
from PIL import Image
from typing import Any, Dict, List, Mapping, Optional
from transformers.image_processing_utils import BatchFeature
from transformers import (
    AutoImageProcessor,
)


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
    dataset = dataset.cast_column("annotation", IMG())
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
    if args.rgb_d == 'multi':  # RGB-D(6 channel)
        dataset = dataset.cast_column("image", [IMG()])
        transform_rgbd = partial(
            rgbd_aug_and_trans, transform=train_augment_and_transform, image_processor=image_processor
        )
        dataset["train"] = dataset["train"].map(transform_rgbd)
        dataset["validation"] = dataset["validation"].map(transform_rgbd)
    elif args.rgb_d == 'ultra': # RGB-D(30 channel)
        dataset = dataset.cast_column("image", [IMG()])
        transform_rgbd_ultra = partial(
            rgbd_ultra_aug_and_trans, transform=train_augment_and_transform, image_processor=image_processor
        )
        dataset["train"] = dataset["train"].map(transform_rgbd_ultra)
        dataset["validation"] = dataset["validation"].map(transform_rgbd_ultra)
    else:  # RGB only(3 channel)
        transform_rgb = partial(
            rgb_aug_and_trans, transform=train_augment_and_transform, image_processor=image_processor
        )
        dataset = dataset.cast_column("image", IMG())
        dataset["train"] = dataset["train"].map(transform_rgb)
        dataset["validation"] = dataset["validation"].map(transform_rgb)

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
    size = (256, 256)
    example["image"] = example["image"].resize(size, Image.BILINEAR)
    example["annotation"] = example["annotation"].resize(size, Image.NEAREST)
    semantic_and_instance_masks = np.array(example["annotation"])[..., :2]
    image = np.array(example["image"])
    output = transform(image=image, mask=semantic_and_instance_masks)
    aug_image = output["image"]
    aug_semantic_and_instance_masks = output["mask"]
    aug_instance_mask = aug_semantic_and_instance_masks[..., 1]

    # Create mapping from instance id to semantic id
    unique_semantic_id_instance_id_pairs = np.unique(aug_semantic_and_instance_masks.reshape(-1, 2), axis=0)
    instance_id_to_semantic_id = {
        instance_id: semantic_id for semantic_id, instance_id in unique_semantic_id_instance_id_pairs
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
    size = (256, 256)
    assert len(example["image"]) >= 2, "the dataset not include multi-modal image, but the param of rgb_d in config.json was multi/ultra"
    example["image"] = [example["image"][0], example["image"][1].convert('RGB')]
    example["image"] = [image.resize(size, Image.BILINEAR) for image in example["image"]]
    # Resize annotation (use NEAREST to preserve label values)
    example["annotation"] = example["annotation"].resize(size, Image.NEAREST)

    image = np.array(example["image"])
    image = image.transpose(1, 2, 0, 3).reshape(image.shape[1], image.shape[2], -1)
    semantic_and_instance_masks = np.array(example["annotation"])[..., :2]
    output = transform(image=image, mask=semantic_and_instance_masks)
    aug_image = output["image"]
    aug_semantic_and_instance_masks = output["mask"]
    aug_instance_mask = aug_semantic_and_instance_masks[..., 1]

    # Create mapping from instance id to semantic id
    unique_semantic_id_instance_id_pairs = np.unique(aug_semantic_and_instance_masks.reshape(-1, 2), axis=0)
    instance_id_to_semantic_id = {
        instance_id: semantic_id for semantic_id, instance_id in unique_semantic_id_instance_id_pairs
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
    size = (256, 256)
    assert len(example["image"]) >= 2, "the dataset not include multi-modal image, but the param of rgb_d in config.json was multi/ultra"
    # example["image"] = [example["image"][0], example["image"][1].convert('RGB')]
    example["image"] = [i.convert('RGB') for i in example["image"]]
    example["image"] = [image.resize(size, Image.BILINEAR) for image in example["image"]]
    # Resize annotation (use NEAREST to preserve label values)
    example["annotation"] = example["annotation"].resize(size, Image.NEAREST)

    image = np.array(example["image"])
    image = image.transpose(1, 2, 0, 3).reshape(image.shape[1], image.shape[2], -1)
    semantic_and_instance_masks = np.array(example["annotation"])[..., :2]
    output = transform(image=image, mask=semantic_and_instance_masks)
    aug_image = output["image"]
    aug_semantic_and_instance_masks = output["mask"]
    aug_instance_mask = aug_semantic_and_instance_masks[..., 1]

    # Create mapping from instance id to semantic id
    unique_semantic_id_instance_id_pairs = np.unique(aug_semantic_and_instance_masks.reshape(-1, 2), axis=0)
    instance_id_to_semantic_id = {
        instance_id: semantic_id for semantic_id, instance_id in unique_semantic_id_instance_id_pairs
    }

    model_inputs = image_processor(
        images=[aug_image[..., :3], aug_image[..., 3:6], aug_image[..., 6:9], aug_image[..., 9:12], aug_image[..., 12:15],
                aug_image[..., 15:18], aug_image[..., 18:21], aug_image[..., 21:24], aug_image[..., 24:27], aug_image[..., 27:30]],
        segmentation_maps=[aug_instance_mask, aug_instance_mask, aug_instance_mask, aug_instance_mask, aug_instance_mask,
                           aug_instance_mask, aug_instance_mask, aug_instance_mask, aug_instance_mask, aug_instance_mask],
        instance_id_to_semantic_id=instance_id_to_semantic_id,
        return_tensors="pt",
    )

    image = model_inputs.pixel_values
    example["pixel_values"] = image.reshape(-1, image.shape[2], image.shape[3]).tolist()
    example["mask_labels"] = model_inputs.mask_labels[0].tolist()
    example["class_labels"] = model_inputs.class_labels[0]

    return example


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
    pixel_values_list = [example["pixel_values"] for example in examples]
    class_labels_list = [example["class_labels"] for example in examples]
    mask_labels_list = [example["mask_labels"] for example in examples]

    # 在 collate_fn 中进行 tensor 转换和 stack
    batch["pixel_values"] = torch.stack([torch.tensor(pv) for pv in pixel_values_list])
    batch["class_labels"] = [torch.tensor(cl) for cl in class_labels_list] # 可以选择 stack 或保持 list of tensors
    batch["mask_labels"] = [torch.tensor(ml) for ml in mask_labels_list] # 可以选择 stack 或保持 list of tensors

    if "pixel_mask" in examples[0]:
        batch["pixel_mask"] = torch.stack([example["pixel_mask"] for example in examples])
    return batch