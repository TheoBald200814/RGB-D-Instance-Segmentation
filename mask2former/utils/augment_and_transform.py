import numpy as np
import torch
from PIL import Image
from typing import Any, Dict, List, Mapping, Optional
from transformers.image_processing_utils import BatchFeature
from transformers import (
    AutoImageProcessor,
)


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


def augment_and_transform(example, transform, image_processor):
    # Resize image
    size = (256, 256)
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
    # example["pixel_values"] = model_inputs.pixel_values[0].tolist()
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