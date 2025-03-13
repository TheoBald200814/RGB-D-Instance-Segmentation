#!/usr/bin/env python
# coding=utf-8
# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and

"""Finetuning 🤗 Transformers model for instance segmentation leveraging the Trainer API."""

import logging
import os
import sys
from typing import Optional, List

import albumentations as A
import numpy as np
import torch
import PIL.Image
import random

from functools import partial

from torch import Tensor
from transformers.models.mask2former.modeling_mask2former import Mask2FormerModel, Mask2FormerPixelLevelModule, \
    Mask2FormerForUniversalSegmentationOutput

from mask2former.tools.data_process import get_label2id
from datasets import load_dataset, Image
from transformers.utils import check_min_version, send_example_telemetry
from transformers.utils.versions import require_version
from transformers import (
    AutoImageProcessor,
    HfArgumentParser,
    Trainer,
    TrainingArguments, Mask2FormerConfig,
)
from transformers import Mask2FormerForUniversalSegmentation

from mask2former.experiments.architecture.utils.arguments import Arguments
from mask2former.experiments.architecture.utils.augment_and_transform import augment_and_transform, augment_and_transform_batch, collate_fn
from mask2former.experiments.architecture.utils.model_essential_part import find_last_checkpoint, Evaluator
from mask2former.experiments.architecture.utils.log import setup_logging

logger = logging.getLogger(__name__)

# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
check_min_version("4.47.0.dev0")

require_version("datasets>=2.0.0", "To fix: pip install -r examples/pytorch/instance-segmentation/requirements.txt")


def resize_images(example, size=(256, 256)):
    # Resize image
    example["image"] = [image.resize(size, PIL.Image.BILINEAR) for image in example["image"]]
    # Resize annotation (use NEAREST to preserve label values)
    example["annotation"] = example["annotation"].resize(size, PIL.Image.NEAREST)
    return example


class CustomConfig(Mask2FormerConfig):
    model_type = "mask2former"

    def __init__(self, attribute=1, **kwargs):
        self.attribute = attribute
        super().__init__(**kwargs)


class CustomMask2FormerModel(Mask2FormerModel):
    main_input_name = "pixel_values"
    def __init__(self, config):
        print("在CustomMask2FormerModel的构造函数中执行super().__init__(config)")
        super().__init__(config)
        self.pixel_level_module = CustomMask2FormerPixelLevelModule(config)


class CustomMask2FormerPixelLevelModule(Mask2FormerPixelLevelModule):
    main_input_name = "pixel_values"
    def __init__(self, config):
        print("在CustomMask2FormerPixelLevelModule的构造函数中执行super().__init__(config)")
        super().__init__(config)


class CustomMask2FormerForUniversalSegmentation(Mask2FormerForUniversalSegmentation):
    main_input_name = "pixel_values"
    config_class = CustomConfig

    def __init__(self, config):
        super().__init__(config)
        set_seed(42)
        self.model = CustomMask2FormerModel(config)

    def forward(
            self,
            pixel_values: Tensor,
            mask_labels: Optional[List[Tensor]] = None,
            class_labels: Optional[List[Tensor]] = None,
            pixel_mask: Optional[Tensor] = None,
            output_hidden_states: Optional[bool] = None,
            output_auxiliary_logits: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            return_dict: Optional[bool] = None,
    ) -> Mask2FormerForUniversalSegmentationOutput:
        print("测试CustomMask2FormerForUniversalSegmentation")
        pixel_values = pixel_values[:, :3, :, :]
        return super().forward(pixel_values, mask_labels, class_labels, pixel_mask, output_hidden_states, output_auxiliary_logits, return_dict)

# 固定随机种子
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def main():
    # See all possible arguments in https://huggingface.co/docs/transformers/main_classes/trainer#transformers.TrainingArguments
    # or by passing the --help flag to this script.

    # set_seed(42) # 在main函数层面显式声明seed似乎不会对训练结果产生影响

    parser = HfArgumentParser([Arguments, TrainingArguments])
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        args, training_args = parser.parse_args_into_dataclasses()

    # Set default training arguments for instance segmentation
    training_args.eval_do_concat_batches = False
    training_args.batch_eval_metrics = True
    training_args.remove_unused_columns = False

    # # Sending telemetry. Tracking the example usage helps us better allocate resources to maintain them. The
    # # information sent is the one passed as arguments along with your Python/PyTorch versions.
    send_example_telemetry("run_instance_segmentation", args)

    # Setup logging and log on each process the small summary:
    setup_logging(training_args, logger)
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}, "
        + f"distributed training: {training_args.parallel_mode.value == 'distributed'}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")

    # Load last checkpoint from output_dir if it exists (and we are not overwriting it)
    checkpoint = find_last_checkpoint(training_args, logger)

    # ------------------------------------------------------------------------------------------------
    # Load dataset, prepare splits
    # ------------------------------------------------------------------------------------------------
    data_files = {
        "train": os.path.join(args.root_path, args.train_json_path),
        "validation": os.path.join(args.root_path, args.valid_json_path),
    }
    dataset = load_dataset("json", data_files=data_files)
    dataset = dataset.cast_column("image", [Image()])
    dataset = dataset.cast_column("annotation", Image())
    # image resize
    dataset["train"] = dataset["train"].map(resize_images)
    dataset["validation"] = dataset["validation"].map(resize_images)

    # We need to specify the label2id mapping for the model
    # it is a mapping from semantic class name to class index.
    label2id = get_label2id(os.path.join(args.root_path, args.label2id_path))

    if args.do_reduce_labels:
        label2id = {name: idx for name, idx in label2id.items() if idx != 0}  # remove background class
        label2id = {name: idx - 1 for name, idx in label2id.items()}  # shift class indices by -1

    id2label = {v: k for k, v in label2id.items()}

    # ------------------------------------------------------------------------------------------------
    # Load pretrained config, model and image processor
    # ------------------------------------------------------------------------------------------------
    # model = AutoModelForUniversalSegmentation.from_pretrained(
    #     args.model_name_or_path,
    #     label2id=label2id,
    #     id2label=id2label,
    #     ignore_mismatched_sizes=True
    # )
    model = CustomMask2FormerForUniversalSegmentation.from_pretrained(
        args.model_name_or_path,
        label2id=label2id,
        id2label=id2label,
        ignore_mismatched_sizes=True
    )

    image_processor = AutoImageProcessor.from_pretrained(
        args.model_name_or_path,
        do_resize=True,
        size={"height": args.image_height, "width": args.image_width},
        do_reduce_labels=args.do_reduce_labels,
        # reduce_labels=args.do_reduce_labels,  # TODO: remove when instance_seg support `do_reduce_labels`
        # token=args.token,
        ignore_index=args.ignore_index,
    )

    # ------------------------------------------------------------------------------------------------
    # Define image augmentations and dataset transforms
    # ------------------------------------------------------------------------------------------------
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

    transform_single = partial(
        augment_and_transform, transform=train_augment_and_transform, image_processor=image_processor
    )
    transform_batch = partial(
        augment_and_transform_batch, image_processor=image_processor
    )
    dataset["train"] = dataset["train"].map(transform_single)
    dataset["validation"] = dataset["validation"].map(transform_single)
    dataset["train"] = dataset["train"].with_transform(transform_batch)
    dataset["validation"] = dataset["validation"].with_transform(transform_batch)

    # ------------------------------------------------------------------------------------------------
    # Model training and evaluation with Trainer API
    # ------------------------------------------------------------------------------------------------

    compute_metrics = Evaluator(image_processor=image_processor, id2label=id2label, threshold=0.0)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"] if training_args.do_train else None,
        eval_dataset=dataset["validation"] if training_args.do_eval else None,
        processing_class=image_processor,
        data_collator=collate_fn,
        compute_metrics=compute_metrics,
    )

    # Training
    if training_args.do_train:
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        trainer.save_model()
        trainer.log_metrics("train", train_result.metrics)
        trainer.save_metrics("train", train_result.metrics)
        trainer.save_state()

    # Final evaluation
    if training_args.do_eval:
        metrics = trainer.evaluate(eval_dataset=dataset["validation"], metric_key_prefix="test")
        trainer.log_metrics("test", metrics)
        trainer.save_metrics("test", metrics)

    # Write model card and (optionally) push to hub
    kwargs = {
        "finetuned_from": args.model_name_or_path,
        "dataset": args.dataset_name,
        "tags": ["image-segmentation", "instance-segmentation", "vision"],
    }
    if training_args.push_to_hub:
        trainer.push_to_hub(**kwargs)
    else:
        trainer.create_model_card(**kwargs)


if __name__ == "__main__":
    main()
