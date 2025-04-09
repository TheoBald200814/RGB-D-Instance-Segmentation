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

from transformers.utils import check_min_version, send_example_telemetry
from transformers.utils.versions import require_version
from transformers import AutoImageProcessor, HfArgumentParser, Trainer, TrainingArguments
from utils.arguments import Arguments
from utils.dataloader import collate_fn_v2, dataloader
from utils.model_essential_part import find_last_checkpoint, Evaluator
from utils.log import setup_logging
from utils.custom_model import CustomMask2FormerForUniversalSegmentation

logger = logging.getLogger(__name__)

# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
check_min_version("4.47.0.dev0")
require_version("datasets>=2.0.0", "To fix: pip install -r examples/pytorch/instance-segmentation/requirements.txt")


def main():
    # See all possible arguments in https://huggingface.co/docs/transformers/main_classes/trainer#transformers.TrainingArguments
    # or by passing the --help flag to this script.

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

    # Load image processor
    image_processor = AutoImageProcessor.from_pretrained(
        args.model_name_or_path,
        do_resize=True,
        size={"height": args.image_height, "width": args.image_width},
        do_reduce_labels=args.do_reduce_labels,
        # reduce_labels=args.do_reduce_labels,  # TODO: remove when instance_seg support `do_reduce_labels`
        # token=args.token,
        ignore_index=args.ignore_index,
    )

    # Load dataset
    dataset, label2id, id2label = dataloader(args, image_processor)

    # Load pretrained config, model
    model = CustomMask2FormerForUniversalSegmentation.from_pretrained(
        args.model_name_or_path,
        label2id=label2id,
        id2label=id2label,
        ignore_mismatched_sizes=True,
        rgb_d=args.rgb_d,
        image_processor=image_processor
    )

    # Load evaluator
    compute_metrics = Evaluator(image_processor=image_processor, id2label=id2label, threshold=0.0)

    # Load trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"] if training_args.do_train else None,
        eval_dataset=dataset["validation"] if training_args.do_eval else None,
        processing_class=image_processor,
        data_collator=collate_fn_v2,
        compute_metrics=compute_metrics,
    )

    # ------------------------------------------------------------------------------------------------
    # Model training and evaluation with Trainer API
    # ------------------------------------------------------------------------------------------------
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