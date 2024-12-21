import os
import torch
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from transformers.trainer import EvalPrediction
from transformers.trainer_utils import get_last_checkpoint
from transformers import (
    AutoImageProcessor,
    TrainingArguments
)


@dataclass
class ModelOutput:
    class_queries_logits: torch.Tensor
    masks_queries_logits: torch.Tensor


def nested_cpu(tensors):
    if isinstance(tensors, (list, tuple)):
        return type(tensors)(nested_cpu(t) for t in tensors)
    elif isinstance(tensors, Mapping):
        return type(tensors)({k: nested_cpu(t) for k, t in tensors.items()})
    elif isinstance(tensors, torch.Tensor):
        return tensors.cpu().detach()
    else:
        return tensors


class Evaluator:
    """
    Compute metrics for the instance segmentation task.
    """

    def __init__(
        self,
        image_processor: AutoImageProcessor,
        id2label: Mapping[int, str],
        threshold: float = 0.0,
    ):
        """
        Initialize evaluator with image processor, id2label mapping and threshold for filtering predictions.

        Args:
            image_processor (AutoImageProcessor): Image processor for
                `post_process_instance_segmentation` method.
            id2label (Mapping[int, str]): Mapping from class id to class name.
            threshold (float): Threshold to filter predicted boxes by confidence. Defaults to 0.0.
        """
        self.image_processor = image_processor
        self.id2label = id2label
        self.threshold = threshold
        self.metric = self.get_metric()

    def get_metric(self):
        metric = MeanAveragePrecision(iou_type="segm", class_metrics=True)
        return metric

    def reset_metric(self):
        self.metric.reset()

    def postprocess_target_batch(self, target_batch) -> List[Dict[str, torch.Tensor]]:
        """Collect targets in a form of list of dictionaries with keys "masks", "labels"."""
        batch_masks = target_batch[0]
        batch_labels = target_batch[1]
        post_processed_targets = []
        for masks, labels in zip(batch_masks, batch_labels):
            post_processed_targets.append(
                {
                    "masks": masks.to(dtype=torch.bool),
                    "labels": labels,
                }
            )
        return post_processed_targets

    def get_target_sizes(self, post_processed_targets) -> List[List[int]]:
        target_sizes = []
        for target in post_processed_targets:
            target_sizes.append(target["masks"].shape[-2:])
        return target_sizes

    def postprocess_prediction_batch(self, prediction_batch, target_sizes) -> List[Dict[str, torch.Tensor]]:
        """Collect predictions in a form of list of dictionaries with keys "masks", "labels", "scores"."""

        model_output = ModelOutput(class_queries_logits=prediction_batch[0], masks_queries_logits=prediction_batch[1])
        post_processed_output = self.image_processor.post_process_instance_segmentation(
            model_output,
            threshold=self.threshold,
            target_sizes=target_sizes,
            return_binary_maps=True,
        )

        post_processed_predictions = []
        for image_predictions, target_size in zip(post_processed_output, target_sizes):
            if image_predictions["segments_info"]:
                post_processed_image_prediction = {
                    "masks": image_predictions["segmentation"].to(dtype=torch.bool),
                    "labels": torch.tensor([x["label_id"] for x in image_predictions["segments_info"]]),
                    "scores": torch.tensor([x["score"] for x in image_predictions["segments_info"]]),
                }
            else:
                # for void predictions, we need to provide empty tensors
                post_processed_image_prediction = {
                    "masks": torch.zeros([0, *target_size], dtype=torch.bool),
                    "labels": torch.tensor([]),
                    "scores": torch.tensor([]),
                }
            post_processed_predictions.append(post_processed_image_prediction)

        return post_processed_predictions

    @torch.no_grad()
    def __call__(self, evaluation_results: EvalPrediction, compute_result: bool = False) -> Mapping[str, float]:
        """
        Update metrics with current evaluation results and return metrics if `compute_result` is True.

        Args:
            evaluation_results (EvalPrediction): Predictions and targets from evaluation.
            compute_result (bool): Whether to compute and return metrics.

        Returns:
            Mapping[str, float]: Metrics in a form of dictionary {<metric_name>: <metric_value>}
        """
        prediction_batch = nested_cpu(evaluation_results.predictions)
        target_batch = nested_cpu(evaluation_results.label_ids)

        # For metric computation we need to provide:
        #  - targets in a form of list of dictionaries with keys "masks", "labels"
        #  - predictions in a form of list of dictionaries with keys "masks", "labels", "scores"
        post_processed_targets = self.postprocess_target_batch(target_batch)
        target_sizes = self.get_target_sizes(post_processed_targets)
        post_processed_predictions = self.postprocess_prediction_batch(prediction_batch, target_sizes)

        # Compute metrics
        self.metric.update(post_processed_predictions, post_processed_targets)

        if not compute_result:
            return

        metrics = self.metric.compute()

        # Replace list of per class metrics with separate metric for each class
        classes = metrics.pop("classes")
        map_per_class = metrics.pop("map_per_class")
        mar_100_per_class = metrics.pop("mar_100_per_class")
        for class_id, class_map, class_mar in zip(classes, map_per_class, mar_100_per_class):
            class_name = self.id2label[class_id.item()] if self.id2label is not None else class_id.item()
            metrics[f"map_{class_name}"] = class_map
            metrics[f"mar_100_{class_name}"] = class_mar

        metrics = {k: round(v.item(), 4) for k, v in metrics.items()}

        # Reset metric for next evaluation
        self.reset_metric()

        return metrics


def find_last_checkpoint(training_args: TrainingArguments, logger) -> Optional[str]:
    """Find the last checkpoint in the output directory according to parameters specified in `training_args`."""

    checkpoint = None
    if training_args.resume_from_checkpoint is not None:
        checkpoint = training_args.resume_from_checkpoint
    elif os.path.isdir(training_args.output_dir) and not training_args.overwrite_output_dir:
        checkpoint = get_last_checkpoint(training_args.output_dir)
        if checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    return checkpoint