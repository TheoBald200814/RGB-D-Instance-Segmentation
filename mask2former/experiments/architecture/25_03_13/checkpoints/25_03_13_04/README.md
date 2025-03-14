---
library_name: transformers
base_model: mask2former/checkpoints/standard
tags:
- image-segmentation
- instance-segmentation
- vision
- generated_from_trainer
model-index:
- name: '25_03_13_04'
  results: []
---

<!-- This model card has been generated automatically according to the information the Trainer had access to. You
should probably proofread and complete it, then remove this comment. -->

# 25_03_13_04

This model is a fine-tuned version of [mask2former/checkpoints/standard](https://huggingface.co/mask2former/checkpoints/standard) on the qubvel-hf/ade20k-mini dataset.
It achieves the following results on the evaluation set:
- Loss: 26.8539
- Map: 0.3688
- Map 50: 0.5115
- Map 75: 0.5115
- Map Small: 0.7333
- Map Medium: 0.4156
- Map Large: -1.0
- Mar 1: 0.525
- Mar 10: 0.55
- Mar 100: 0.775
- Mar Small: 0.8
- Mar Medium: 0.7667
- Mar Large: -1.0
- Map Background: -1.0
- Mar 100 Background: -1.0
- Map Organ: 0.3688
- Mar 100 Organ: 0.775

## Model description

More information needed

## Intended uses & limitations

More information needed

## Training and evaluation data

More information needed

## Training procedure

### Training hyperparameters

The following hyperparameters were used during training:
- learning_rate: 1e-05
- train_batch_size: 1
- eval_batch_size: 8
- seed: 42
- optimizer: Use adamw_torch with betas=(0.9,0.999) and epsilon=1e-08 and optimizer_args=No additional optimizer arguments
- lr_scheduler_type: constant
- num_epochs: 1

### Training results

| Training Loss | Epoch | Step | Validation Loss | Map    | Map 50 | Map 75 | Map Small | Map Medium | Map Large | Mar 1 | Mar 10 | Mar 100 | Mar Small | Mar Medium | Mar Large | Map Background | Mar 100 Background | Map Organ | Mar 100 Organ |
|:-------------:|:-----:|:----:|:---------------:|:------:|:------:|:------:|:---------:|:----------:|:---------:|:-----:|:------:|:-------:|:---------:|:----------:|:---------:|:--------------:|:------------------:|:---------:|:-------------:|
| 30.4855       | 1.0   | 3    | 26.8539         | 0.3688 | 0.5115 | 0.5115 | 0.7333    | 0.4156     | -1.0      | 0.525 | 0.55   | 0.775   | 0.8       | 0.7667     | -1.0      | -1.0           | -1.0               | 0.3688    | 0.775         |


### Framework versions

- Transformers 4.47.0.dev0
- Pytorch 2.2.2
- Datasets 3.1.0
- Tokenizers 0.20.3
