---
library_name: transformers
base_model: mask2former/checkpoints/standard
tags:
- image-segmentation
- instance-segmentation
- vision
- generated_from_trainer
model-index:
- name: '25_03_13_09'
  results: []
---

<!-- This model card has been generated automatically according to the information the Trainer had access to. You
should probably proofread and complete it, then remove this comment. -->

# 25_03_13_09

This model is a fine-tuned version of [mask2former/checkpoints/standard](https://huggingface.co/mask2former/checkpoints/standard) on the qubvel-hf/ade20k-mini dataset.
It achieves the following results on the evaluation set:
- Loss: 25.1356
- Map: 0.1623
- Map 50: 0.5321
- Map 75: 0.1782
- Map Small: 0.1857
- Map Medium: 0.2487
- Map Large: -1.0
- Mar 1: 0.2
- Mar 10: 0.375
- Mar 100: 0.425
- Mar Small: 0.7
- Mar Medium: 0.3333
- Mar Large: -1.0
- Map Background: -1.0
- Mar 100 Background: -1.0
- Map Organ: 0.1623
- Mar 100 Organ: 0.425

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
- eval_batch_size: 1
- seed: 42
- optimizer: Use adamw_torch with betas=(0.9,0.999) and epsilon=1e-08 and optimizer_args=No additional optimizer arguments
- lr_scheduler_type: constant
- num_epochs: 10

### Training results

| Training Loss | Epoch | Step | Validation Loss | Map    | Map 50 | Map 75 | Map Small | Map Medium | Map Large | Mar 1 | Mar 10 | Mar 100 | Mar Small | Mar Medium | Mar Large | Map Background | Mar 100 Background | Map Organ | Mar 100 Organ |
|:-------------:|:-----:|:----:|:---------------:|:------:|:------:|:------:|:---------:|:----------:|:---------:|:-----:|:------:|:-------:|:---------:|:----------:|:---------:|:--------------:|:------------------:|:---------:|:-------------:|
| 69.6291       | 1.0   | 3    | 48.7655         | 0.0013 | 0.0042 | 0.0    | 0.0       | 0.0022     | -1.0      | 0.0   | 0.0    | 0.075   | 0.0       | 0.1        | -1.0      | -1.0           | -1.0               | 0.0013    | 0.075         |
| 49.2939       | 2.0   | 6    | 39.7085         | 0.0    | 0.0    | 0.0    | 0.0       | 0.0        | -1.0      | 0.0   | 0.0    | 0.0     | 0.0       | 0.0        | -1.0      | -1.0           | -1.0               | 0.0       | 0.0           |
| 37.1232       | 3.0   | 9    | 38.2639         | 0.0421 | 0.2525 | 0.0    | 0.0533    | 0.0673     | -1.0      | 0.075 | 0.1    | 0.1     | 0.2       | 0.0667     | -1.0      | -1.0           | -1.0               | 0.0421    | 0.1           |
| 42.2145       | 4.0   | 12   | 42.0970         | 0.0659 | 0.202  | 0.0368 | 0.06      | 0.1964     | -1.0      | 0.175 | 0.25   | 0.25    | 0.3       | 0.2333     | -1.0      | -1.0           | -1.0               | 0.0659    | 0.25          |
| 40.1038       | 5.0   | 15   | 38.5478         | 0.09   | 0.202  | 0.0644 | 0.0809    | 0.2441     | -1.0      | 0.0   | 0.3    | 0.3     | 0.4       | 0.2667     | -1.0      | -1.0           | -1.0               | 0.09      | 0.3           |
| 43.2808       | 6.0   | 18   | 32.5288         | 0.0387 | 0.0842 | 0.0135 | 0.0447    | 0.1212     | -1.0      | 0.0   | 0.325  | 0.325   | 0.5       | 0.2667     | -1.0      | -1.0           | -1.0               | 0.0387    | 0.325         |
| 33.1785       | 7.0   | 21   | 30.8641         | 0.0748 | 0.1353 | 0.0777 | 0.1515    | 0.2693     | -1.0      | 0.1   | 0.35   | 0.35    | 0.6       | 0.2667     | -1.0      | -1.0           | -1.0               | 0.0748    | 0.35          |
| 34.9336       | 8.0   | 24   | 29.1330         | 0.1376 | 0.3812 | 0.0612 | 0.3313    | 0.1389     | -1.0      | 0.075 | 0.325  | 0.375   | 0.7       | 0.2667     | -1.0      | -1.0           | -1.0               | 0.1376    | 0.375         |
| 29.004        | 9.0   | 27   | 24.9001         | 0.0909 | 0.2165 | 0.1122 | 0.1738    | 0.2432     | -1.0      | 0.05  | 0.35   | 0.425   | 0.7       | 0.3333     | -1.0      | -1.0           | -1.0               | 0.0909    | 0.425         |
| 26.2719       | 10.0  | 30   | 25.1356         | 0.1623 | 0.5321 | 0.1782 | 0.1857    | 0.2487     | -1.0      | 0.2   | 0.375  | 0.425   | 0.7       | 0.3333     | -1.0      | -1.0           | -1.0               | 0.1623    | 0.425         |


### Framework versions

- Transformers 4.47.0.dev0
- Pytorch 2.2.2
- Datasets 3.1.0
- Tokenizers 0.20.3
