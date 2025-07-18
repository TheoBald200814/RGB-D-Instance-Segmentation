---
library_name: transformers
base_model: /root/autodl-tmp/shrimpDetection/mask2former/checkpoints/standard
tags:
- image-segmentation
- instance-segmentation
- vision
- generated_from_trainer
model-index:
- name: single_640
  results: []
---

<!-- This model card has been generated automatically according to the information the Trainer had access to. You
should probably proofread and complete it, then remove this comment. -->

# single_640

This model is a fine-tuned version of [/root/autodl-tmp/shrimpDetection/mask2former/checkpoints/standard](https://huggingface.co//root/autodl-tmp/shrimpDetection/mask2former/checkpoints/standard) on the qubvel-hf/ade20k-mini dataset.
It achieves the following results on the evaluation set:
- Loss: 16.2728
- Map: 0.3721
- Map 50: 0.8177
- Map 75: 0.3495
- Map Small: 0.3208
- Map Medium: 0.6058
- Map Large: -1.0
- Mar 1: 0.0791
- Mar 10: 0.4423
- Mar 100: 0.4679
- Mar Small: 0.4228
- Mar Medium: 0.6356
- Mar Large: -1.0
- Map Organ: 0.1968
- Mar 100 Organ: 0.3474
- Map Shrimp: 0.5473
- Mar 100 Shrimp: 0.5883

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
- train_batch_size: 4
- eval_batch_size: 8
- seed: 42
- optimizer: Use OptimizerNames.ADAMW_TORCH with betas=(0.9,0.999) and epsilon=1e-08 and optimizer_args=No additional optimizer arguments
- lr_scheduler_type: constant
- num_epochs: 300
- mixed_precision_training: Native AMP

### Training results

| Training Loss | Epoch    | Step | Validation Loss | Map    | Map 50 | Map 75 | Map Small | Map Medium | Map Large | Mar 1  | Mar 10 | Mar 100 | Mar Small | Mar Medium | Mar Large | Map Background | Mar 100 Background | Map Organ | Mar 100 Organ | Map Shrimp | Mar 100 Shrimp |
|:-------------:|:--------:|:----:|:---------------:|:------:|:------:|:------:|:---------:|:----------:|:---------:|:------:|:------:|:-------:|:---------:|:----------:|:---------:|:--------------:|:------------------:|:---------:|:-------------:|:----------:|:--------------:|
| 32.2791       | 9.7143   | 136  | 30.1672         | 0.2473 | 0.4536 | 0.2806 | 0.1952    | 0.5496     | -1.0      | 0.0503 | 0.2726 | 0.2784  | 0.2286    | 0.595      | -1.0      | -1.0           | -1.0               | 0.0005    | 0.0139        | 0.4941     | 0.5429         |
| 26.6929       | 19.4286  | 272  | 24.9398         | 0.277  | 0.5851 | 0.2927 | 0.2268    | 0.567      | -1.0      | 0.0624 | 0.3399 | 0.3684  | 0.3213    | 0.6059     | -1.0      | -1.0           | -1.0               | 0.0454    | 0.1803        | 0.5087     | 0.5565         |
| 23.6755       | 29.1429  | 408  | 23.1876         | 0.309  | 0.6648 | 0.3392 | 0.2526    | 0.5834     | -1.0      | 0.0672 | 0.3872 | 0.4273  | 0.3757    | 0.6198     | -1.0      | -1.0           | -1.0               | 0.0974    | 0.2891        | 0.5206     | 0.5656         |
| 21.5345       | 38.8571  | 544  | 20.8646         | 0.3365 | 0.7429 | 0.3587 | 0.2861    | 0.5942     | -1.0      | 0.0724 | 0.4069 | 0.4456  | 0.3988    | 0.6257     | -1.0      | -1.0           | -1.0               | 0.1394    | 0.3146        | 0.5336     | 0.5766         |
| 20.2141       | 48.5714  | 680  | 20.4941         | 0.3477 | 0.7984 | 0.3443 | 0.2985    | 0.5842     | -1.0      | 0.0786 | 0.4164 | 0.4478  | 0.3987    | 0.6158     | -1.0      | -1.0           | -1.0               | 0.1663    | 0.3314        | 0.529      | 0.5643         |
| 18.2227       | 58.2857  | 816  | 19.5230         | 0.3545 | 0.8052 | 0.3499 | 0.3056    | 0.5944     | -1.0      | 0.0736 | 0.4195 | 0.4545  | 0.4068    | 0.6248     | -1.0      | -1.0           | -1.0               | 0.1719    | 0.3343        | 0.5371     | 0.5747         |
| 17.3733       | 68.0     | 952  | 19.0869         | 0.3583 | 0.8118 | 0.34   | 0.3069    | 0.5885     | -1.0      | 0.0799 | 0.4213 | 0.4534  | 0.4067    | 0.6178     | -1.0      | -1.0           | -1.0               | 0.1884    | 0.338         | 0.5282     | 0.5688         |
| 16.7332       | 77.7143  | 1088 | 18.7423         | 0.3545 | 0.7954 | 0.3329 | 0.3094    | 0.5881     | -1.0      | 0.0781 | 0.4198 | 0.4548  | 0.4096    | 0.6228     | -1.0      | -1.0           | -1.0               | 0.1732    | 0.3343        | 0.5358     | 0.5753         |
| 16.1726       | 87.4286  | 1224 | 18.4705         | 0.357  | 0.8204 | 0.3374 | 0.3135    | 0.5871     | -1.0      | 0.0732 | 0.4231 | 0.456   | 0.4119    | 0.6267     | -1.0      | -1.0           | -1.0               | 0.1795    | 0.3314        | 0.5345     | 0.5805         |
| 15.2877       | 97.1429  | 1360 | 18.2020         | 0.3478 | 0.7935 | 0.3384 | 0.3035    | 0.5881     | -1.0      | 0.0695 | 0.4263 | 0.4529  | 0.4088    | 0.6287     | -1.0      | 0.1616         | 0.3234             | 0.534     | 0.5825        |
| 14.4883       | 106.8571 | 1496 | 17.8601         | 0.3591 | 0.8161 | 0.3503 | 0.3102    | 0.6038     | -1.0      | 0.0725 | 0.4199 | 0.4546  | 0.4058    | 0.6356     | -1.0      | -1.0           | -1.0               | 0.1745    | 0.3248        | 0.5437     | 0.5844         |
| 14.2121       | 116.5714 | 1632 | 17.6283         | 0.3605 | 0.7909 | 0.3501 | 0.3154    | 0.5994     | -1.0      | 0.0718 | 0.4212 | 0.449   | 0.4036    | 0.6347     | -1.0      | 0.1804         | 0.3109             | 0.5406    | 0.587         |
| 14.1863       | 126.2857 | 1768 | 17.3429         | 0.3795 | 0.843  | 0.3384 | 0.3352    | 0.612      | -1.0      | 0.0792 | 0.4392 | 0.4669  | 0.4249    | 0.6416     | -1.0      | -1.0           | -1.0               | 0.202     | 0.3365        | 0.5571     | 0.5974         |
| 13.785        | 136.0    | 1904 | 17.2800         | 0.3829 | 0.8436 | 0.3551 | 0.3399    | 0.6075     | -1.0      | 0.0784 | 0.4409 | 0.4715  | 0.4301    | 0.6376     | -1.0      | 0.2121         | 0.3489             | 0.5538    | 0.5942        |
| 13.2736       | 145.7143 | 2040 | 17.5556         | 0.3754 | 0.8316 | 0.3617 | 0.332     | 0.6049     | -1.0      | 0.0812 | 0.4352 | 0.4538  | 0.4087    | 0.6337     | -1.0      | 0.1974         | 0.3212             | 0.5534    | 0.5864        |
| 13.101        | 155.4286 | 2176 | 16.4822         | 0.3816 | 0.8381 | 0.368  | 0.3359    | 0.6155     | -1.0      | 0.0813 | 0.4419 | 0.4747  | 0.4346    | 0.6376     | -1.0      | 0.202          | 0.354              | 0.5613    | 0.5955        |
| 12.7733       | 165.1429 | 2312 | 17.0054         | 0.3866 | 0.852  | 0.3711 | 0.3429    | 0.617      | -1.0      | 0.0837 | 0.4457 | 0.4683  | 0.4243    | 0.6455     | -1.0      | 0.2079         | 0.3372             | 0.5654    | 0.5994        |
| 12.0416       | 174.8571 | 2448 | 16.7158         | 0.3838 | 0.8415 | 0.3681 | 0.3387    | 0.6133     | -1.0      | 0.0833 | 0.4402 | 0.4665  | 0.4225    | 0.6396     | -1.0      | 0.2072         | 0.3394             | 0.5605    | 0.5935        |
| 12.0686       | 184.5714 | 2584 | 16.1310         | 0.3829 | 0.8732 | 0.3626 | 0.3402    | 0.6078     | -1.0      | 0.0791 | 0.4465 | 0.4677  | 0.4238    | 0.6376     | -1.0      | 0.2076         | 0.3438             | 0.5583    | 0.5916        |
| 11.7637       | 194.2857 | 2720 | 16.9094         | 0.3771 | 0.8292 | 0.3465 | 0.3289    | 0.6029     | -1.0      | 0.0779 | 0.4427 | 0.4606  | 0.4127    | 0.6327     | -1.0      | 0.2104         | 0.3387             | 0.5438    | 0.5825        |
| 11.711        | 204.0    | 2856 | 17.0272         | 0.3793 | 0.8364 | 0.3509 | 0.3358    | 0.6103     | -1.0      | 0.0763 | 0.4469 | 0.4786  | 0.4337    | 0.6446     | -1.0      | 0.2004         | 0.3599             | 0.5581    | 0.5974        |
| 11.2233       | 213.7143 | 2992 | 16.5971         | 0.3912 | 0.8549 | 0.3691 | 0.3444    | 0.6134     | -1.0      | 0.0794 | 0.4474 | 0.4719  | 0.4305    | 0.6376     | -1.0      | 0.2215         | 0.3496             | 0.5609    | 0.5942        |
| 11.0667       | 223.4286 | 3128 | 16.6105         | 0.376  | 0.8257 | 0.3534 | 0.332     | 0.6034     | -1.0      | 0.0767 | 0.4429 | 0.4688  | 0.4271    | 0.6347     | -1.0      | 0.1995         | 0.3467             | 0.5524    | 0.5909        |
| 10.8935       | 233.1429 | 3264 | 16.4624         | 0.3799 | 0.8321 | 0.3626 | 0.3302    | 0.6119     | -1.0      | 0.0773 | 0.4468 | 0.4717  | 0.4302    | 0.6416     | -1.0      | 0.2022         | 0.3453             | 0.5576    | 0.5981        |
| 10.6283       | 242.8571 | 3400 | 16.4740         | 0.3814 | 0.829  | 0.379  | 0.3325    | 0.6189     | -1.0      | 0.0815 | 0.4426 | 0.4729  | 0.4258    | 0.6455     | -1.0      | 0.2021         | 0.3496             | 0.5606    | 0.5961        |
| 10.0824       | 252.5714 | 3536 | 16.4195         | 0.3747 | 0.7991 | 0.3655 | 0.326     | 0.6066     | -1.0      | 0.0822 | 0.4447 | 0.4659  | 0.4238    | 0.6416     | -1.0      | 0.198          | 0.3343             | 0.5514    | 0.5974        |
| 10.3928       | 262.2857 | 3672 | 15.8370         | 0.3798 | 0.8101 | 0.3587 | 0.3331    | 0.614      | -1.0      | 0.0857 | 0.4406 | 0.4647  | 0.4204    | 0.6446     | -1.0      | 0.1978         | 0.3314             | 0.5618    | 0.5981        |
| 10.3044       | 272.0    | 3808 | 16.4322         | 0.3807 | 0.8136 | 0.3852 | 0.3303    | 0.6216     | -1.0      | 0.0786 | 0.4461 | 0.4636  | 0.4174    | 0.6465     | -1.0      | 0.1972         | 0.3292             | 0.5642    | 0.5981        |
| 10.0074       | 281.7143 | 3944 | 16.7280         | 0.3873 | 0.8172 | 0.374  | 0.3369    | 0.6148     | -1.0      | 0.0813 | 0.4452 | 0.4663  | 0.4159    | 0.6426     | -1.0      | 0.2154         | 0.3431             | 0.5592    | 0.5896        |
| 9.9264        | 291.4286 | 4080 | 16.2728         | 0.3721 | 0.8177 | 0.3495 | 0.3208    | 0.6058     | -1.0      | 0.0791 | 0.4423 | 0.4679  | 0.4228    | 0.6356     | -1.0      | 0.1968         | 0.3474             | 0.5473    | 0.5883        |


### Framework versions

- Transformers 4.47.0
- Pytorch 2.3.0+cu121
- Datasets 3.1.0
- Tokenizers 0.21.0
