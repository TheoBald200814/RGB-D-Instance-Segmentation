# 基于 Mask2Former 的架构调整实验
## Mask2Former 模型概述
### 模型架构
- Backbone(Swin): 主干网络，用于提取图像特征，生成feature map
- Pixel decoder: 像素级decoder，对feature map进行上采样操作，并实现多尺度特称embedding操作
- Transformer decoder: 初始化N个queries，并基于masked attention机制和FFN处理多尺度pixel-level embedding数据，用于增强N个queries
- Detect header: 基于MLP生成pixel-level masks and class of each mask。后处理（丢弃无效queries及其对应的pixel-level mask）

### 模型用途
- 基于虾类图像数据，学习虾及其脏器图像区域特征，执行较为精确的实例分割任务。

## 实验清单
| 编号  | 内容                                                                                             | 状态  |    日期     |
|:---:|------------------------------------------------------------------------------------------------|-----|:---------:|
| 实验一 | 测试Mask2Former各模块之间的数据传递格式(例如Backbone的input格式和output格式)                                         |     |           |
| 实验二 | 准备小规模实验数据集，作为对照实验(使用指定seed训练)，训练标准Mask2Former模型，并得到validation数据                                | 已完成 | 24/12/19  |
| 实验三 | 继承标准模型使用的Config类、Backbone(Swin)类、Pixel decoder类、Transformer类，使用上述seed进行训练，验证得出的validation数据是否一致 | 已完成 | 24/12/20  |

## 实验路径
``` 
/Users/theobald/Documents/code_lib/python_lib/shrimpDetection/mask2former/experiments
```

## 实验记录
### 实验一
### 实验二
#### 实验数据集
```/Users/theobald/Documents/code_lib/python_lib/shrimpDetection/mask2former/experiments/tiny_datasets ```
#### test metrics

  |            指标            | 24/12/19/01 | 24/12/19/02 | 24/12/19/03 |
  |:------------------------:|:-----------:|:----------:|:----------:|
  |          epoch           |     1.0     |     1.0    |     1.0    |
  |        test_loss         |   21.3779   |   21.3779  |   21.3779  |
  |         test_map         |   0.2113    |   0.2113   |   0.2113   |
  |       test_map_50        |   0.3012    |   0.3012   |   0.3012   |
  |       test_map_75        |   0.3012    |   0.3012   |   0.3012   |
  |   test_map_background    |    -1.0     |    -1.0    |    -1.0    |
  |      test_map_large      |    -1.0     |    -1.0    |    -1.0    |
  |     test_map_medium      |   0.2649    |   0.2649   |   0.2649   |
  |     test_map_shrimp      |   0.2113    |   0.2113   |   0.2113   |
  |      test_map_small      |     0.2     |     0.2    |     0.2    |
  |        test_mar_1        |    0.375    |    0.375   |    0.375   |
  |       test_mar_10        |    0.525    |    0.525   |    0.525   |
  |       test_mar_100       |    0.725    |    0.725   |    0.725   |
  | test_mar_100_backgr ound |    -1.0     |    -1.0    |    -1.0    |
  |   test_mar_100_shrimp    |    0.725    |    0.725   |    0.725   |
  |      test_mar_large      |    -1.0     |    -1.0    |    -1.0    |
  |     test_mar_medium      |     0.7     |     0.7    |     0.7    |
  |      test_mar_small      |     0.8     |     0.8    |     0.8    |
  |       test_runtime       | 0:00:03.31  | 0:00:03.15 | 0:00:03.33 |
  | test_samples_per_second  |    0.905    |    0.949   |     0.9    |
  |  test_steps_per_second   |    0.302    |   0.316    |    0.3     |

#### 结论
在exp2_finetuning中设置了set_seed(seed=42)；exp_config.json中配置"dataloader_num_woekers":0后，能够实现训练结果及验证结果一致

### 实验三
#### Mask2Former 代码结构解析(UML 类图)
![Mask2Former UML 类图](Mask2FormerArchitecture.png)
#### finetuning 结构优化&整理
finetuning.py在实验三中的路径
```
/Users/theobald/Documents/code_lib/python_lib/shrimpDetection/mask2former/experiments/24_12_20/exp3_finetuning.py
```
|                                函数/类/结构                                 | 原位置                                     | 现位置                                        |
|:----------------------------------------------------------------------:|-----------------------------------------|--------------------------------------------|
|                               Arguments                                | experiments/24_12_20/exp3_finetuning.py | experiments/utils/arguments.py             |
| augment_and_transform_batch <br> augment_and_transform <br> collate_fn | experiments/24_12_20/exp3_finetuning.py | experiments/utils/augment_and_transform.py |
|                             setup_logging                              | experiments/24_12_20/exp3_finetuning.py | experiments/utils/log.py                   |
| ModelOutput <br> nested_cpu <br> Evaluator <br> find_last_checkpoint   | experiments/24_12_20/exp3_finetuning.py | experiments/utils/model_essential_part.py  |
#### 继承 Mask2FormerConfig
```python
class CustomConfig(Mask2FormerConfig):
    model_type = "mask2former"

    def __init__(self, attribute=1, **kwargs):
        self.attribute = attribute
        super().__init__(**kwargs)
```
#### 继承 Mask2FormerForUniversalSegmentation
```python
class CustomMask2FormerForUniversalSegmentation(Mask2FormerForUniversalSegmentation):
    main_input_name = "pixel_values"
    config_class = CustomConfig

    def __init__(self, config):
        super().__init__(config)
        set_seed(42)
        self.model = CustomMask2FormerModel(config)
```
#### 继承 Mask2FormerModel
```python
class CustomMask2FormerModel(Mask2FormerModel):
    main_input_name = "pixel_values"
    def __init__(self, config):
        print("在CustomMask2FormerModel的构造函数中执行super().__init__(config)")
        super().__init__(config)
```
#### 继承后进行模型训练实验，得到的test metrics

  |            指标            | 24/12/20/03 |
  |:------------------------:|:-----------:|
  |          epoch           |     1.0     |
  |        test_loss         |   21.3779   |
  |         test_map         |   0.2113    |
  |       test_map_50        |   0.3012    |
  |       test_map_75        |   0.3012    |
  |   test_map_background    |    -1.0     |
  |      test_map_large      |    -1.0     |
  |     test_map_medium      |   0.2649    |
  |     test_map_shrimp      |   0.2113    |
  |      test_map_small      |     0.2     |
  |        test_mar_1        |    0.375    |
  |       test_mar_10        |    0.525    |
  |       test_mar_100       |    0.725    |
  | test_mar_100_backgr ound |    -1.0     |
  |   test_mar_100_shrimp    |    0.725    |
  |      test_mar_large      |    -1.0     |
  |     test_mar_medium      |     0.7     |
  |      test_mar_small      |     0.8     |
  |       test_runtime       | 0:00:03.51  |
  | test_samples_per_second  |    0.853    |
  |  test_steps_per_second   |    0.284    |
 #### 结论
根据上述“类图”可分析出Mask2Former模型内部的继承关系、聚合关系。基于这些关系实现类的继承、替换，就可以实现架构的调整。

在实验过程中，为保证继承后的类（未作出任何逻辑变化和调整时）与原有的类完全一致（能够完全接替原有的类且保证实验数据不产生任何变化），在继承类内部实例化聚合类时，需要使用```set_seed(42)```重置种子，保证实例化的聚合类的初始化参数权重完全一致。
