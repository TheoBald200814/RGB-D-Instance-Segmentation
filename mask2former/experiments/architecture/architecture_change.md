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
| 编号  | 内容                                                                                                                              | 状态  |    日期    |
|:---:|---------------------------------------------------------------------------------------------------------------------------------|-----|:--------:|
| 实验一 | 测试Mask2Former各模块之间的数据传递格式(例如Backbone的input格式和output格式)                                                                          | 已完成 | 25/03/13 |
| 实验二 | 准备小规模实验数据集，作为对照实验(使用指定seed训练)，训练标准Mask2Former模型，并得到validation数据                                                                 | 已完成 | 24/12/19 |
| 实验三 | 继承标准模型使用的Config类、Backbone(Swin)类、Pixel decoder类、Transformer类，使用上述seed进行训练，验证得出的validation数据是否一致                                 | 已完成 | 24/12/20 |
| 实验四 | 扩展深度数据输入流：改造数据集配置文件(.json)、load_dataset策略、augment_and_transform、CustomMask2FormerForUniversalSegmentation.forward中的pixel_calues | 已完成 | 25/03/13 |
| 实验五 | 扩展Mask2FormerPixelLevelModule中的backbone(encoder),实现颜色数据和深度数据的分立特征提取、特征融合                                                        |     | 25_03_13 |

## 实验路径
``` 
/shrimpDetection/mask2former/experiments/architecture
```
## 实验数据集路径
```
/shrimpDetection/dataset/local/experiment_tiby_set
```

---

## 实验记录
### 实验一(测试Mask2Former各模块之间的数据传递格式)
#### backbone(encoder)
|      input       |                                      output                                       |
|:----------------:|:---------------------------------------------------------------------------------:|
| (1, 6, 256, 256) | ((1, 96, 64, 64),<br/>(1, 192, 32, 32),<br/>(1, 384, 16, 16),<br/>(1, 768, 8, 8)) |

- backbone input
![backbone_input](../../../log/25_03_13/backbone_input.png)
- backbone output
![backbone_output](../../../log/25_03_13/backbone_output.png)

#### pixel decoder
|                                       input                                       |                                       output                                       |
|:---------------------------------------------------------------------------------:|:----------------------------------------------------------------------------------:|
| ((1, 96, 64, 64),<br/>(1, 192, 32, 32),<br/>(1, 384, 16, 16),<br/>(1, 768, 8, 8)) | ((1, 256, 8, 8),<br/>(1, 256, 16, 16),<br/>(1, 256, 32, 32),<br/>(1, 256, 64, 64)) |

- pixel decoder input
![pixel_decoder_input](../../../log/25_03_13/pixel_decoder_input.png)
- pixel decoder output
![pixel_decoder_output](../../../log/25_03_13/pixel_decoder_output.png)

#### transformer decoder
|                                        input                                       |
|:----------------------------------------------------------------------------------:|
| ((1, 256, 8, 8),<br/>(1, 256, 16, 16),<br/>(1, 256, 32, 32),<br/>(1, 256, 64, 64)) |

- transformer decoder input
![transofrmer_decoder_input](../../../log/25_03_13/transformer_input.png)
---

### 实验二(固定seed对照试验)

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

---

### 实验三(继承模型核心逻辑class)
#### Mask2Former 代码结构解析(UML 类图)
![Mask2Former UML 类图](Mask2FormerArchitecture.png)
#### finetuning 结构优化&整理
finetuning.py在实验三中的路径
```
./24_12_20/exp3_finetuning.py
```
|                                函数/类/结构                                 | 原位置                           | 现位置                              |
|:----------------------------------------------------------------------:|-------------------------------|----------------------------------|
|                               Arguments                                | ./24_12_20/exp3_finetuning.py | ./utils/arguments.py             |
| augment_and_transform_batch <br> augment_and_transform <br> collate_fn | ./24_12_20/exp3_finetuning.py | ./utils/augment_and_transform.py |
|                             setup_logging                              | ./24_12_20/exp3_finetuning.py | ./utils/log.py                   |
| ModelOutput <br> nested_cpu <br> Evaluator <br> find_last_checkpoint   | ./24_12_20/exp3_finetuning.py | ./utils/model_essential_part.py  |
#### 继承一 Mask2FormerConfig
```python
class CustomConfig(Mask2FormerConfig):
    model_type = "mask2former"

    def __init__(self, attribute=1, **kwargs):
        self.attribute = attribute
        super().__init__(**kwargs)
```
#### 继承二 Mask2FormerForUniversalSegmentation
```python
class CustomMask2FormerForUniversalSegmentation(Mask2FormerForUniversalSegmentation):
    main_input_name = "pixel_values"
    config_class = CustomConfig

    def __init__(self, config):
        super().__init__(config)
        set_seed(42)
        self.model = CustomMask2FormerModel(config)
```
#### 继承三 Mask2FormerModel
```python
class CustomMask2FormerModel(Mask2FormerModel):
    main_input_name = "pixel_values"
    def __init__(self, config):
        print("在CustomMask2FormerModel的构造函数中执行super().__init__(config)")
        super().__init__(config)
```
#### 继承四 Mask2FormerPixelLevelModule
```python
class CustomMask2FormerPixelLevelModule(Mask2FormerPixelLevelModule):
    main_input_name = "pixel_values"
    def __init__(self, config):
        print("在CustomMask2FormerPixelLevelModule的构造函数中执行super().__init__(config)")
        super().__init__(config)
```
#### 继承后进行模型训练实验，得到的test metrics

  |            指标            | 24/12/20/03（继承一、二、三）| 24/12/20/04（继承一、二、三、四） |
  |:------------------------:|:--------------------:|:----------------------:|
  |          epoch           |         1.0          |          1.0           |
  |        test_loss         |       21.3779        |        21.3779         |
  |         test_map         |        0.2113        |         0.2113         |
  |       test_map_50        |        0.3012        |         0.3012         |
  |       test_map_75        |        0.3012        |         0.3012         |
  |   test_map_background    |         -1.0         |          -1.0          |
  |      test_map_large      |         -1.0         |          -1.0          |
  |     test_map_medium      |        0.2649        |         0.2649         |
  |     test_map_shrimp      |        0.2113        |         0.2113         |
  |      test_map_small      |         0.2          |          0.2           |
  |        test_mar_1        |        0.375         |         0.375          |
  |       test_mar_10        |        0.525         |         0.525          |
  |       test_mar_100       |        0.725         |         0.725          |
  | test_mar_100_backgr ound |         -1.0         |          -1.0          |
  |   test_mar_100_shrimp    |        0.725         |         0.725          |
  |      test_mar_large      |         -1.0         |          -1.0          |
  |     test_mar_medium      |         0.7          |          0.7           |
  |      test_mar_small      |         0.8          |          0.8           |
  |       test_runtime       |      0:00:03.51      |       0:00:03.41       |
  | test_samples_per_second  |        0.853         |         0.877          |
  |  test_steps_per_second   |        0.284         |         0.292          |
 #### 结论
根据上述“类图”可分析出Mask2Former模型内部的继承关系、聚合关系。基于这些关系实现类的继承、替换，就可以实现架构的调整。

在实验过程中，为保证继承后的类（未作出任何逻辑变化和调整时）与原有的类完全一致（能够完全接替原有的类且保证实验数据不产生任何变化），在继承类内部实例化聚合类时，需要使用```set_seed(42)```重置种子，保证实例化的聚合类的初始化参数权重完全一致。

---

### 实验四(扩展深度数据输入流)
#### 扩展深度数据输入流的方法

- 在数据集的配置文件中，image字段使用list记录多张输入图像路径（颜色图像数据、深度图像数据）
```json
[
    { 
        "image": [
            "dataset/local/experiment_tiny_set/valid/images/10.jpg",
            "dataset/local/experiment_tiny_set/valid/images/11.jpg"
        ],
        "annotation": "dataset/local/experiment_tiny_set/valid/mask/10.png",
        "semantic_class_to_id": {
            "background": 0,
            "organ": 1
        }
    }
]
```

- 在使用load_dataset通过json文件格式加载数据时，使用以下方式调整加载格式
```python
dataset = load_dataset("json", data_files=data_files)
dataset = dataset.cast_column("image", [Image()])
dataset = dataset.cast_column("annotation", Image())
```

- augment_and_transform适配调整
```python
image = np.array(example["image"])
image = image.transpose(1, 2, 0, 3).reshape(image.shape[1], image.shape[2], -1)
    
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
```

- 模型数据输入流适配调整
```python
class CustomMask2FormerForUniversalSegmentation(Mask2FormerForUniversalSegmentation):
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

```

#### 对照实验结果

  |            指标            | exp4_finetuning_single | exp4_finetuning_single |
  |:------------------------:|:----------------------:|:----------------------:|
  |          epoch           |          1.0           |          1.0           |
  |        test_loss         |        23.9534         |        23.9534         |
  |         test_map         |         0.3795         |         0.3795         |
  |       test_map_50        |         0.5136         |         0.5136         |
  |       test_map_75        |         0.5136         |         0.5136         |
  |   test_map_background    |          -1.0          |          -1.0          |
  |      test_map_large      |          -1.0          |          -1.0          |
  |     test_map_medium      |         0.4899         |         0.4899         |
  |      test_map_organ      |         0.3795         |         0.3795         |
  |      test_map_small      |         0.0229         |         0.0229         |
  |        test_mar_1        |         0.4          |          0.4           |
  |       test_mar_10        |         0.575          |         0.575          |
  |       test_mar_100       |         0.6          |          0.6           |
  | test_mar_100_backgr ound |          -1.0          |          -1.0          |
  |    test_mar_100_organ    |          0.6          |          0.6           |
  |      test_mar_large      |          -1.0          |          -1.0          |
  |     test_mar_medium      |          0.5333           |         0.5333         |
  |      test_mar_small      |          0.8           |          0.8           |
  |       test_runtime       |       0:00:03.63       |       0:00:03.12       |
  | test_samples_per_second  |         0.825          |         0.957          |
  |  test_steps_per_second   |         0.275          |         0.319          |

#### 结论
对于相同的数据集，做出上述代码层面的调整后，训练及验证结果保持不变

---

### 实验五
