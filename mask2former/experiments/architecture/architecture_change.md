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
| 实验五 | 扩展Mask2FormerPixelLevelModule中的backbone(encoder),实现颜色数据和深度数据的分立特征提取、特征融合                                                        | 已完成 | 25_03_14 |
| 实验六 | 构造DSA模块（深度频率统计、深度阈值分解、深度敏感注意力机制），测试本地数据集                                                                                        | 已完成 | 25/03/23 |
| 实验七 | 构造CSF模块（余弦相似度矩阵、CSFed Image生成算法、可视化监控模块），测试本地数据集                                                                                | 已完成 | 25/03/23 |
| 实验八 | 基于实验五的“通用数据集”基准测试                                                                                                               | 已完成 | 25/03/21 |
| 实验九 | DSA模块中的深度阈值分解参数可学习化（使用额外的神经网络模块来预测window_size_ratio）                                                                                             | 已完成 | 25/05/07 |

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

### 实验五(backbone特征融合实验)
#### 实验架构设计
![实验架构设计图](../../../log/25_03_14/exp5.png)
![实验架构设计图](../../../log/25_03_14/featurefuser.png)
- 更新CustomMask2FormerPixelLevelModule：```self.color_encoder``` ```self.depth_encoder``` 和 ```self.feature_fuser```。
```python
   def __init__(self, config):
        print("[CustomMask2FormerPixelLevelModule] constructing...")
        super().__init__(config)

        self.color_encoder = load_backbone(config)
        self.depth_encoder = load_backbone(config)
        self.feature_fuser = FeatureFuser()
```

- 新增FeatureFuser
```python
merged_map = [torch.cat([c, d], dim=1) for c, d in zip(color_feature_map, depth_feature_map)]
fused_map = [conv(m) for conv, m in zip(self.fuse_conv, merged_map)]
```
#### 模型训练测试
| 指标 | 原生backbone | 双重backbone(仅初始化，未参与数据流) | 双重backbone(参与数据流) |
|:--:|:----------:|:-----------------------:|:-----------------:|
 |          epoch           |    1.0     |           1.0           |       10.0        |
  |        test_loss         |  23.9534   |         26.9443         |      25.1690      |
  |         test_map         |   0.3795   |         0.3688          |      0.1623       |
  |       test_map_50        |   0.5136   |         0.5115          |      0.5321       |
  |       test_map_75        |   0.5136   |         0.5115          |      0.1782       |
  |   test_map_background    |    -1.0    |          -1.0           |       -1.0        |
  |      test_map_large      |    -1.0    |          -1.0           |       -1.0        |
  |     test_map_medium      |   0.4899   |         0.4156          |      0.2487       |
  |      test_map_organ      |   0.3795   |         0.3688          |      0.1623       |
  |      test_map_small      |   0.0229   |         0.7333          |      0.1857       |
  |        test_mar_1        |    0.4     |          0.525          |        0.2        |
  |       test_mar_10        |   0.575    |          0.55           |       0.375       |
  |       test_mar_100       |    0.6     |          0.775          |       0.425       |
  | test_mar_100_backgr ound |    -1.0    |          -1.0           |       -1.0        |
  |    test_mar_100_organ    |    0.6     |          0.775          |       0.425       |
  |      test_mar_large      |    -1.0    |          -1.0           |       -1.0        |
  |     test_mar_medium      |   0.5333   |         0.7667          |      0.3333       |
  |      test_mar_small      |    0.8     |           0.8           |        0.7        |
  |       test_runtime       | 0:00:03.81 |       0:00:03.78        |    0:00:04.50     |
  | test_samples_per_second  |   0.788    |          0.794          |       0.652       |
  |  test_steps_per_second   |   0.263    |          0.265          |       0.652       |

#### 结论
在架构调整及新增模块注入后，数据流通正常，但模型性能指标有所下降。
可能的原因：
- 新增的模块为初始化参数，未经过预训练
- 输入的图像数据中，深度数据通道和彩色数据通道均为彩色数据；并为使用真正的深度数据
- 双backbone的融合机制过于粗暴（简单的卷积合并），待调整及优化

---

### 实验六(DSAM)
#### DSAM简述
- 深度分解 (Depth Decomposition)
    
    深度分解是DSAM的第一步，它的目的是将原始的深度图转化为一系列空间注意力掩码 (spatial attention masks)，这些掩码能够指示图像中不同深度范围的区域。 
    论文作者观察到，显著性物体往往分布在特定的深度区间内，因此可以利用深度信息来初步定位潜在的显著性区域，并减少背景干扰。

    ```深度分解的具体步骤```
    1. 深度直方图统计：首先，对原始深度图进行量化，并计算深度值的直方图。直方图可以反映深度值在图像中的分布情况，也就是哪些深度值出现的频率更高。
    2. 选择深度分布模式 (Depth Distribution Modes)：在深度直方图中，深度分布模式 指的是直方图中的峰值，也就是频率最高的深度值或深度值范围。
       论文中选择 T个最大的深度分布模式。 可以理解为，这些峰值对应的深度值范围，很可能包含了图像中主要的物体或区域。
    3. 深度区间窗口 (Depth Interval Windows): 每个深度分布模式 (峰值) 对应一个 深度区间窗口。 例如，如果一个峰值出现在深度值 1.0 附近，那么可能设置一个深度区间窗口为 [0.9, 1.1]。这些区间窗口定义了我们感兴趣的深度范围。
    4. 生成深度区域 (Depth Regions): 使用这些深度区间窗口，将原始深度图分解为 T个区域。 每个区域包含原始深度图中深度值落在对应深度区间窗口内的像素。 换句话说，对于第 t 个深度区间窗口，我们创建一个二值掩码，原始深度图中深度值在该窗口内的像素位置为 1，否则为 0。 这就得到了 T 个深度区域。
    5. 剩余区域 (Remaining Part): 直方图中，除了被选中的 T 个最大深度分布模式之外，剩余的部分 自然形成 最后一个区域 (第 T+1 个区域)。 这个区域可以理解为深度值分布相对分散或不显著的区域。
    6. 归一化 (Normalization): 每个深度区域 (包括剩余区域) 都被归一化到 [0, 1] 范围。 归一化后的深度区域就成为了 空间注意力掩码 (b<sub>1</sub>, b<sub>2</sub>, ..., b<sub>T+1</sub>)。 掩码中的值在 0 到 1 之间，值越大表示该区域的权重越高。
      
    ```总结深度分解的目的```

    - 通过深度分解，我们将原始的深度图转化为 T+1 个空间注意力掩码，每个掩码对应一个特定的深度范围，用于指示图像中不同深度层次的区域。 这些掩码将作为深度敏感注意力模块的输入，指导RGB特征的提取。


- 深度敏感注意力模块 (Depth-Sensitive Attention Module) 

    深度敏感注意力模块 (DSAM) 的作用是 利用深度分解得到的空间注意力掩码，来增强RGB特征，并抑制背景干扰。 它在RGB分支的每个下采样层之后被插入。


- DSAM数据流

    ```输入```
RGB 特征图 (F_rgb_k): 来自RGB分支第 k 阶段的特征图。
深度分解得到的空间注意力掩码 (b_1, b_2, ..., b_(T+1)): 共 T+1 个掩码。
掩码尺寸对齐 (Pooling): 由于RGB特征图 F_rgb_k 经过了下采样，其尺寸可能小于原始深度图。因此，需要使用 最大池化 (MaxPool) 操作 将深度掩码 b_t 调整到与 RGB 特征图 F_rgb_k 相同的尺寸，得到调整后的掩码 p_t = MaxPool(b_t)。 最大池化能够保留深度区域的空间范围，并降低计算复杂度。
并行子分支 (Parallel Sub-branches): DSAM 创建 T+1 个并行的子分支，每个子分支对应一个深度掩码 p_t。
元素级乘法 (Element-wise Multiplication): 在每个子分支中，将对应的掩码 p_t 与 RGB 特征图 F_rgb_k 的每个通道 进行 元素级乘法 (⊗)。 p_t ⊗ F_rgb_k 这意味着，深度掩码 p_t 作为空间注意力权重，作用于 RGB 特征图 F_rgb_k 的空间位置上。 掩码值高的区域，对应的RGB特征被增强；掩码值低的区域，对应的RGB特征被抑制。 这就实现了 深度敏感的特征选择和增强。
1x1 卷积层 (1x1 Convolution Layer): 在每个子分支的元素级乘法之后，使用一个 1x1 卷积层 (Conv<sub>1x1</sub>)。 这个 1x1 卷积层的作用是：
特征提炼 (Feature Refinement): 对经过深度掩码加权后的特征进行进一步的提炼和调整。
通道维度变换 (Channel Dimension Transformation): 1x1 卷积可以改变特征图的通道数，使其更适合后续的融合。
非线性变换 (Non-linearity): 卷积层通常包含激活函数 (虽然图中未明确标出，但通常会使用)，引入非线性，增强模型的表达能力。
元素级求和 (Element-wise Summation): 将 所有 T+1 个子分支的 1x1 卷积层的输出 进行 元素级求和 (∑)。 F_enh_k = ∑ Conv_1x1(p_t ⊗ F_rgb_k) 这样，来自不同深度区域的深度敏感特征被 聚合 起来。
残差连接 (Residual Connection): 为了更好地训练深层网络，并保留原始RGB特征的信息，DSAM 引入了 残差连接。 将 增强后的 RGB 特征 F_enh_k 与 原始 RGB 特征 F_rgb_k 进行 元素级相加 (+)，得到最终的输出特征 r_k = F_enh_k + F_rgb_k。 残差连接有助于信息流通，并减轻梯度消失问题。
输出: 增强后的 RGB 特征图 r_k，作为DSAM的输出，将传递到网络的后续层进行处理。

- 总结

    利用深度分解得到的空间注意力掩码，显式地将深度信息融入到RGB特征提取过程中。
通过元素级乘法，根据深度信息对RGB特征进行空间加权，增强与显著性物体深度范围相关的特征，抑制背景区域的特征。
通过并行子分支和 1x1 卷积，对不同深度区域的特征进行独立处理和提炼，并进行有效融合。
通过残差连接，保证信息的有效传递和网络的稳定训练。
#### [参考文献: Deep RGB-D Saliency Detection with Depth-Sensitive Attention and Automatic Multi-Modal Fusion](../../../log/25_03_23/paper_DSAM.pdf)

#### 代码实现

- 深度频率统计
```python
def calculate_depth_histogram(depth_map, bins=512, value_range=None):
    """
    计算深度图的直方图。

    Args:
        depth_map (numpy.ndarray): 输入深度图 (单通道).
        bins (int): 直方图的柱子数量 (bins). 默认 256.
        value_range (tuple, optional): 深度值的范围 (min, max).
                                      如果不指定，则使用深度图中的最小值和最大值. Defaults to None.

    Returns:
        tuple: 包含直方图计数 (hist) 和 bin 边缘 (bin_edges).
    """
    if value_range is None:
        value_range = (np.nanmin(depth_map), np.nanmax(depth_map)) # 忽略 NaN 值

    hist, bin_edges = np.histogram(depth_map.flatten(), bins=bins, range=value_range, density=False)
    return hist, bin_edges

def select_depth_distribution_modes(hist, bin_edges, num_modes=3, prominence_threshold=0.01):
    """
    从深度直方图中选择深度分布模式 (峰值).

    Args:
        hist (numpy.ndarray): 直方图计数.
        bin_edges (numpy.ndarray): bin 边缘.
        num_modes (int): 要选择的深度分布模式的数量. 默认 3.
        prominence_threshold (float): 峰值的显著性阈值 (相对于最大峰值高度).
                                     用于过滤不显著的峰值. 默认 0.01.

    Returns:
        list: 包含选定深度分布模式的中心值 (近似).
              如果找不到足够的显著峰值，则返回少于 num_modes 的列表.
    """
    from scipy.signal import find_peaks

    # 查找峰值索引
    peaks_indices, _ = find_peaks(hist, prominence=prominence_threshold * np.max(hist)) # 使用显著性阈值

    if not peaks_indices.size: # 如果没有找到峰值
        return []

    # 获取峰值的高度和位置 (近似中心值)
    peak_heights = hist[peaks_indices]
    peak_centers = bin_edges[:-1][peaks_indices] + np.diff(bin_edges)[peaks_indices] / 2.0 # 近似中心值

    # 将峰值按照高度降序排序
    peak_data = sorted(zip(peak_heights, peak_centers), reverse=True)

    selected_modes = [center for _, center in peak_data[:num_modes]] # 选择前 num_modes 个峰值中心

    return selected_modes
```

- 深度分解
```python
def define_depth_interval_windows(depth_modes, window_size_ratio=0.1):
    """
    根据深度分布模式定义深度区间窗口.

    Args:
        depth_modes (list): 深度分布模式的中心值列表.
        window_size_ratio (float): 窗口大小相对于深度模式中心值的比例. 默认 0.1.
                                   例如，ratio=0.1，则窗口宽度为中心值的 10%.

    Returns:
        list: 包含深度区间窗口 (元组 (lower_bound, upper_bound)) 的列表.
    """
    interval_windows = []
    for mode_center in depth_modes:
        window_half_width = mode_center * window_size_ratio / 2.0  # 半宽度，保证窗口宽度与中心值比例一致
        lower_bound = max(0, mode_center - window_half_width) # 保证下界不小于0，假设深度值非负
        upper_bound = mode_center + window_half_width
        interval_windows.append((lower_bound, upper_bound))
    return interval_windows


def generate_depth_region_masks(depth_map, interval_windows):
    """
    根据深度区间窗口生成深度区域掩码.

    Args:
        depth_map (numpy.ndarray): 输入深度图 (单通道).
        interval_windows (list): 深度区间窗口列表，每个窗口为元组 (lower_bound, upper_bound).

    Returns:
        list: 包含深度区域掩码 (numpy.ndarray, bool 类型) 的列表.
              最后一个掩码是剩余区域掩码.
    """
    region_masks = []
    combined_mask = np.zeros_like(depth_map, dtype=bool) # 用于记录已覆盖的区域

    for lower_bound, upper_bound in interval_windows:
        mask = (depth_map >= lower_bound) & (depth_map <= upper_bound)
        region_masks.append(mask)
        combined_mask |= mask # 累积覆盖区域

    # 生成剩余区域掩码 (深度值不在任何定义的窗口内的区域)
    remaining_mask = ~combined_mask
    region_masks.append(remaining_mask)

    return region_masks
```

- DSAM
```python
class DSAModule(nn.Module):
    def __init__(self, in_channels, out_channels, num_depth_regions=3):
        """
        深度敏感注意力模块 (DSAM)。

        Args:
            in_channels (int): 输入 RGB 特征图的通道数.
            out_channels (int): 输出增强 RGB 特征图的通道数.
            num_depth_regions (int): 深度分解的区域数量 (T).  实际子分支数量为 T+1 (包含剩余区域). 默认 3.
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_depth_regions = num_depth_regions
        self.conv_layers = nn.ModuleList([
            nn.Conv2d(in_channels, out_channels, kernel_size=1) for _ in range(num_depth_regions + 1)
        ]) # T+1 个 1x1 卷积层

    def forward(self, rgb_features, depth_map):
        """
        DSAM 的前向传播过程.

        Args:
            rgb_features (torch.Tensor): 输入 RGB 特征图, 形状为 (B, C_in, H, W).
            depth_map (torch.Tensor): 输入原始深度图, 形状为 (B, 1, H_d, W_d) 或 (B, H_d, W_d) 或 (H_d, W_d)  (单通道).
                                      注意：为了代码的通用性，函数内部会处理不同形状的深度图.

        Returns:
            torch.Tensor: 增强后的 RGB 特征图, 形状为 (B, C_out, H, W).
        """
        # 1. 深度分解 (Depth Decomposition)
        # 确保深度图是 NumPy 数组且为单通道 (如果输入是 Tensor，先转为 NumPy)
        if isinstance(depth_map, torch.Tensor):
            depth_map_np = depth_map.squeeze().cpu().detach().numpy()  # 去除通道维度，转为 NumPy, 放到 CPU
        elif isinstance(depth_map, np.ndarray):
            depth_map_np = depth_map.squeeze() # 确保是单通道
        else:
            raise TypeError("Depth map must be torch.Tensor or numpy.ndarray")

        interval_windows = [] # 初始化为空列表，防止在没有检测到 depth_modes 时报错
        region_masks = []

        hist, bin_edges = calculate_depth_histogram(depth_map_np)
        depth_modes = select_depth_distribution_modes(hist, bin_edges, num_modes=self.num_depth_regions)
        if depth_modes: # 只有当检测到 depth_modes 时才进行后续步骤，防止空列表导致错误
            interval_windows = define_depth_interval_windows(depth_modes)
            region_masks = generate_depth_region_masks(depth_map_np, interval_windows)
        else:
            # 如果没有检测到深度模式，则创建一个全零的掩码列表，保证程序正常运行，但不进行深度引导
            region_masks = [np.zeros_like(depth_map_np, dtype=bool)] * (self.num_depth_regions + 1)


        # 2. 深度敏感注意力 (Depth-Sensitive Attention)
        enhanced_features = 0
        for i in range(len(region_masks)):
            # 将 NumPy mask 转换为 PyTorch Tensor, 并放到与 rgb_features 相同的设备上
            mask_tensor = torch.from_numpy(region_masks[i]).float().unsqueeze(0).unsqueeze(0).to(rgb_features.device) # (1, 1, H_d, W_d)
            # resize mask to match rgb_features' spatial size using adaptive max pooling
            resized_mask = nn.functional.adaptive_max_pool2d(mask_tensor, rgb_features.shape[2:]) # (1, 1, H, W)

            masked_features = rgb_features * resized_mask  # 元素级乘法 (B, C_in, H, W) * (1, 1, H, W)  -> (B, C_in, H, W)
            refined_features = self.conv_layers[i](masked_features) # 1x1 卷积 (B, C_in, H, W) -> (B, C_out, H, W)
            enhanced_features += refined_features # 元素级求和


        output_features = enhanced_features + rgb_features  # 残差连接
        return output_features
```

#### 深度频率统计&分解实验结果

![结果](../../../log/25_03_23/exp6.png)

#### 结论
DSAM构建完毕，待集成接入主模型

---

### 实验7（CSF模块）
#### 余弦相似度算法（Cosine Similarity Algorithm）
- Image Cosine Similarity Fusion

  图像级别的余弦相似度计算，要求两（多）张图像具有相同的尺⼨，并将其展开为1维向量的形式。不妨设图像A和图像B，则有vector_A 和 vector_B。
  余弦相似度 = vector_A 内积 vector_B / ||vector_A|| * ||vector_B||，其中|| ||为⼆范数。


- Feature Cosine Similarity Fusion

  特征尺度的余弦相似度计算，两（多）张特征图通常具有不同的尺⼨。⾸先引⼊Conv2d将特征图
  转化为注意⼒图（⽬的在于引⼊可学习参数），并将其展开为1维向量的形式。不妨设特征图A和
  特征图B，则有vector_A 和 vector_B。
  余弦相似度 = vector_A 内积 vector_B / ||vector_A|| * ||vector_B||，其中|| ||为⼆范数。

#### CSF算法（Cosine Similarity Fuse Algorithm）
CSFed Image: An image processed by the Cosine Similarity Fusion algorithm

- ⽣成评分矩阵：

  不妨设有N张图像(O_N)需要做CSF融合（N -> 1），以轮换
  ⽅式规定1张图为标准图（A），剩余的N-1张图与A分别做余弦相似度分析，
  得到N-1个相似度评分矩阵（D_N-1）。
  
- ⽣成轮次图像及原始图积分：

  在第k轮次下，⽣成⼀张第k轮结果图（B_k），
  考虑B_k中任意位置(i, j)，该位置数据来⾃D_N-1中的位置(i, j)处得分最⾼的评
  分矩阵所对应的图像位置像素。统计B_k中的像素信息，计算出像素信息贡献
  最多的图像（C），为C增加与其当前轮次贡献像素信息等值的积分。
  
- 对齐原始图与⽣成图：
 
  N张图共轮换N个轮次，因此会得到N张轮次结果图
  （B_N）；同时原始每张图（O_N）都会拥有⼀个累计N轮次之后的积分结
  果。将B_N与O_N对齐：O_k为标准图 -> B_k为⽣成图。
  
- ⽣成融合图像：

  对原始图的积分进⾏归⼀化，得到权重分数T_N。融合图像
  N = T_1 * B_1 + T_2 * B_2 + … + T_N * B_N

#### 代码实现
- Cosine Similarity
```python
def cosine_similarity(image_A, image_B):
    """
    计算图像像素级别的余弦相似度图，**向量化版本，效率更高。**
    兼容 NumPy array 和 PyTorch Tensor 输入。
    特殊处理两个像素向量都为零向量的情况，返回相似度 1.0。
    使用 float32 数据类型进行计算，避免 uint8 溢出问题。

    参数:
    image_A (numpy.ndarray or torch.Tensor): 图像 A, 形状 (H, W, C) 或 (H, W)  (灰度图), dtype 可以是 uint8 或其他。
    image_B (numpy.ndarray or torch.Tensor): 图像 B, 形状 (H, W, C') 或 (H, W) (灰度图)。
                                图像 A 和 图像 B 需要 resize 到相同的 Height 和 Width。

    返回值:
    numpy.ndarray: 像素级别的余弦相似度图, 形状 (H, W), dtype=float32。
                   每个像素值表示对应位置的余弦相似度得分，范围在 [-1, 1] 之间。
                   **返回 NumPy array 格式的相似度图。**
    """

    # 1. 检查输入类型并转换为 NumPy array (保持与之前版本一致)
    if isinstance(image_A, torch.Tensor):
        image_A_np = image_A.cpu().numpy()
    else:
        image_A_np = image_A

    if isinstance(image_B, torch.Tensor):
        image_B_np = image_B.cpu().numpy()
    else:
        image_B_np = image_B

    # 2. 转换为 float32 数据类型 (向量化操作的关键)
    image_A_float = image_A_np.astype(np.double)
    image_B_float = image_B_np.astype(np.double)

    # 3. 向量化计算点积 (pixel-wise dot product)
    # 如果是彩色图像 (H, W, C)，则沿着通道维度 (axis=-1) 求和，得到 (H, W) 的点积图
    # 如果是灰度图像 (H, W)，则直接 element-wise 乘法，得到 (H, W) 的点积图
    dot_product_map = np.sum(image_A_float * image_B_float, axis=-1, keepdims=False)  # keepdims=False 去除维度为 1 的轴

    # 4. 向量化计算 L2 范数 (pixel-wise norm)
    # 同样，沿着通道维度 (axis=-1) 计算范数，得到 (H, W) 的范数图
    norm_A_map = np.linalg.norm(image_A_float, axis=-1, keepdims=False)
    norm_B_map = np.linalg.norm(image_B_float, axis=-1, keepdims=False)

    # 5. 向量化计算余弦相似度 (避免除以零)
    # 初始化相似度图为 0 (处理分母为零的情况)
    similarity_map_np = np.zeros_like(dot_product_map, dtype=np.double)

    # 找到分母不为零的位置 (即 norm_A * norm_B != 0 的位置)
    valid_denominator_mask = (norm_A_map * norm_B_map) != 0

    # 在分母不为零的位置，计算余弦相似度
    similarity_map_np[valid_denominator_mask] = (
        dot_product_map[valid_denominator_mask] / (norm_A_map[valid_denominator_mask] * norm_B_map[valid_denominator_mask])
    )

    # **[可选] 特殊处理：两个像素向量都为零向量的情况，设置为 1.0 (如果需要)**
    zero_vector_mask = (norm_A_map == 0) & (norm_B_map == 0)
    similarity_map_np[zero_vector_mask] = 1.0  #  向量化设置为 1.0

    return similarity_map_np  # 返回 NumPy array 格式的相似度图
```

- Cosine Similarity Fuse
```python
def cosine_similarity_fuse_v3(original_images, check=None):
    """
    Implements the Cosine Similarity Fuse (CSF) algorithm to fuse multiple images.
    Includes a check parameter to collect intermediate data for visualization.

    Args:
        original_images (list of numpy.ndarray): A list of N original images (O_N).
                                                Images should have the same height and width.
        check (bool or function, optional): If True or a function is provided, intermediate
                                            data will be collected and passed to the function.
                                            Defaults to None.

    Returns:
        numpy.ndarray: The fused image.
    """
    num_images = len(original_images)
    if num_images <= 1:
        return original_images[0] if original_images else None  # Handle cases with 0 or 1 image

    visualization_data = { # Initialize the dictionary to store visualization data
        'similarity_score_matrices_rounds': [],
        'contributing_pixel_counts_rounds': [],
        'round_result_images': [],
        'final_scores_and_weights': {}
    }

    round_result_images = []
    original_image_scores = {i: 0 for i in range(num_images)} # Initialize scores for each original image

    for k_standard_index in range(num_images):
        standard_image = original_images[k_standard_index]
        similarity_score_matrices = []
        compared_image_indices = [i for i in range(num_images) if i != k_standard_index]

        # 1. Generate Similarity Score Matrices (for round k)
        current_round_similarity_matrices = [] # Store similarity matrices for current round
        for compared_index in compared_image_indices:
            compared_image = original_images[compared_index]
            similarity_matrix = cosine_similarity(standard_image, compared_image)
            similarity_score_matrices.append(similarity_matrix)
            current_round_similarity_matrices.append(similarity_matrix) # Append to round list
        visualization_data['similarity_score_matrices_rounds'].append(current_round_similarity_matrices) # Store for visualization data

        # 2. Generate Round Image (B_k) and Original Image Scores
        round_result_image_Bk = np.zeros_like(standard_image, dtype=np.float32) # Initialize B_k
        contributing_image_counts = {i: 0 for i in compared_image_indices} # Count pixel contributions

        height, width = standard_image.shape[:2]
        for h in range(height):
            for w in range(width):
                max_similarity = -float('inf')
                best_source_image_index = -1

                for i, sim_matrix in enumerate(similarity_score_matrices):
                    current_similarity = sim_matrix[h, w]
                    if current_similarity > max_similarity:
                        max_similarity = current_similarity
                        best_source_image_index = compared_image_indices[i]

                if best_source_image_index != -1: # Should always be true in this logic but good to check
                    source_image = original_images[best_source_image_index]
                    round_result_image_Bk[h, w] = source_image[h, w]
                    contributing_image_counts[best_source_image_index] += 1

        round_result_images.append(round_result_image_Bk)
        visualization_data['round_result_images'].append(round_result_image_Bk) # Store round result image

        visualization_data['contributing_pixel_counts_rounds'].append(contributing_image_counts) # Store contributing counts for round

        # Find image C with most contribution and update score
        max_contribution_count = -1
        image_C_index = -1
        for index, count in contributing_image_counts.items():
            if count > max_contribution_count:
                max_contribution_count = count
                image_C_index = index

        if image_C_index != -1:
            original_image_scores[image_C_index] += max_contribution_count

    # 4. Generate Fused Image
    total_score = sum(original_image_scores.values())
    if total_score == 0:
        weights_normalized = [1.0 / num_images] * num_images # Default uniform weights if all scores are zero
    else:
        weights = [original_image_scores[i] for i in range(num_images)]
        weights_normalized = [w / total_score for w in weights] # Normalize scores to weights

    visualization_data['final_scores_and_weights']['original_image_scores'] = original_image_scores # Store final scores
    visualization_data['final_scores_and_weights']['weights_normalized'] = weights_normalized # Store normalized weights


    fused_image = np.zeros_like(original_images[0], dtype=np.float32)
    for i in range(num_images):
        fused_image += weights_normalized[i] * round_result_images[i]

    if check: # Check if check is True or a function is provided
        if callable(check):
            check(visualization_data) # Call the injected function with visualization data
        elif check == True:
            pass # If check=True and no function, you can add default data printing here if needed

    return fused_image.astype(original_images[0].dtype) # Return fused image with original dtype
```

#### 实验结果
![实验结果](../../../log/25_03_23/exp7.png)

#### 结论
CSF构建完毕，待集成接入主模型

---

### 实验八（通用数据集基准测试）
- 数据集：archive
- 标签："brick": 1, "rubber": 2, "concrete": 3, "wood": 4
- 训练集：601
- 验证集：257

|    指标     |  RGB-D  |   RGB   |
|:---------:|:-------:|:-------:|
|   Epoch   |   100   |   100   |
| TrainLoss | 6.1064  | 8.0953  |
| TestLoss  | 10.9563 | 29.1223 |
|    mAP    |  0.693  | 0.4562  |
|  mAP_50   | 0.7948  | 0.6351  |
|  mAP_75   | 0.7877  |  0.54   |

---

### 实验九（DSA模块阈值分割部分参数可学习化）

- DSA模块中的固定算法部分

  |    算法     |         函数名         |               参数                |
  |:---------:|:-------------------:|:-------------------------------:|
  | 深度频率直方图统计 | ```_calculate_depth_histogram```|        bins, value_range        |
  | 深度分布模式选择  |       ```_select_depth_distribution_modes```              | num_modes, prominence_threshold |
  | 深度区间窗口选择  |    ```_define_depth_interval_windows```                |        window_size_ratio        |
  |  深度区域生成   |          ```_generate_depth_region_masks```           |                -                |

- window_size_ratio 观察

  原始深度图
![original](../../../log/25_05_07/window_size_ratio_original.png)

  windows_size_ratio=0.01
![0.01](../../../log/25_05_07/window_size_ratio_0.01.png)

  windows_size_ratio=0.1
![0.1](../../../log/25_05_07/window_size_ratio_0.1.png)

  windows_size_ratio=0.05
![0.5](../../../log/25_05_07/window_size_ratio_0.5.png)

  windows_size_ratio=0.8
![0.8](../../../log/25_05_07/window_size_ratio_0.8.png)

  windows_size_ratio=1.0
![1.0](../../../log/25_05_07/window_size_ratio_1.0.png)

根据如上观测可知，采用不同的window_size_ratio阈值，对深度图像的阈值分割效果具有显著影响。
为了增强模型对于不同数据集（尤其是不同深度图像而言）的泛化能力，本实验考虑将```window_size_ratio```
定义为一个动态参数，依赖深度学习策略进行自适应调整。

为了实现上述需求，在```custom_model.py```中扩展一个神经网络模块用于捕获深度图像特征，并预测```window_size_ratio```。

```python
class RatioPredictor(nn.Module):
    """
    Predicts the window_size_ratio based on multi-scale input features.
    Takes a list of feature maps from a backbone.
    """
    def __init__(self, depth_channels_list: list[int]):
        """
        Args:
            depth_channels_list (list[int]): A list of channel counts for the
                                             depth feature maps at each scale
                                             (e.g., [96, 192, 384, 768]).
        """
        super().__init__()
        self.depth_channels_list = depth_channels_list
        self.num_scales = len(depth_channels_list)

        # Calculate the total number of features after pooling and concatenation
        # This will be the sum of the channel counts from all scales
        total_pooled_features = sum(depth_channels_list)

        # Define the fully connected layers that take the concatenated features
        self.fc_layers = nn.Sequential(
            nn.Linear(total_pooled_features, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 1) # Output a single scalar ratio per image in the batch
        )

        # Global Average Pooling layer to reduce spatial dimensions
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)

        # Constrain the output range of the parameter (ratio)
        self.output_min = 0.01 # Example minimum ratio
        self.output_max = 0.5  # Example maximum ratio
        self.sigmoid = nn.Sigmoid()

    def forward(self, depth_feature_maps: list[torch.Tensor]) -> torch.Tensor:
        """
        Args:
            depth_feature_maps (list[torch.Tensor]): A list of depth feature maps
                                                    from the depth backbone at different scales.
                                                    Shapes: [(B, C1, H1, W1), (B, C2, H2, W2), ...].

        Returns:
            torch.Tensor: Predicted window_size_ratio (B, 1).
        """
        assert len(depth_feature_maps) == self.num_scales, \
            f"Expected {self.num_scales} depth feature maps, but got {len(depth_feature_maps)}"

        pooled_features = []
        for i, feature_map in enumerate(depth_feature_maps):
            # Ensure channel count matches expected
            assert feature_map.shape[1] == self.depth_channels_list[i], \
                f"Expected {self.depth_channels_list[i]} channels for scale {i}, but got {feature_map.shape[1]}"

            # Apply Global Average Pooling
            pooled = self.global_avg_pool(feature_map) # Shape (B, C_i, 1, 1)

            # Squeeze spatial dimensions to get (B, C_i)
            pooled = pooled.squeeze(-1).squeeze(-1) # Shape (B, C_i)

            pooled_features.append(pooled)

        # Concatenate pooled features from all scales along the channel dimension (dim=1)
        # Resulting shape will be (B, sum(C_i))
        concatenated_features = torch.cat(pooled_features, dim=1)

        # Pass the concatenated features through the fully connected layers
        raw_ratio = self.fc_layers(concatenated_features) # Shape (B, 1)

        # Apply constraint to map output to [output_min, output_max]
        predicted_ratio = self.output_min + (self.output_max - self.output_min) * self.sigmoid(raw_ratio) # Shape (B, 1)

        return predicted_ratio
```



