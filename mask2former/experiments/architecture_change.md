# 基于 Mask2Former 的架构调整实验
## Mask2Former 模型概述
### 模型架构
- Backbone(Swin): 主干网络，用于提取图像特征，生成feature map
- Pixel decoder: 像素级decoder，对feature map进行上采样操作，并实现多尺度特称embedding操作
- Transformer decoder: 初始化N个queries，并基于masked attention机制和FFN处理多尺度pixel-level embedding数据，用于增强N个queries
- Detect header: 基于MLP生成pixel-level masks and class of each mask。后处理（丢弃无效queries及其对应的pixel-level mask）

### 模型用途
- 基于虾类图像数据，学习虾及其脏器图像区域特征，执行较为精确的实例分割任务。

## 2024/12/19 实验记录
### 实验清单
- 实验一：测试Mask2Former各模块之间的数据传递格式(例如Backbone的input格式和output格式)
- 实验二：准备小规模实验数据集，作为对照实验(使用指定seed训练)，训练标准Mask2Former模型，并得到validation数据
- 实验三：继承标准模型使用的Config类、Backbone(Swin)类、Pixel decoder类、Transformer类，使用上述seed进行训练，验证得出的validation数据是否一致

### 实验路径
``` 
/Users/theobald/Documents/code_lib/python_lib/shrimpDetection/mask2former/experiments/24_12_19
```

### 实验结果
- 实验二
  - 24_12_19_01
    ```
    ***** test metrics *****
    epoch                   =        1.0
    test_loss               =    21.3779
    test_map                =     0.2113
    test_map_50             =     0.3012
    test_map_75             =     0.3012
    test_map_background     =       -1.0
    test_map_large          =       -1.0
    test_map_medium         =     0.2649
    test_map_shrimp         =     0.2113
    test_map_small          =        0.2
    test_mar_1              =      0.375
    test_mar_10             =      0.525
    test_mar_100            =      0.725
    test_mar_100_background =       -1.0
    test_mar_100_shrimp     =      0.725
    test_mar_large          =       -1.0
    test_mar_medium         =        0.7
    test_mar_small          =        0.8
    test_runtime            = 0:00:03.31
    test_samples_per_second =      0.905
    test_steps_per_second   =      0.302
    ```
  - 24_12_19_02
    ```
    ***** test metrics *****
    epoch                   =        1.0
    test_loss               =    21.3779
    test_map                =     0.2113
    test_map_50             =     0.3012
    test_map_75             =     0.3012
    test_map_background     =       -1.0
    test_map_large          =       -1.0
    test_map_medium         =     0.2649
    test_map_shrimp         =     0.2113
    test_map_small          =        0.2
    test_mar_1              =      0.375
    test_mar_10             =      0.525
    test_mar_100            =      0.725
    test_mar_100_background =       -1.0
    test_mar_100_shrimp     =      0.725
    test_mar_large          =       -1.0
    test_mar_medium         =        0.7
    test_mar_small          =        0.8
    test_runtime            = 0:00:03.15
    test_samples_per_second =      0.949
    test_steps_per_second   =      0.316
    ```
  - 24_12_19_03
    ```
    ***** test metrics *****
    epoch                   =        1.0
    test_loss               =    21.3779
    test_map                =     0.2113
    test_map_50             =     0.3012
    test_map_75             =     0.3012
    test_map_background     =       -1.0
    test_map_large          =       -1.0
    test_map_medium         =     0.2649
    test_map_shrimp         =     0.2113
    test_map_small          =        0.2
    test_mar_1              =      0.375
    test_mar_10             =      0.525
    test_mar_100            =      0.725
    test_mar_100_background =       -1.0
    test_mar_100_shrimp     =      0.725
    test_mar_large          =       -1.0
    test_mar_medium         =        0.7
    test_mar_small          =        0.8
    test_runtime            = 0:00:03.33
    test_samples_per_second =        0.9
    test_steps_per_second   =        0.3
    ```
  - 保证训练及验证结果一致的调整：set_seed(seed=42)、"dataloader_num_workers": 0
