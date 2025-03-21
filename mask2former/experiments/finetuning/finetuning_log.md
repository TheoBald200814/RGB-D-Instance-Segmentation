# 基于Mask2Former模型的微调实验
## 实验清单
| ID |               Code Version               | Dataset | Train Set | Valid Set | Depth |                          Ckpt Path                          | Epoch |     架构简述     |
|:--:|:----------------------------------------:|:-------:|:---------:|:---------:|:-----:|:-----------------------------------------------------------:|:-----:|:------------:|
| 1  | e5f88dabbed106f9c00389970155ceec05cf8483 | coco82  |    57     |    25     | False |    ```mask2former/checkpoints/remote/300_epoch_result```    |  300  |   标准RGB数据流   |
| 2  | db9bbc22b210a8454a675bd3ed89522ce822df88 | coco82  |    57     |    25     | True  | ```mask2former/checkpoints/remote/200_epoch_depth_result``` |  300  | 初探RGB-D融合数据流 | 
