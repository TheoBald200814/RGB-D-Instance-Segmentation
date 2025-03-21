
from typing import List, Optional
import random
import numpy as np
import torch
from torch import Tensor, nn
from transformers import Mask2FormerConfig, Mask2FormerModel, Mask2FormerForUniversalSegmentation
from transformers.models.mask2former.modeling_mask2former import Mask2FormerPixelLevelModule, \
    Mask2FormerForUniversalSegmentationOutput, Mask2FormerPixelLevelModuleOutput
from transformers.utils.backbone_utils import load_backbone


# 固定随机种子
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class CustomConfig(Mask2FormerConfig):
    model_type = "mask2former"

    def __init__(self, attribute=1, **kwargs):
        print("[CustomConfig] constructing...")
        self.attribute = attribute
        super().__init__(**kwargs)


class CustomMask2FormerModel(Mask2FormerModel):
    main_input_name = "pixel_values"
    def __init__(self, config, rgb_d=False):
        print("[CustomMask2FormerModel] constructing...")
        super().__init__(config)
        self.pixel_level_module = CustomMask2FormerPixelLevelModule(config, rgb_d=rgb_d)


class CustomMask2FormerForUniversalSegmentation(Mask2FormerForUniversalSegmentation):
    main_input_name = "pixel_values"
    config_class = CustomConfig

    def __init__(self, config, rgb_d=False):
        print("[CustomMask2FormerForUniversalSegmentation] constructing...")
        super().__init__(config)
        set_seed(42)
        self.model = CustomMask2FormerModel(config, rgb_d=rgb_d)


class CustomMask2FormerPixelLevelModule(Mask2FormerPixelLevelModule):
    main_input_name = "pixel_values"
    def __init__(self, config, rgb_d=False):
        print("[CustomMask2FormerPixelLevelModule] constructing...")
        super().__init__(config)
        self.rgb_d = rgb_d
        print(f"[CustomMask2FormerPixelLevelModule] rgb_d={self.rgb_d}")
        if self.rgb_d: # RGB-D
            self.depth_encoder = load_backbone(config)
            self.feature_fuser = FeatureFuser()
        self.color_encoder = load_backbone(config)

    def forward(self, pixel_values: Tensor, output_hidden_states: bool = False) -> Mask2FormerPixelLevelModuleOutput:
        if self.rgb_d: # RGB-D
            color_pixel_values = pixel_values[:, :3, :, :]
            depth_pixel_values = pixel_values[:, 3:6, :, :]
            color_feature_map = self.color_encoder(color_pixel_values).feature_maps
            depth_feature_map = self.depth_encoder(depth_pixel_values).feature_maps
            backbone_features = self.feature_fuser(color_feature_map, depth_feature_map)
        else: # RGB only
            backbone_features = self.color_encoder(pixel_values).feature_maps

        decoder_output = self.decoder(backbone_features, output_hidden_states=output_hidden_states)

        return Mask2FormerPixelLevelModuleOutput(
            encoder_last_hidden_state=backbone_features[-1],
            encoder_hidden_states=tuple(backbone_features) if output_hidden_states else None,
            decoder_last_hidden_state=decoder_output.mask_features,
            decoder_hidden_states=decoder_output.multi_scale_features,
        )


class FeatureFuser(nn.Module):
    def __init__(self):
        super().__init__()
        self.fuse_conv = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(192, 96, 1),
                nn.ReLU()
            ),
            nn.Sequential(
                nn.Conv2d(384, 192, 1),
                nn.ReLU()
            ),
            nn.Sequential(
                nn.Conv2d(768, 384, 1),
                nn.ReLU()
            ),
            nn.Sequential(
                nn.Conv2d(1536, 768, 1),
                nn.ReLU()
            )
        ])

    def forward(self, color_feature_map, depth_feature_map):
        assert len(color_feature_map) == len(depth_feature_map), \
            "the tuple length of color_feature_map and depth_feature_map should be the same"
        size_list = [
            torch.Size([1, 96, 64, 64]),
            torch.Size([1, 192, 32, 32]),
            torch.Size([1, 384, 16, 16]),
            torch.Size([1, 768, 8, 8])
        ]
        # TODO: DEBUG this assert
        # for c, d, size in zip(color_feature_map, depth_feature_map, size_list):
        #     assert c.shape == d.shape == size, \
        #         (f"the shape of color_feature_map and depth_feature_map should be the same, and both of them should be equal to the {size}."
        #          f"But color_feature_map.shape == {c.shape}, depth_feature_map.shape == {d.shape}")

        # merged_map = [torch.cat([color_feature_map[i], depth_feature_map[i]], dim=1) for i in range(len(depth_feature_map))]
        # fused_map = [self.fuse_conv[i](merged_map[i]) for i in range(len(merged_map))]
        merged_map = [torch.cat([c, d], dim=1) for c, d in zip(color_feature_map, depth_feature_map)]
        fused_map = [conv(m) for conv, m in zip(self.fuse_conv, merged_map)]

        return fused_map



