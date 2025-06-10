import torch
import torch.nn as nn
from torchvision.models import resnet50
from pangaea.encoders.base import Encoder
from pathlib import Path
from logging import Logger
from einops import rearrange


class SpatialResNet(Encoder):
    def __init__(
        self,
        encoder_weights: str | Path,
        input_size: int,
        output_dim: int | list[int],
        input_bands: dict[str, list[str]],
        output_layers: int | list[int],
        input_channels: int = 10,
        pretrained: bool = True,
        download_url: str = "",
        aggregation_op: str = "mean",  # 可选的聚合方式，如 max 或 mean
    ) -> None:
        super().__init__(
            model_name="spatial_resnet",
            encoder_weights=encoder_weights,
            input_bands=input_bands,
            input_size=input_size,
            embed_dim=2048,  # ResNet50的最终输出维度
            output_dim=output_dim,  # 输出维度
            output_layers=output_layers,  # 传递给基类
            multi_temporal=True,  # 开启多时间步支持
            multi_temporal_output=False,
            pyramid_output=False,  # 默认没有金字塔输出
            download_url=download_url,  # 权重下载地址
        )

        self.output_layers = output_layers
        self.aggregation_op = aggregation_op  # 聚合方式：max 或 mean

        # 初始化ResNet
        self.resnet = resnet50(pretrained=pretrained)
        self.resnet.conv1 = nn.Conv2d(
            input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False
        )

        # 获取ResNet每一层的特征提取模块
        self.layer_mapping = nn.ModuleDict(
            {
                "0": nn.Sequential(self.resnet.conv1, self.resnet.bn1, self.resnet.relu, self.resnet.maxpool),
                "1": self.resnet.layer1,
                "2": self.resnet.layer2,
                "3": self.resnet.layer3,
                "4": self.resnet.layer4,
            }
        )

    def load_encoder_weights(self, logger: Logger) -> None:
        pretrained_model = torch.load(self.encoder_weights, map_location="cpu")
        visual_only_model = {}
        for k, v in pretrained_model.items():
            if k.startswith("visual."):
                visual_only_model[k.replace("visual.", "")] = v
        pretrained_model = visual_only_model

        k = pretrained_model.keys()
        pretrained_encoder = {}
        incompatible_shape = {}
        missing = {}
        for name, param in self.named_parameters():
            if name not in k:
                missing[name] = param.shape
            elif pretrained_model[name].shape != param.shape:
                incompatible_shape[name] = (param.shape, pretrained_model[name].shape)
            else:
                pretrained_encoder[name] = pretrained_model[name]

        self.load_state_dict(pretrained_encoder, strict=False)
        self.parameters_warning(missing, incompatible_shape, logger)

    def forward(self, x: dict[str, torch.Tensor]) -> list[torch.Tensor]:
        """
        Forward pass of the SpatialResNet encoder.

        Args:
            x (dict[str, torch.Tensor]): 输入包含键值对：{modality1: tensor1, ...}，如 x = {"optical": tensor1}。
            输入张量形状为 (B, C, T, H, W)（多时间步）。

        Returns:
            list[torch.Tensor]: 返回每一指定层的特征，形状为 (B, C, H, W)。
        """
        x = x["optical"]  # shape: (B, C, T, H, W)
        B, C, T, H, W = x.shape
        
        x = x.permute(0, 2, 1, 3, 4) # from (B, C, T, H, W) to (B, T, C, H, W)

        # 将时间维度和通道维度合并
        x = rearrange(x, "B T C H W -> (B T) C H W")

        outputs = []
        current_output = x

        # 逐层计算特征
        for idx, layer in self.layer_mapping.items():
            current_output = layer(current_output)
            if int(idx) in self.output_layers:
                outputs.append(current_output)

        # 将时间维度还原并进行聚合
        aggregated_outputs = []
        for feature in outputs:
            feature = rearrange(feature, "(B T) C H W -> B T C H W", B=B, T=T)
            
            # 在时间维度上进行聚合
            if self.aggregation_op == "max":
                feature = torch.amax(feature, dim=1)  # 聚合时间维度 (B, C, H, W)
            elif self.aggregation_op == "mean":
                feature = torch.mean(feature, dim=1)  # 聚合时间维度 (B, C, H, W)
            else:
                raise ValueError(f"Unsupported aggregation operation: {self.aggregation_op}")

            aggregated_outputs.append(feature)

        return aggregated_outputs

