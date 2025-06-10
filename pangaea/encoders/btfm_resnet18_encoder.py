from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18
from logging import Logger

from pangaea.encoders.base import Encoder

class TemporalResnet18(Encoder):
    def __init__(
        self,
        encoder_weights: str | Path,
        output_dim: int,
        input_bands: dict[str, list[str]],
        output_layers: int | list[int],
        input_size: int = 1,
        embed_dim: int = 512,
    ) -> None:
        super().__init__(
            model_name="btfm_resnet18_encoder",
            encoder_weights=encoder_weights,
            input_bands=input_bands,
            input_size=input_size,
            embed_dim=embed_dim,
            output_layers=output_layers,
            output_dim=output_dim,
            multi_temporal=True,
            multi_temporal_output=False,
            pyramid_output=False,
            download_url=None,
        )
        # 加载预训练的ResNet-18模型
        self.resnet = resnet18(pretrained=True)
        # 修改第一层以适应输入形状 (1, 96, 10)
        self.resnet.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.output_dim = output_dim
        # 修改最后一层输出
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, output_dim)
    
    def freeze(self):
        for param in self.parameters():
            param.requires_grad = False

    def load_encoder_weights(self, logger: Logger) -> None:
        pretrained_model = torch.load(self.encoder_weights, map_location="cpu")
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
        """Foward pass of the encoder.

        Args:
            x (dict[str, torch.Tensor]): encoder's input structured as a dictionary:
            x = {modality1: tensor1, modality2: tensor2, ...}, e.g. x = {"optical": tensor1, "sar": tensor2}.
            If the encoder is multi-temporal (self.multi_temporal==True), input tensor shape is (B C T H W) with C the
            number of bands required by the encoder for the given modality and T the number of time steps. If the
            encoder is not multi-temporal, input tensor shape is (B C H W) with C the number of bands required by the
            encoder for the given modality.

        Returns:
            list[torch.Tensor]: list of the embeddings for each modality. For single-temporal encoders, the list's
            elements are of shape (B, embed_dim, H', W'). For multi-temporal encoders, the list's elements are of shape
            (B, C', T, H', W') with T the number of time steps if the encoder does not have any time-merging strategy,
            else (B, C', H', W') if the encoder has a time-merging strategy (where C'==self.output_dim).
        """
        # 输入需要是 (batch_size, 1, 96, 10)，但是目前x是 (batch_size, 10, 96)
        x = x["optical"].unsqueeze(1).permute(0, 1, 3, 2)
        x = self.resnet(x)
        return [x]