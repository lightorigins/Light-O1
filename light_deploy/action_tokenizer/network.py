import math

import torch.nn.functional as F
from torch import Tensor, nn


def _weight_norm(module: nn.Module) -> nn.Module:
    return nn.utils.parametrizations.weight_norm(module)


class CausalConv1d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        *,
        dilation: int = 1,
        bias: bool = True,
    ) -> None:
        super().__init__()
        self.pad = (kernel_size - 1) * dilation
        self.conv = _weight_norm(
            nn.Conv1d(
                in_channels,
                out_channels,
                kernel_size,
                dilation=dilation,
                bias=bias,
            )
        )

    def forward(self, values: Tensor) -> Tensor:
        return self.conv(F.pad(values, (self.pad, 0)))


class TemporalResidualBlock(nn.Module):
    def __init__(self, channels: int, kernel_size: int, dilation: int, residual_scale: float) -> None:
        super().__init__()
        self.conv1 = CausalConv1d(channels, channels, kernel_size, dilation=dilation, bias=False)
        self.conv2 = _weight_norm(nn.Conv1d(channels, channels, 1, bias=False))
        self.residual_scale = residual_scale

    def forward(self, values: Tensor) -> Tensor:
        hidden = self.conv1(F.silu(values))
        hidden = self.conv2(F.silu(hidden))
        return values + hidden * self.residual_scale


class ChannelResidualBlock(nn.Module):
    def __init__(self, channels: int, expansion_ratio: int, residual_scale: float) -> None:
        super().__init__()
        inner_channels = channels * expansion_ratio
        self.conv1 = _weight_norm(nn.Conv1d(channels, inner_channels, 1, bias=False))
        self.conv2 = _weight_norm(nn.Conv1d(inner_channels, channels, 1, bias=False))
        self.residual_scale = residual_scale

    def forward(self, values: Tensor) -> Tensor:
        hidden = self.conv1(F.silu(values))
        hidden = self.conv2(F.silu(hidden))
        return values + hidden * self.residual_scale


class DecoderStack(nn.Module):
    def __init__(
        self,
        *,
        channels: int,
        temporal_kernel_size: int,
        channel_blocks_per_stage: int,
        dilations: list[int],
        expansion_ratio: int,
    ) -> None:
        super().__init__()
        blocks_per_stage = 1 + channel_blocks_per_stage
        residual_scale = 1.0 / math.sqrt(len(dilations) * blocks_per_stage)
        blocks: list[nn.Module] = []
        for dilation in dilations:
            blocks.append(TemporalResidualBlock(channels, temporal_kernel_size, dilation, residual_scale))
            for _ in range(channel_blocks_per_stage):
                blocks.append(ChannelResidualBlock(channels, expansion_ratio, residual_scale))
        self.blocks = nn.ModuleList(blocks)

    def forward(self, values: Tensor) -> Tensor:
        for block in self.blocks:
            values = block(values)
        return values


class ActionDecoderNetwork(nn.Module):
    def __init__(self, config: dict) -> None:
        super().__init__()
        self.feature_size = int(config["feature_size"])
        hidden_dim = int(config["hidden_dim"])
        z_channels = int(config["z_channels"])
        num_stages = int(config["num_stages"])
        dilations = [int(value) for value in config["dilations"]]
        if len(dilations) != num_stages:
            raise ValueError(f"len(dilations)={len(dilations)} must equal num_stages={num_stages}")

        self.proj_in = _weight_norm(nn.Conv1d(z_channels, hidden_dim, 1, bias=False))
        self.stack = DecoderStack(
            channels=hidden_dim,
            temporal_kernel_size=int(config["temporal_kernel_size"]),
            channel_blocks_per_stage=int(config["channel_blocks_per_stage"]),
            dilations=dilations,
            expansion_ratio=int(config["expansion_ratio"]),
        )
        self.norm_out = nn.Identity()
        self.proj_out = CausalConv1d(
            hidden_dim,
            self.feature_size,
            int(config["proj_out_kernel_size"]),
            bias=True,
        )

    def forward(self, latents: Tensor) -> Tensor:
        values = self.proj_in(latents.squeeze(-1))
        values = self.stack(values)
        values = F.silu(self.norm_out(values))
        values = self.proj_out(values)
        return values.permute(0, 2, 1).unsqueeze(1)
