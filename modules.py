"""
Reusable building blocks for MS-DBNet.

This module provides the five core components used by the MS-DBNet model:

- Conv2dWithConstraint:      Max-norm constrained 2-D convolution.
- LinearWithConstraint:      Max-norm constrained linear layer.
- ChannelTimeAttention (CTA): Hybrid channel–time attention mechanism.
- MultiScaleTemporalConv:    Multi-scale temporal convolution (MSDB Block 3).
- DilatedMultiScaleConv:     Dilated multi-scale convolution  (MSDB Block 4).

Reference:
    Shi, F. (2025). MS-DBNet: A heterogeneous temporal convolutional network
    for robust subject-specific cross-session motor imagery decoding.
    Proc. of SPIE Vol. 13940, 139400M.  DOI: 10.1117/12.3092280
"""

import math
import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
#  Constrained layers
# ---------------------------------------------------------------------------

class Conv2dWithConstraint(nn.Conv2d):
    """Conv2d with max-norm weight constraint applied before each forward pass."""

    def __init__(self, *args, doWeightNorm=True, max_norm=1, **kwargs):
        self.max_norm = max_norm
        self.doWeightNorm = doWeightNorm
        super(Conv2dWithConstraint, self).__init__(*args, **kwargs)

    def forward(self, x):
        if self.doWeightNorm:
            self.weight.data = torch.renorm(
                self.weight.data, p=2, dim=0, maxnorm=self.max_norm
            )
        return super(Conv2dWithConstraint, self).forward(x)


class LinearWithConstraint(nn.Linear):
    """Linear layer with max-norm weight constraint applied before each forward pass."""

    def __init__(self, *args, doWeightNorm=True, max_norm=1, **kwargs):
        self.max_norm = max_norm
        self.doWeightNorm = doWeightNorm
        super(LinearWithConstraint, self).__init__(*args, **kwargs)

    def forward(self, x):
        if self.doWeightNorm:
            self.weight.data = torch.renorm(
                self.weight.data, p=2, dim=0, maxnorm=self.max_norm
            )
        return super(LinearWithConstraint, self).forward(x)


# ---------------------------------------------------------------------------
#  Channel-Time Attention (CTA)
# ---------------------------------------------------------------------------

class ChannelTimeAttention(nn.Module):
    """Hybrid Channel-Time Attention (CTA) module for EEG feature maps.

    Combines an ECA-style efficient channel attention branch with a
    spatial-pooling-based temporal attention branch.  The two branches are
    fused via learnable weights so the network can adaptively balance
    channel-wise and temporal emphasis.

    Args:
        channels:        Number of input feature-map channels.
        reduction_ratio: (kept for compatibility; not used internally).
        kernel_size:     Temporal convolution kernel size for the time branch.
    """

    def __init__(self, channels, reduction_ratio=8, kernel_size=7):
        super(ChannelTimeAttention, self).__init__()

        # --- Channel attention (ECA-style) ---
        t = int(abs((math.log2(channels) + 1) / 2))
        k_size = t if t % 2 else t + 1

        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv1d(1, 1, kernel_size=k_size,
                      padding=(k_size - 1) // 2, bias=False),
            nn.Sigmoid()
        )

        # --- Temporal attention (avg + max pooling → conv) ---
        self.time_conv = nn.Conv2d(2, 1, kernel_size=(1, kernel_size),
                                   padding=(0, kernel_size // 2), bias=False)
        self.time_sigmoid = nn.Sigmoid()

        # --- Learnable fusion weights ---
        self.fusion_weights = nn.Parameter(torch.FloatTensor([0.5, 0.5]))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        """
        Args:
            x: Tensor of shape ``[B, C, 1, T]``.
        Returns:
            Tensor of the same shape with channel-time attention applied.
        """
        # Channel attention
        y_ch = self.channel_attention[0](x)                        # [B, C, 1, 1]
        y_ch = y_ch.squeeze(-1).squeeze(-1).unsqueeze(1)           # [B, 1, C]
        y_ch = self.channel_attention[1](y_ch)                     # [B, 1, C]
        y_ch = y_ch.squeeze(1).unsqueeze(-1).unsqueeze(-1)         # [B, C, 1, 1]
        y_ch = self.channel_attention[2](y_ch)                     # [B, C, 1, 1]

        # Temporal attention
        avg_pool = torch.mean(x, dim=1, keepdim=True)             # [B, 1, 1, T]
        max_pool, _ = torch.max(x, dim=1, keepdim=True)           # [B, 1, 1, T]
        y_time = self.time_sigmoid(
            self.time_conv(torch.cat([avg_pool, max_pool], dim=1)) # [B, 1, 1, T]
        )

        # Adaptive fusion
        w1, w2 = self.sigmoid(self.fusion_weights)
        norm = w1 + w2
        weights = torch.stack([w1, w2]) / norm

        return (weights[0] * x * y_ch.expand_as(x)
                + weights[1] * x * y_time.expand_as(x))


# ---------------------------------------------------------------------------
#  Multi-Scale Temporal Convolution  (MSDB Block 3)
# ---------------------------------------------------------------------------

class MultiScaleTemporalConv(nn.Module):
    """Parallel multi-kernel temporal convolution with CTA fusion.

    Multiple branches with different temporal kernel sizes process the input
    in parallel; their outputs are concatenated along the channel dimension
    and refined by a :class:`ChannelTimeAttention` module.

    Corresponds to **Block 3** of the MSDB branch in the paper
    (kernel set ``{3, 7, 15, 31}`` by default).

    Args:
        in_channels:  Number of input channels.
        out_channels: Total number of output channels (split across branches).
        kernel_sizes: Temporal kernel sizes for each parallel branch.
    """

    def __init__(self, in_channels, out_channels,
                 kernel_sizes=(3, 7, 15, 31)):
        super(MultiScaleTemporalConv, self).__init__()

        num_branches = len(kernel_sizes)
        base = out_channels // num_branches
        extra = out_channels % num_branches

        self.branches = nn.ModuleList()
        for i, k in enumerate(kernel_sizes):
            ch = base + (extra if i == num_branches - 1 else 0)
            self.branches.append(nn.Sequential(
                nn.Conv2d(in_channels, ch, (1, k),
                          padding='same', bias=False),
                nn.BatchNorm2d(ch),
            ))

        self.attention = ChannelTimeAttention(out_channels)

    def forward(self, x):
        return self.attention(
            torch.cat([b(x) for b in self.branches], dim=1)
        )


# ---------------------------------------------------------------------------
#  Dilated Multi-Scale Convolution  (MSDB Block 4)
# ---------------------------------------------------------------------------

class DilatedMultiScaleConv(nn.Module):
    """Parallel dilated temporal convolution with CTA fusion.

    All branches share the same kernel size but use different dilation rates,
    allowing the module to capture long-range temporal dependencies without
    increasing the number of parameters.  Outputs are concatenated and
    refined by a :class:`ChannelTimeAttention` module.

    Corresponds to **Block 4** of the MSDB branch in the paper
    (dilation set ``{1, 2, 4, 8}`` by default).

    Args:
        in_channels:  Number of input channels.
        out_channels: Total number of output channels (split across branches).
        kernel_size:  Shared temporal kernel size for every branch.
        dilations:    Dilation rates for each parallel branch.
    """

    def __init__(self, in_channels, out_channels,
                 kernel_size=3, dilations=(1, 2, 4, 8)):
        super(DilatedMultiScaleConv, self).__init__()

        num_branches = len(dilations)
        base = out_channels // num_branches
        extra = out_channels % num_branches

        self.branches = nn.ModuleList()
        for i, d in enumerate(dilations):
            ch = base + (extra if i == num_branches - 1 else 0)
            self.branches.append(nn.Sequential(
                nn.Conv2d(in_channels, ch, (1, kernel_size),
                          padding='same', dilation=(1, d), bias=False),
                nn.BatchNorm2d(ch),
                nn.ELU(),
            ))

        self.attention = ChannelTimeAttention(out_channels, reduction_ratio=4)

    def forward(self, x):
        return self.attention(
            torch.cat([b(x) for b in self.branches], dim=1)
        )
