"""
MS-DBNet: Multi-Scale Dual-Branch Network for motor imagery EEG decoding.

This module implements three network classes:

- **SSTB**     – Single-Scale Temporal Branch  (fine-grained local features).
- **MSDB**     – Multi-Scale Dilated Branch    (global contextual features).
- **MSDBNet**  – Full dual-branch model that fuses SSTB and MSDB outputs.

All default hyper-parameters match the configuration reported in the paper
(Table II), so ``MSDBNet(nChan, nTime, nClass)`` reproduces the published
architecture out of the box.

Reference:
    Shi, F. (2025). MS-DBNet: A heterogeneous temporal convolutional network
    for robust subject-specific cross-session motor imagery decoding.
    Proc. of SPIE Vol. 13940, 139400M.  DOI: 10.1117/12.3092280
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from modules import (
    Conv2dWithConstraint,
    LinearWithConstraint,
    ChannelTimeAttention,
    MultiScaleTemporalConv,
    DilatedMultiScaleConv,
)


# ===================================================================
#  Single-Scale Temporal Branch (SSTB)
# ===================================================================

class SSTB(nn.Module):
    """Single-Scale Temporal Branch.

    Combines spatial depthwise convolution with single-scale temporal
    depthwise separable convolution to capture fine-grained local temporal
    patterns at high resolution.

    Architecture (see paper Table II – SSTB column):
        Block 1  – Temporal Conv (F1 filters, kernel 1×32) + BN
        Block 2  – Spatial Depthwise Conv (D=8) + BN + ELU + AvgPool + Dropout + CTA
        Block 3  – Depthwise Separable Conv (F3=32, kernel 1×32) + BN + ELU + AvgPool + Dropout + CTA
        Block 4  – Standard Conv (F4=32, kernel 1×3) + BN + ELU + AvgPool + Dropout + CTA
        Flatten → FC Classifier

    Args:
        nChan:               Number of EEG channels (C).
        nTime:               Number of time points (T).
        nClass:              Number of output classes.
        F1:                  Number of temporal filters in Block 1.
        D:                   Depth multiplier for spatial depthwise convolution.
        conv1_tkern_len:     Temporal kernel length in Block 1.
        pool2_tfactor:       Average-pooling factor after Block 2.
        b3_out_channels:     Output channels of Block 3 (F3).
        b3_dw_tkern_len:     Depthwise temporal kernel length in Block 3.
        b3_pool_tfactor:     Average-pooling factor after Block 3.
        b4_out_channels:     Output channels of Block 4 (F4).
        b4_std_conv_tkern_len: Temporal kernel length in Block 4.
        b4_pool_tfactor:     Average-pooling factor after Block 4.
        dropoutRate:         Dropout probability.
        norm_rate:           Max-norm constraint for the classifier.
        use_attention:       Whether to apply CTA after each block.
    """

    def __init__(self, nChan, nTime, nClass,
                 F1=16, D=8, conv1_tkern_len=32, pool2_tfactor=8,
                 b3_out_channels=32, b3_dw_tkern_len=32, b3_pool_tfactor=4,
                 b4_out_channels=32, b4_std_conv_tkern_len=3, b4_pool_tfactor=1,
                 dropoutRate=0.5, norm_rate=0.25, use_attention=True):
        super(SSTB, self).__init__()
        self.nChan = nChan
        self.nTime = nTime
        self.nClass = nClass
        self.F1 = F1
        self.D = D
        self.dropoutRate = dropoutRate
        self.norm_rate = norm_rate
        self.use_attention = use_attention

        F2 = F1 * D  # channels after spatial depthwise conv

        # Block 1 – Temporal convolution
        self.conv1 = nn.Conv2d(1, F1, (1, conv1_tkern_len),
                               padding='same', bias=False)
        self.bn1 = nn.BatchNorm2d(F1)

        # Block 2 – Spatial depthwise convolution
        self.depthwise_conv = Conv2dWithConstraint(
            F1, F2, (nChan, 1), groups=F1, bias=False, max_norm=1.0)
        self.bn2 = nn.BatchNorm2d(F2)
        self.elu2 = nn.ELU()
        self.pool2 = nn.AvgPool2d((1, pool2_tfactor))
        self.dropout2 = nn.Dropout(dropoutRate)
        self.cta2 = ChannelTimeAttention(F2)

        # Block 3 – Temporal depthwise separable convolution
        self.dw_conv3 = nn.Conv2d(F2, F2, (1, b3_dw_tkern_len),
                                  padding='same', groups=F2, bias=False)
        self.bn3_dw = nn.BatchNorm2d(F2)
        self.elu3_dw = nn.ELU()
        self.pw_conv3 = nn.Conv2d(F2, b3_out_channels, (1, 1), bias=False)
        self.bn3_pw = nn.BatchNorm2d(b3_out_channels)
        self.elu3_pw = nn.ELU()
        self.pool3 = nn.AvgPool2d((1, b3_pool_tfactor))
        self.dropout3 = nn.Dropout(dropoutRate)
        self.cta3 = ChannelTimeAttention(b3_out_channels)

        # Block 4 – Standard temporal convolution
        self.conv4 = nn.Conv2d(b3_out_channels, b4_out_channels,
                               (1, b4_std_conv_tkern_len),
                               padding='same', bias=False)
        self.bn4 = nn.BatchNorm2d(b4_out_channels)
        self.elu4 = nn.ELU()
        self.pool4 = nn.AvgPool2d((1, b4_pool_tfactor))
        self.dropout4 = nn.Dropout(dropoutRate)
        self.cta4 = ChannelTimeAttention(b4_out_channels)

        self.flatten = nn.Flatten()

    # ------------------------------------------------------------------

    def extract_features(self, x):
        """Run the convolutional backbone and return the flattened feature
        vector **before** the classifier head.

        Args:
            x: ``[B, 1, C, T]`` or ``[B, C, T]``.
        Returns:
            Flattened feature tensor ``[B, D_feat]``.
        """
        if x.ndim == 3:
            x = x.unsqueeze(1)

        # Block 1
        out = self.bn1(self.conv1(x))
        # Block 2
        out = self.dropout2(self.pool2(self.elu2(self.bn2(
            self.depthwise_conv(out)))))
        if self.use_attention:
            out = self.cta2(out)
        # Block 3
        out = self.elu3_dw(self.bn3_dw(self.dw_conv3(out)))
        out = self.dropout3(self.pool3(self.elu3_pw(self.bn3_pw(
            self.pw_conv3(out)))))
        if self.use_attention:
            out = self.cta3(out)
        # Block 4
        out = self.dropout4(self.pool4(self.elu4(self.bn4(
            self.conv4(out)))))
        if self.use_attention:
            out = self.cta4(out)

        return self.flatten(out)

    def forward(self, x):
        out_flat = self.extract_features(x)

        if not hasattr(self, 'fc'):
            self.fc = LinearWithConstraint(
                out_flat.shape[1], self.nClass, max_norm=self.norm_rate)
            self.fc.to(out_flat.device)

        return F.log_softmax(self.fc(out_flat), dim=1)


# ===================================================================
#  Multi-Scale Dilated Branch (MSDB)
# ===================================================================

class MSDB(nn.Module):
    """Multi-Scale Dilated Branch.

    Integrates multi-scale temporal convolution and dilated multi-scale
    convolution to systematically expand the receptive field, learning
    long-range contextual dependencies tolerant to temporal variations.

    Architecture (see paper Table II – MSDB column):
        Block 1  – Temporal Conv (F'1=16, kernel 1×32) + BN
        Block 2  – Spatial Depthwise Conv (D'=2) + BN + ELU + AvgPool + Dropout + CTA
        Block 3  – MultiScaleTemporalConv (F'3=64, kernels {3,7,15,31}) + BN + ELU + AvgPool + Dropout + CTA
        Block 4  – DilatedMultiScaleConv  (F'4=64, dilations {1,2,4,8}) + BN + ELU + AvgPool + Dropout + CTA
        Flatten → FC Classifier

    Args:
        nChan:            Number of EEG channels (C).
        nTime:            Number of time points (T).
        nClass:           Number of output classes.
        F1:               Number of temporal filters in Block 1.
        D:                Depth multiplier for spatial depthwise convolution.
        conv1_tkern_len:  Temporal kernel length in Block 1.
        pool2_tfactor:    Average-pooling factor after Block 2.
        b3_out_channels:  Output channels of Block 3 (F'3).
        b3_pool_tfactor:  Average-pooling factor after Block 3.
        b4_out_channels:  Output channels of Block 4 (F'4).
        b4_pool_tfactor:  Average-pooling factor after Block 4.
        dropoutRate:      Dropout probability.
        norm_rate:        Max-norm constraint for the classifier.
        use_attention:    Whether to apply CTA after each block.
    """

    def __init__(self, nChan, nTime, nClass,
                 F1=16, D=2, conv1_tkern_len=32, pool2_tfactor=8,
                 b3_out_channels=64, b3_pool_tfactor=4,
                 b4_out_channels=64, b4_pool_tfactor=1,
                 dropoutRate=0.5, norm_rate=0.25, use_attention=True):
        super(MSDB, self).__init__()
        self.nChan = nChan
        self.nTime = nTime
        self.nClass = nClass
        self.F1 = F1
        self.D = D
        self.dropoutRate = dropoutRate
        self.norm_rate = norm_rate
        self.use_attention = use_attention

        F2 = F1 * D

        # Block 1 – Temporal convolution
        self.conv1 = nn.Conv2d(1, F1, (1, conv1_tkern_len),
                               padding='same', bias=False)
        self.bn1 = nn.BatchNorm2d(F1)

        # Block 2 – Spatial depthwise convolution
        self.depthwise_conv = Conv2dWithConstraint(
            F1, F2, (nChan, 1), groups=F1, bias=False, max_norm=1.0)
        self.bn2 = nn.BatchNorm2d(F2)
        self.elu2 = nn.ELU()
        self.pool2 = nn.AvgPool2d((1, pool2_tfactor))
        self.dropout2 = nn.Dropout(dropoutRate)
        self.cta2 = ChannelTimeAttention(F2)

        # Block 3 – Multi-scale temporal convolution
        self.multi_scale_conv3 = MultiScaleTemporalConv(
            F2, b3_out_channels, kernel_sizes=[3, 7, 15, 31])
        self.bn3 = nn.BatchNorm2d(b3_out_channels)
        self.elu3 = nn.ELU()
        self.pool3 = nn.AvgPool2d((1, b3_pool_tfactor))
        self.dropout3 = nn.Dropout(dropoutRate)
        self.cta3 = ChannelTimeAttention(b3_out_channels)

        # Block 4 – Dilated multi-scale convolution
        self.dilated_conv4 = DilatedMultiScaleConv(
            b3_out_channels, b4_out_channels,
            kernel_size=3, dilations=[1, 2, 4, 8])
        self.bn4 = nn.BatchNorm2d(b4_out_channels)
        self.elu4 = nn.ELU()
        self.pool4 = nn.AvgPool2d((1, b4_pool_tfactor))
        self.dropout4 = nn.Dropout(dropoutRate)
        self.cta4 = ChannelTimeAttention(b4_out_channels)

        self.flatten = nn.Flatten()

    # ------------------------------------------------------------------

    def extract_features(self, x):
        """Run the convolutional backbone and return the flattened feature
        vector **before** the classifier head.

        Args:
            x: ``[B, 1, C, T]`` or ``[B, C, T]``.
        Returns:
            Flattened feature tensor ``[B, D_feat]``.
        """
        if x.ndim == 3:
            x = x.unsqueeze(1)

        # Block 1
        out = self.bn1(self.conv1(x))
        # Block 2
        out = self.dropout2(self.pool2(self.elu2(self.bn2(
            self.depthwise_conv(out)))))
        if self.use_attention:
            out = self.cta2(out)
        # Block 3
        out = self.dropout3(self.pool3(self.elu3(self.bn3(
            self.multi_scale_conv3(out)))))
        if self.use_attention:
            out = self.cta3(out)
        # Block 4
        out = self.dropout4(self.pool4(self.elu4(self.bn4(
            self.dilated_conv4(out)))))
        if self.use_attention:
            out = self.cta4(out)

        return self.flatten(out)

    def forward(self, x):
        out_flat = self.extract_features(x)

        if not hasattr(self, 'fc'):
            self.fc = LinearWithConstraint(
                out_flat.shape[1], self.nClass, max_norm=self.norm_rate)
            self.fc.to(out_flat.device)

        return F.log_softmax(self.fc(out_flat), dim=1)


# ===================================================================
#  MS-DBNet  (full model)
# ===================================================================

class MSDBNet(nn.Module):
    """Multi-Scale Dual-Branch Network (MS-DBNet).

    Processes EEG signals through a parallel heterogeneous dual-branch
    architecture: the **SSTB** captures fine-grained local temporal patterns
    while the **MSDB** captures robust global contextual features.  Their
    flattened outputs are concatenated and fed into a fully-connected
    classifier.

    All default hyper-parameters reproduce the architecture described in
    Table II of the paper:

        >>> model = MSDBNet(nChan=22, nTime=1000, nClass=4)

    Args:
        nChan:  Number of EEG channels.
        nTime:  Number of time points.
        nClass: Number of output classes.
        use_attention:         Enable/disable CTA in both branches.
        dropoutRate_branches:  Default dropout rate shared by both branches.
        norm_rate_branches:    Default max-norm constraint shared by both branches.
        sstb_*:  Hyper-parameters forwarded to :class:`SSTB`.
        msdb_*:  Hyper-parameters forwarded to :class:`MSDB`.
        fused_fc_hidden_dim:   If set, use a two-layer classifier with this
                               hidden size; otherwise use a single linear layer.
        fused_fc_dropoutRate:  Dropout in the fused classifier.
        fused_fc_norm_rate:    Max-norm constraint for the fused classifier.
    """

    def __init__(self, nChan, nTime, nClass, use_attention=True,
                 dropoutRate_branches=0.5,
                 norm_rate_branches=0.25,
                 # --- SSTB parameters ---
                 sstb_F1=16, sstb_D=8,
                 sstb_conv1_tkern_len=32, sstb_pool2_tfactor=8,
                 sstb_b3_out_channels=32, sstb_b3_dw_tkern_len=32,
                 sstb_b3_pool_tfactor=4,
                 sstb_b4_out_channels=32, sstb_b4_std_conv_tkern_len=3,
                 sstb_b4_pool_tfactor=1,
                 sstb_dropoutRate=None, sstb_norm_rate=None,
                 # --- MSDB parameters ---
                 msdb_F1=16, msdb_D=2,
                 msdb_conv1_tkern_len=32, msdb_pool2_tfactor=8,
                 msdb_b3_out_channels=64, msdb_b3_pool_tfactor=4,
                 msdb_b4_out_channels=64, msdb_b4_pool_tfactor=1,
                 msdb_dropoutRate=None, msdb_norm_rate=None,
                 # --- Fused classifier parameters ---
                 fused_fc_hidden_dim=None,
                 fused_fc_dropoutRate=0.5,
                 fused_fc_norm_rate=0.25):
        super(MSDBNet, self).__init__()
        self.nClass = nClass

        _sstb_dr = sstb_dropoutRate if sstb_dropoutRate is not None else dropoutRate_branches
        _sstb_nr = sstb_norm_rate   if sstb_norm_rate   is not None else norm_rate_branches
        _msdb_dr = msdb_dropoutRate if msdb_dropoutRate is not None else dropoutRate_branches
        _msdb_nr = msdb_norm_rate   if msdb_norm_rate   is not None else norm_rate_branches

        self.sstb = SSTB(
            nChan=nChan, nTime=nTime, nClass=nClass,
            use_attention=use_attention,
            F1=sstb_F1, D=sstb_D,
            conv1_tkern_len=sstb_conv1_tkern_len,
            pool2_tfactor=sstb_pool2_tfactor,
            b3_out_channels=sstb_b3_out_channels,
            b3_dw_tkern_len=sstb_b3_dw_tkern_len,
            b3_pool_tfactor=sstb_b3_pool_tfactor,
            b4_out_channels=sstb_b4_out_channels,
            b4_std_conv_tkern_len=sstb_b4_std_conv_tkern_len,
            b4_pool_tfactor=sstb_b4_pool_tfactor,
            dropoutRate=_sstb_dr, norm_rate=_sstb_nr,
        )

        self.msdb = MSDB(
            nChan=nChan, nTime=nTime, nClass=nClass,
            use_attention=use_attention,
            F1=msdb_F1, D=msdb_D,
            conv1_tkern_len=msdb_conv1_tkern_len,
            pool2_tfactor=msdb_pool2_tfactor,
            b3_out_channels=msdb_b3_out_channels,
            b3_pool_tfactor=msdb_b3_pool_tfactor,
            b4_out_channels=msdb_b4_out_channels,
            b4_pool_tfactor=msdb_b4_pool_tfactor,
            dropoutRate=_msdb_dr, norm_rate=_msdb_nr,
        )

        self.fused_fc_hidden_dim = fused_fc_hidden_dim
        self.fused_fc_dropoutRate = fused_fc_dropoutRate
        self.fused_fc_norm_rate = fused_fc_norm_rate
        self.fused_fc = None

    # ------------------------------------------------------------------

    def forward(self, x):
        if x.ndim == 3:
            x = x.unsqueeze(1)

        feat_sstb = self.sstb.extract_features(x)
        feat_msdb = self.msdb.extract_features(x)
        fused = torch.cat((feat_sstb, feat_msdb), dim=1)

        # Lazily build the classifier on first forward pass
        if self.fused_fc is None:
            n_in = fused.shape[1]
            if self.fused_fc_hidden_dim and self.fused_fc_hidden_dim > 0:
                self.fused_fc = nn.Sequential(
                    LinearWithConstraint(n_in, self.fused_fc_hidden_dim,
                                         max_norm=self.fused_fc_norm_rate),
                    nn.ELU(),
                    nn.Dropout(self.fused_fc_dropoutRate),
                    LinearWithConstraint(self.fused_fc_hidden_dim, self.nClass,
                                         max_norm=self.fused_fc_norm_rate),
                )
            else:
                self.fused_fc = LinearWithConstraint(
                    n_in, self.nClass, max_norm=self.fused_fc_norm_rate)
            self.fused_fc.to(x.device)

        return F.log_softmax(self.fused_fc(fused), dim=1)
