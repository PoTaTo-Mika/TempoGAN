import torch
import torch.nn as nn
import torch.nn.functional as F
from models.phycell import PhyCell

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.skip_conv = None
        if in_channels != out_channels:
            self.skip_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        
    def forward(self, x):
        identity = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        
        if self.skip_conv is not None:
            identity = self.skip_conv(identity)
        
        out += identity
        out = F.relu(out)
        return out

class PhyGenerator(nn.Module):
    def __init__(self, scale_factor=4, in_channels=1, base_channels=64):
        """
        PhyTempoGAN Generator
        Args:
            scale_factor: 上采样倍率 (4: 128 -> 512)
            in_channels: 输入单帧的通道数 (灰度图=1)
            base_channels: 基础特征通道数
        """
        super(PhyGenerator, self).__init__()
        self.scale_factor = scale_factor
        
        # 1. Encoder (将图像编码到特征空间)
        # 输入: [B, 1, H, W] -> 输出: [B, 64, H, W]
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, base_channels // 2, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(base_channels // 2, base_channels, kernel_size=3, padding=1),
            nn.GroupNorm(4, base_channels), # 使用 GroupNorm 配合 PhyCell
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        # 2. PhyCell Core (物理核心)
        # 在 128x128 的低分辨率特征上学习动力学
        self.phy_cell = PhyCell(
            input_dim=base_channels, 
            F_hidden_dims=[base_channels], # 隐藏层维度保持一致
            n_layers=1, 
            kernel_size=(7, 7) # 较大的感受野捕捉 PDE
        )
        
        # 3. Residual Backbone (纹理细化)
        # 结合 PhyCell 的物理特征 + 原始图像特征
        # 输入通道 = PhyCell Hidden (64) + Encoder Feature (64) = 128
        self.res_blocks = nn.Sequential(
            ResidualBlock(base_channels * 2, base_channels * 2),
            ResidualBlock(base_channels * 2, base_channels * 2),
            ResidualBlock(base_channels * 2, base_channels * 2),
            ResidualBlock(base_channels * 2, base_channels * 4), # 通道扩张
        )
        
        # 4. Upsampler (上采样)
        # 128 -> 256 -> 512
        self.upsampler = nn.Sequential(
            nn.Conv2d(base_channels * 4, base_channels * 8, 3, 1, 1),
            nn.PixelShuffle(2), # 2x Upsample -> channels / 4 (base*2)
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(base_channels * 2, base_channels * 4, 3, 1, 1),
            nn.PixelShuffle(2), # 2x Upsample -> channels / 4 (base)
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        # 5. Final Output
        self.final_conv = nn.Conv2d(base_channels, 1, kernel_size=3, padding=1)

    def forward(self, x_seq):
        """
        Args:
            x_seq: [B, 3, 1, H, W] 输入连续三帧 (t-1, t, t+1)
                   注意：这里输入的 H, W 是低分辨率 (128)
        Returns:
            out: [B, 1, 4H, 4W] 中间帧 t 的高分辨率预测
        """
        B, T, C, H, W = x_seq.size()
        
        # A. 提取每一帧的特征
        # reshape [B*3, 1, H, W] 进行批处理编码
        x_flat = x_seq.view(-1, C, H, W)
        feats_flat = self.encoder(x_flat)
        # 还原回序列 [B, 3, 64, H, W]
        feats_seq = feats_flat.view(B, T, -1, H, W)
        
        # B. 物理演变 (PhyCell)
        # 输入整个序列，PhyCell 会自动处理状态传递
        # output_seq: [B, 3, 64, H, W]
        # hidden_final: 最后的隐藏状态
        phy_feats_seq, _ = self.phy_cell(feats_seq)
        
        # C. 聚焦于中间帧 t (index 1)
        # 我们利用 t-1 推演出的 t 时刻物理状态，结合 t 时刻本身的特征
        # phy_feats_seq[:, 1] 包含了从 t-1 到 t 的动力学信息
        feat_t = feats_seq[:, 1]      # Encoder 直接提取的 t 时刻特征
        phy_t = phy_feats_seq[:, 1]   # PhyCell 演变出的 t 时刻物理特征
        
        # 拼接：物理 + 外观
        combined = torch.cat([feat_t, phy_t], dim=1)
        
        # D. 残差修正与上采样
        res_feat = self.res_blocks(combined)
        high_res_feat = self.upsampler(res_feat)
        
        out = self.final_conv(high_res_feat)
        
        return torch.tanh(out)

# =========================================================================
# 判别器部分 (Discriminators)
# =========================================================================

class SpatialDiscriminator(nn.Module):
    def __init__(self, input_channels=2, base_channels=32):
        super(SpatialDiscriminator, self).__init__()
        # 保持原版结构不变，用于判断单帧真实性
        self.layer1 = nn.Sequential(
            nn.Conv2d(input_channels, base_channels, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(base_channels, base_channels * 2, 4, 2, 1),
            nn.BatchNorm2d(base_channels * 2),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(base_channels * 2, base_channels * 4, 4, 2, 1),
            nn.BatchNorm2d(base_channels * 4),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.layer4 = nn.Sequential(
            nn.Conv2d(base_channels * 4, base_channels * 8, 4, 2, 1),
            nn.BatchNorm2d(base_channels * 8),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.final_layers = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(base_channels * 8, 1),
            nn.Sigmoid() # 配合 BCELoss
        )
        
    def forward(self, upsampled_low_res, high_res_candidate):
        x = torch.cat([upsampled_low_res, high_res_candidate], dim=1)
        f1 = self.layer1(x)
        f2 = self.layer2(f1)
        f3 = self.layer3(f2)
        f4 = self.layer4(f3)
        score = self.final_layers(f4)
        return score, [f1, f2, f3, f4]

class TemporalDiscriminator3D(nn.Module):
    """
    替代原版的 Advection-based D_t
    使用 3D 卷积直接判断序列的时空连贯性，无需光流
    """
    def __init__(self, input_channels=1, base_channels=32):
        super(TemporalDiscriminator3D, self).__init__()
        
        self.model = nn.Sequential(
            # Input: [B, C, T=3, H, W]
            # Layer 1: 压缩时间维度
            nn.Conv3d(input_channels, base_channels, kernel_size=(3, 4, 4), stride=(1, 2, 2), padding=(0, 1, 1)),
            nn.LeakyReLU(0.2, inplace=True),
            
            # Layer 2
            nn.Conv3d(base_channels, base_channels * 2, kernel_size=(1, 4, 4), stride=(1, 2, 2), padding=(0, 1, 1)),
            nn.BatchNorm3d(base_channels * 2),
            nn.LeakyReLU(0.2, inplace=True),
            
            # Layer 3
            nn.Conv3d(base_channels * 2, base_channels * 4, kernel_size=(1, 4, 4), stride=(1, 2, 2), padding=(0, 1, 1)),
            nn.BatchNorm3d(base_channels * 4),
            nn.LeakyReLU(0.2, inplace=True),
            
            # Layer 4
            nn.Conv3d(base_channels * 4, base_channels * 8, kernel_size=(1, 4, 4), stride=(1, 2, 2), padding=(0, 1, 1)),
            nn.BatchNorm3d(base_channels * 8),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.AdaptiveAvgPool3d(1),
            nn.Flatten(),
            nn.Linear(base_channels * 8, 1),
            nn.Sigmoid()
        )

    def forward(self, frame_sequence):
        # frame_sequence: [B, 3, 1, H, W] -> permute to [B, 1, 3, H, W] for Conv3d
        x = frame_sequence.permute(0, 2, 1, 3, 4)
        return self.model(x)