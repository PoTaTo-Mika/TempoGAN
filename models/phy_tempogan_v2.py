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
    def __init__(self, scale_factor=4, in_channels=1, base_channels=32):
        """
        High-Res Physics Guided Generator
        思路：
        1. Input (Low) -> Nearest Upsample -> Input (High)
        2. Main Branch: 处理高分辨率纹理
        3. Physics Branch: Downsample -> PhyCell (学习动态) -> Upsample
        4. Fusion: 将物理信息注入主分支
        """
        super(PhyGenerator, self).__init__()
        self.scale_factor = scale_factor
        
        # ============================================================
        # 1. 前端处理 (High-Res Input Space)
        # ============================================================
        # 类似 TempoGAN，先提取特征。这里输入已经是 Upsampled 的尺寸
        self.encoder_high = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, 5, 1, 2),
            nn.ReLU(inplace=True)
        )

        # ============================================================
        # 2. 物理分支 (The "Brain" - Low Resolution Dynamics)
        # ============================================================
        # 将高分特征压缩 4 倍 (比如 512 -> 128)，在这里跑 PhyCell
        self.phy_downsampler = nn.Sequential(
            nn.Conv2d(base_channels, base_channels*2, 3, 2, 1), # /2
            nn.BatchNorm2d(base_channels*2),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels*2, base_channels*4, 3, 2, 1), # /4
            nn.BatchNorm2d(base_channels*4),
            nn.ReLU(inplace=True)
        )
        
        # PhyCell 在压缩后的特征空间运行
        # Input dim: base_channels * 4 (e.g., 32*4=128)
        self.phy_cell = PhyCell(
            input_dim=base_channels * 4,
            F_hidden_dims=[base_channels * 4], 
            n_layers=1, 
            kernel_size=(3, 3) # 在 latent space 不需要太大的 kernel
        )
        
        # 将物理预测还原回高分辨率
        self.phy_upsampler = nn.Sequential(
            nn.ConvTranspose2d(base_channels * 4, base_channels * 2, 4, 2, 1),
            nn.BatchNorm2d(base_channels * 2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(base_channels * 2, base_channels, 4, 2, 1),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True)
        )

        # ============================================================
        # 3. 纹理主分支 (The "Artist" - High Resolution Texture)
        # ============================================================
        # 这是 TempoGAN 原版的 ResNet 部分，负责画细节
        # 输入通道 = Encoder(32) + Physics(32) = 64
        self.main_res_blocks = nn.Sequential(
            ResidualBlock(base_channels * 2, base_channels * 2),
            ResidualBlock(base_channels * 2, base_channels * 2),
            ResidualBlock(base_channels * 2, base_channels * 2),
            ResidualBlock(base_channels * 2, base_channels * 2)
        )
        
        # 最终细化
        self.final_refine = nn.Sequential(
            ResidualBlock(base_channels * 2, base_channels),
            nn.Conv2d(base_channels, 1, 3, 1, 1),
            nn.Tanh() # 假设数据归一化在 [-1, 1]
        )

    def forward(self, x_seq):
        """
        Args:
            x_seq: [B, T, C, H_low, W_low] (例如: 3帧的低分序列)
        Returns:
            out: [B, C, H_high, W_high] (中间帧 t 的高分预测)
        """
        B, T, C, H, W = x_seq.size()
        
        # -------------------------------------------------------
        # Step 1: 预上采样 (Pre-Upsampling)
        # -------------------------------------------------------
        # 将整个序列上采样到目标分辨率
        # [B*T, C, H, W] -> Upsample -> [B*T, C, 4H, 4W]
        x_flat = x_seq.view(-1, C, H, W)
        x_high_flat = F.interpolate(x_flat, scale_factor=self.scale_factor, mode='nearest')
        
        # -------------------------------------------------------
        # Step 2: 提取高分特征
        # -------------------------------------------------------
        feats_high_flat = self.encoder_high(x_high_flat)
        # 还原回序列维度 [B, T, C_base, H_high, W_high]
        feats_high_seq = feats_high_flat.view(B, T, -1, x_high_flat.size(2), x_high_flat.size(3))
        
        # -------------------------------------------------------
        # Step 3: 物理分支 (Physics Branch)
        # -------------------------------------------------------
        # 我们需要在序列上运行 PhyCell 来捕捉 t-1 -> t 的动态
        
        # A. 再次扁平化以进行下采样 (为了高效)
        feat_for_phy = feats_high_seq.view(-1, feats_high_seq.size(2), feats_high_seq.size(3), feats_high_seq.size(4))
        feats_low_flat = self.phy_downsampler(feat_for_phy)
        
        # B. 还原序列维度供 PhyCell 使用
        # [B, T, C_deep, H_deep, W_deep]
        feats_low_seq = feats_low_flat.view(B, T, -1, feats_low_flat.size(2), feats_low_flat.size(3))
        
        # C. 运行 PhyCell
        # out_seq: [B, T, C, H, W]
        # 我们只需要 PhyCell 在时刻 t (索引1) 的输出，因为它融合了 t-1 的动态
        phy_out_seq, _ = self.phy_cell(feats_low_seq)
        phy_state_t = phy_out_seq[:, 1] # 取中间帧对应的物理状态
        
        # D. 将物理状态上采样回高分辨率
        phy_guidance = self.phy_upsampler(phy_state_t) # [B, 32, 4H, 4W]
        
        # -------------------------------------------------------
        # Step 4: 纹理分支融合 (Texture Fusion)
        # -------------------------------------------------------
        # 取出中间帧 t 的原始高分特征
        center_feat_high = feats_high_seq[:, 1] # [B, 32, 4H, 4W]
        
        # 将 "视觉特征" 和 "物理指导" 拼接
        # Combined: [B, 64, 4H, 4W]
        combined = torch.cat([center_feat_high, phy_guidance], dim=1)
        
        # -------------------------------------------------------
        # Step 5: 最终生成
        # -------------------------------------------------------
        # 使用 ResNet 细化纹理，此时网络知道：
        # 1. 原始图像长什么样 (center_feat_high)
        # 2. 物理上应该怎么动 (phy_guidance)
        out_feat = self.main_res_blocks(combined)
        final_image = self.final_refine(out_feat)
        
        return final_image

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