import torch
import torch.nn as nn
import torch.nn.functional as F

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

class Generator(nn.Module):
    def __init__(self, scale_factor=4, input_channels=3, base_channels=64):
        super(Generator, self).__init__()
        self.scale_factor = scale_factor
        
        # 初始层 (通常不加 BN，保持原始特征分布)
        self.initial_conv = nn.Conv2d(input_channels, base_channels, kernel_size=5, padding=2)
        
        # 你的结构很稳健，保留这个深层 ResNet 结构
        self.res_blocks = nn.Sequential(
            ResidualBlock(base_channels, base_channels),
            ResidualBlock(base_channels, base_channels),
            ResidualBlock(base_channels, base_channels * 2),
            ResidualBlock(base_channels * 2, base_channels * 2),
            ResidualBlock(base_channels * 2, base_channels * 4),
            ResidualBlock(base_channels * 4, base_channels * 2),
            ResidualBlock(base_channels * 2, base_channels),
            ResidualBlock(base_channels, base_channels // 2)
        )
        
        # 最终输出层 (无BN, 无Activation，输出原始数值)
        self.final_conv = nn.Conv2d(base_channels // 2, 1, kernel_size=3, padding=1)
        
    def forward(self, x):
        # 1. Nearest Upsample First (符合论文)
        upsampled = F.interpolate(x, scale_factor=self.scale_factor, mode='nearest')
        
        out = self.initial_conv(upsampled)
        out = self.res_blocks(out)
        out = self.final_conv(out)
        
        # 注意：如果你归一化到了 [-1, 1]，这里可能需要 Tanh()
        # 如果是 [0, 1]，可能需要 Sigmoid()
        # 论文中通常输出 Linear，依靠 Loss 来约束，或者加个 Tanh
        return torch.tanh(out) 

class SpatialDiscriminator(nn.Module):
    def __init__(self, input_channels=2, base_channels=32):
        super(SpatialDiscriminator, self).__init__()
        
        # 将层分开定义，以便在前向传播中提取中间特征
        # 论文参数：Kernel=4, Stride=2, Padding=1 (经典 PatchGAN 设置)
        
        self.layer1 = nn.Sequential(
            nn.Conv2d(input_channels, base_channels, 4, 2, 1), # [256, 256]
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        self.layer2 = nn.Sequential(
            nn.Conv2d(base_channels, base_channels * 2, 4, 2, 1), # [128, 128]
            nn.BatchNorm2d(base_channels * 2),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        self.layer3 = nn.Sequential(
            nn.Conv2d(base_channels * 2, base_channels * 4, 4, 2, 1), # [64, 64]
            nn.BatchNorm2d(base_channels * 4),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        self.layer4 = nn.Sequential(
            nn.Conv2d(base_channels * 4, base_channels * 8, 4, 2, 1), # [32, 32]
            nn.BatchNorm2d(base_channels * 8),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        # 论文最后用的是全连接，但对于大图，保留 AdaptiveAvgPool 是个好主意
        # 或者继续卷积直到 1x1
        self.final_layers = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(base_channels * 8, 1),
            nn.Sigmoid()
        )
        
    def forward(self, upsampled_low_res, high_res_candidate):
        """
        返回: (score, features_list)
        features_list 用于计算 Feature Space Loss
        """
        x = torch.cat([upsampled_low_res, high_res_candidate], dim=1)
        
        f1 = self.layer1(x)
        f2 = self.layer2(f1)
        f3 = self.layer3(f2)
        f4 = self.layer4(f3)
        
        score = self.final_layers(f4)
        
        # 返回所有中间特征，用于 Loss 计算
        return score, [f1, f2, f3, f4]

class TemporalDiscriminator(nn.Module):
    def __init__(self, input_channels=3, base_channels=32):
        super(TemporalDiscriminator, self).__init__()
        
        # Temporal Discriminator 不需要提取特征 Loss，只需要输出真假
        
        self.model = nn.Sequential(
            # [B, 3, H, W] -> [B, 32, H/2, W/2]
            nn.Conv2d(input_channels, base_channels, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(base_channels, base_channels * 2, 4, 2, 1),
            nn.BatchNorm2d(base_channels * 2),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(base_channels * 2, base_channels * 4, 4, 2, 1),
            nn.BatchNorm2d(base_channels * 4),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(base_channels * 4, base_channels * 8, 4, 2, 1),
            nn.BatchNorm2d(base_channels * 8),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(base_channels * 8, 1),
            nn.Sigmoid()
        )
        
    def forward(self, aligned_frames):
        # aligned_frames 已经是 [Warped_t-1, Real_t, Warped_t+1]
        return self.model(aligned_frames)