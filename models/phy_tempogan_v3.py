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
        
# 简单的 LSTM 实现 (用于残差分支)
class LSTMCell(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size, bias=True):
        super(LSTMCell, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        padding = kernel_size // 2
        
        self.conv = nn.Conv2d(in_channels=input_dim + hidden_dim,
                              out_channels=4 * hidden_dim,
                              kernel_size=kernel_size,
                              padding=padding,
                              bias=bias)

    def forward(self, input_tensor, cur_state):
        h_cur, c_cur = cur_state
        combined = torch.cat([input_tensor, h_cur], dim=1)
        combined_conv = self.conv(combined)
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1)
        
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)
        
        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)
        
        return h_next, c_next

class PhyGenerator(nn.Module):
    def __init__(self, scale_factor=4, in_channels=1, base_channels=32):
        super(PhyGenerator, self).__init__()
        self.scale_factor = scale_factor
        
        # --------------------------------------------------------
        # 1. Shared Encoder (共享编码器)
        # --------------------------------------------------------
        # 将输入从 [B, 1, 512, 512] 压缩到特征空间 [B, 32, 128, 128]
        # 假设输入已经是通过最近邻插值放大过的
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, 5, 1, 2),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(base_channels, base_channels*2, 3, 2, 1), # Downsample /2
            nn.BatchNorm2d(base_channels*2),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(base_channels*2, base_channels*4, 3, 2, 1), # Downsample /4
            nn.BatchNorm2d(base_channels*4),
            nn.LeakyReLU(0.1, inplace=True)
        )
        
        feature_dim = base_channels * 4 # e.g., 32*4 = 128 channels
        
        # --------------------------------------------------------
        # 2. Branch A: Physical Dynamics (PhyCell)
        # --------------------------------------------------------
        # 负责预测宏观流动和 PDE 约束
        self.phy_cell = PhyCell(
            input_dim=feature_dim,
            F_hidden_dims=[feature_dim],
            n_layers=1,
            kernel_size=(7, 7) # 大核感受野，捕捉物理
        )
        
        # --------------------------------------------------------
        # 3. Branch B: Residual Texture (LSTM)
        # --------------------------------------------------------
        # 负责记忆高频细节和非物理纹理
        self.res_cell = LSTMCell(
            input_dim=feature_dim,
            hidden_dim=feature_dim,
            kernel_size=3 # 小核感受野，关注局部细节
        )
        
        # --------------------------------------------------------
        # 4. Decoder & Fusion
        # --------------------------------------------------------
        # 融合两个分支的信息
        self.fusion_conv = nn.Conv2d(feature_dim * 2, feature_dim, 1)
        
        self.decoder = nn.Sequential(
            # Upsample /2 -> x2
            nn.ConvTranspose2d(feature_dim, base_channels*2, 4, 2, 1),
            nn.BatchNorm2d(base_channels*2),
            nn.LeakyReLU(0.1, inplace=True),
            
            # Upsample /4 -> x1 (Back to High Res)
            nn.ConvTranspose2d(base_channels*2, base_channels, 4, 2, 1),
            nn.BatchNorm2d(base_channels),
            nn.LeakyReLU(0.1, inplace=True),
            
            # Final Refinement
            nn.Conv2d(base_channels, base_channels, 3, 1, 1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(base_channels, 1, 3, 1, 1),
            nn.Tanh()
        )

    def forward(self, x_seq):
        """
        x_seq: [B, T=3, C, H, W] (Pre-upsampled high-res input)
        """
        B, T, C, H, W = x_seq.size()
        
        # 1. Encode the whole sequence
        # Reshape to [B*T, C, H, W]
        x_flat = x_seq.view(-1, C, H, W)
        x_flat = F.interpolate(x_flat, scale_factor=self.scale_factor, mode='nearest')
        
        feats_flat = self.encoder(x_flat) # -> [B*T, 128, H/4, W/4]
        
        # Reshape back to sequence [B, T, 128, H/4, W/4]
        feats_seq = feats_flat.view(B, T, -1, feats_flat.size(2), feats_flat.size(3))
        
        # 2. Run Branches Recurrently
        # 初始化状态
        h_phy = None # PhyCell handles its own init
        h_res = None 
        c_res = None
        
        # 初始化 LSTM 状态
        feat_h, feat_w = feats_flat.size(2), feats_flat.size(3)
        h_res = torch.zeros(B, feats_flat.size(1), feat_h, feat_w).to(x_seq.device)
        c_res = torch.zeros(B, feats_flat.size(1), feat_h, feat_w).to(x_seq.device)
        
        # PhyCell 内部已经处理了序列，但为了和 LSTM 同步，我们可以手动循环，
        # 或者直接调用 PhyCell 处理序列，再单独处理 LSTM。
        # 这里为了清晰，我们让 PhyCell 一次性处理完。
        phy_out_seq, _ = self.phy_cell(feats_seq) # [B, T, C, H', W']
        
        # LSTM 手动循环 (处理 Branch B)
        res_outputs = []
        for t in range(T):
            current_input = feats_seq[:, t]
            h_res, c_res = self.res_cell(current_input, (h_res, c_res))
            res_outputs.append(h_res)
        res_out_seq = torch.stack(res_outputs, dim=1)
        
        # 3. Focus on the Center Frame (t=1)
        # 我们不仅要当前帧的特征，还要利用过去和未来的上下文
        # PhyDNet 的逻辑是 sum(h_phy, h_res)
        
        h_phy_t = phy_out_seq[:, 1]
        h_res_t = res_out_seq[:, 1]
        
        # 融合
        # 论文中是直接相加: decoded = D(h_phy + h_res)
        # 但我们这里可以用 concat 让网络自己学习权重
        combined = torch.cat([h_phy_t, h_res_t], dim=1)
        fused = self.fusion_conv(combined)
        
        # 4. Decode
        out = self.decoder(fused)
        
        return out

# =========================================================================
# 判别器部分 (Discriminators)
# =========================================================================
from torch.nn.utils import spectral_norm

class SpatialDiscriminator(nn.Module):
    def __init__(self, input_channels=2, base_channels=32):
        super(SpatialDiscriminator, self).__init__()
        
        self.layer1 = nn.Sequential(
            spectral_norm(nn.Conv2d(input_channels, base_channels, 4, 2, 1)),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.layer2 = nn.Sequential(
            spectral_norm(nn.Conv2d(base_channels, base_channels * 2, 4, 2, 1)),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.layer3 = nn.Sequential(
            spectral_norm(nn.Conv2d(base_channels * 2, base_channels * 4, 4, 2, 1)),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.layer4 = nn.Sequential(
            spectral_norm(nn.Conv2d(base_channels * 4, base_channels * 8, 4, 2, 1)),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.final_layers = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            # === 修改点 1: 最后一层移除 spectral_norm ===
            # 这允许输出值突破 [-1, 1] 的限制，对 Hinge Loss 至关重要
            nn.Linear(base_channels * 8, 1) 
        )
        
    def forward(self, upsampled_low_res, high_res_candidate):
        x = torch.cat([upsampled_low_res, high_res_candidate], dim=1)
        f1 = self.layer1(x)
        f2 = self.layer2(f1)
        f3 = self.layer3(f2)
        f4 = self.layer4(f3)
        score = self.final_layers(f4)
        return score, [f1, f2, f3, f4]


class TemporalDiscriminator(nn.Module):
    def __init__(self, input_channels=3, base_channels=32):
        super(TemporalDiscriminator, self).__init__()
        
        # input_channels 虽然传入 3，但在 forward 里我们会计算差分
        # 变成 9 通道输入: [Img_t, Diff_prev, Diff_next] 或其他组合
        # 简单起见，我们把 原始3帧 + 2个差分帧 = 5通道 堆叠
        
        actual_in_channels = 5 # t-1, t, t+1, (t)-(t-1), (t+1)-(t)
        
        self.layer1 = nn.Sequential(
            spectral_norm(nn.Conv2d(actual_in_channels, base_channels, 4, 2, 1)),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.layer2 = nn.Sequential(
            spectral_norm(nn.Conv2d(base_channels, base_channels * 2, 4, 2, 1)),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.layer3 = nn.Sequential(
            spectral_norm(nn.Conv2d(base_channels * 2, base_channels * 4, 4, 2, 1)),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.layer4 = nn.Sequential(
            spectral_norm(nn.Conv2d(base_channels * 4, base_channels * 8, 4, 2, 1)),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.final_layers = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            # === 修改点 1: 最后一层移除 spectral_norm ===
            nn.Linear(base_channels * 8, 1)
        )

    def forward(self, frame_sequence):
        # frame_sequence: [B, 3, 1, H, W]
        B, T, C, H, W = frame_sequence.size()
        
        # 拆分帧
        f_prev = frame_sequence[:, 0] # [B, 1, H, W]
        f_curr = frame_sequence[:, 1]
        f_next = frame_sequence[:, 2]
        
        # === 修改点 2: 显式计算时序差分 (Motion Cues) ===
        # 强迫判别器关注变化量
        diff_prev = f_curr - f_prev
        diff_next = f_next - f_curr
        
        # 拼接: [原始信息(3) + 运动信息(2)] = 5通道
        # 在通道维度堆叠
        x = torch.cat([f_prev, f_curr, f_next, diff_prev, diff_next], dim=1) # [B, 5, H, W]
        
        f1 = self.layer1(x)
        f2 = self.layer2(f1)
        f3 = self.layer3(f2)
        f4 = self.layer4(f3)
        score = self.final_layers(f4)
        return score