import os
import json
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from PIL import Image

class TyphoonDataset(Dataset):
    def __init__(self, json_path, root_dir, flow_dir, transform=None, scale_factor=4):
        """
        Args:
            json_path: 三元组 JSON 路径
            root_dir: 图片根目录
            flow_dir: 光流 .npy 根目录
            scale_factor: 下采样倍率 (默认 4: 512 -> 128)
        """
        self.root_dir = root_dir
        self.flow_dir = flow_dir
        self.scale_factor = scale_factor
        
        with open(json_path, 'r') as f:
            self.triplets = json.load(f)
            
        # 预定义的归一化参数
        # 速度建议归一化。假设最大风速/光流一般不超过 50 像素(512尺度下)
        self.max_flow_mag = 50.0 

    def __len__(self):
        return len(self.triplets)

    def _load_img(self, path):
        """加载图片，归一化到 [-1, 1]"""
        # 1. Load as Grayscale
        img = Image.open(path).convert('L')
        img_np = np.array(img)
        
        # 2. To Tensor [1, H, W] and Normalize to [-1, 1]
        img_tensor = torch.from_numpy(img_np).float().unsqueeze(0)
        img_tensor = (img_tensor / 127.5) - 1.0
        
        return img_tensor

    def _load_flow_pair(self, img_path):
        """
        尝试加载某张图片对应的 forward/backward 光流
        如果文件不存在（比如序列的第一帧或最后一帧），返回零张量
        """
        base_name = os.path.basename(img_path).replace('.png', '')
        fwd_path = os.path.join(self.flow_dir, f"{base_name}_forward.npy")
        bwd_path = os.path.join(self.flow_dir, f"{base_name}_backward.npy")
        
        # 默认形状 [2, 512, 512]
        flow_shape = (2, 512, 512) 
        
        if os.path.exists(fwd_path):
            fwd = np.load(fwd_path)
            fwd_tensor = torch.from_numpy(fwd).float()
        else:
            fwd_tensor = torch.zeros(flow_shape)
            
        if os.path.exists(bwd_path):
            bwd = np.load(bwd_path)
            bwd_tensor = torch.from_numpy(bwd).float()
        else:
            bwd_tensor = torch.zeros(flow_shape)
            
        return fwd_tensor, bwd_tensor

    def _downsample(self, tensor, is_flow=False):
        """
        下采样 Tensor.
        Args:
            tensor: [C, H, W]
            is_flow: 如果是光流，数值需要除以 scale_factor
        """
        # interpolate 需要 [B, C, H, W]
        t = tensor.unsqueeze(0)
        
        # Area 插值适合图像缩小，Bilinear 适合光流
        mode = 'bilinear' if is_flow else 'area'
        
        downsampled = F.interpolate(
            t, 
            scale_factor=1/self.scale_factor, 
            mode=mode, 
            recompute_scale_factor=True,
            align_corners=False if mode=='bilinear' else None
        ).squeeze(0)
        
        if is_flow:
            downsampled = downsampled / self.scale_factor
            
        return downsampled

    def __getitem__(self, idx):
        path_prev, path_curr, path_next = self.triplets[idx]
        
        paths = [path_prev, path_curr, path_next]
        
        # 容器
        hr_imgs = []      # [High-Res Density]
        lr_inputs = []    # [Low-Res Density + Low-Res Velocity] (for G)
        
        # 专门用于 Advection 的 High-Res Flow (只取中间帧 t 的)
        # 因为 Advection 只需要把 t-1 和 t+1 拉向 t
        _, hr_flow_curr_bwd = self._load_flow_pair(path_curr) # t -> t-1 (backward)
        hr_flow_curr_fwd, _ = self._load_flow_pair(path_curr) # t -> t+1 (forward)
        
        # 遍历 t-1, t, t+1
        for p in paths:
            # 1. Load HR Image
            hr_img = self._load_img(p) # [1, 512, 512]
            hr_imgs.append(hr_img)
            
            # 2. Load HR Flow (尝试加载该帧自己的光流，用于构造输入特征)
            # 注意：G 输入通常只需要一个速度矢量即可。
            # 论文中 G 输入 velocity 是 (u, v)。
            # 我们可以把 forward flow 和 backward flow 平均一下，或者只用 forward。
            # 为了信息量最大化，我们可以把 fwd 和 bwd 拼起来变成 4 通道 flow，或者取平均。
            # 这里简化处理：我们取 Forward Flow 作为主要速度特征。
            fwd, bwd = self._load_flow_pair(p)
            
            # 简单的速度估计：(Forward - Backward) / 2 或者直接用 Forward
            # 更加鲁棒的方式：如果 Forward 是 0 (边缘帧)，就用 -Backward
            velocity_hr = fwd 
            
            # 3. Downsample Image
            lr_img = self._downsample(hr_img, is_flow=False) # [1, 128, 128]
            
            # 4. Downsample Flow
            lr_vel = self._downsample(velocity_hr, is_flow=True) # [2, 128, 128]
            
            # 5. Normalize Flow (for Network Input stability)
            # 限制在 [-1, 1] 之间，避免数值过大导致梯度爆炸
            lr_vel = torch.clamp(lr_vel / (self.max_flow_mag / self.scale_factor), -1.0, 1.0)
            
            # 6. Concatenate for G Input
            # [1, 128, 128] cat [2, 128, 128] -> [3, 128, 128]
            lr_input = torch.cat([lr_img, lr_vel], dim=0)
            lr_inputs.append(lr_input)

        return {
            # 生成器输入 (List of 3 tensors: t-1, t, t+1)
            'lr_inputs': lr_inputs, 
            
            # 判别器真值 / Loss计算 (List of 3 tensors: t-1, t, t+1)
            'hr_imgs': hr_imgs,
            
            # 时间判别器需要的 Warp 场 (都是 t 时刻发出的)
            'flow_bwd': hr_flow_curr_bwd, # [2, 512, 512]
            'flow_fwd': hr_flow_curr_fwd  # [2, 512, 512]
        }