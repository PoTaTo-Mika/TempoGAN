import os
import torch
import numpy as np
from PIL import Image
import torchvision.utils as vutils

def save_checkpoint(state, save_dir, filename):
    os.makedirs(save_dir, exist_ok=True)
    torch.save(state, os.path.join(save_dir, filename))

def tensor_to_img(tensor):
    """
    将 [-1, 1] 的 Tensor 转换为 [0, 255] 的 numpy uint8 图片
    """
    # Denormalize: [-1, 1] -> [0, 1]
    img = (tensor.detach().cpu() + 1) / 2.0
    img = torch.clamp(img, 0, 1)
    return img

def save_sample_images(curr_iter, lr_stack, hr_stack, gen_stack, save_dir):
    """
    保存对比图: LowRes | Generated | HighRes
    只保存中间帧 (Frame t) 以供观察
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # 取 batch 中的第一组数据
    # lr_stack: [B, 3, C, H, W], hr_stack: [B, 3, 1, H, W], gen_stack: [B, 3, 1, H, W]
    
    # 取中间帧 t (index 1)
    lr_t = lr_stack[0, 1, 0:1, :, :] # 取密度通道
    hr_t = hr_stack[0, 1, :, :, :]
    gen_t = gen_stack[0, 1, :, :, :]
    
    # 因为 LR 尺寸小，用插值放大方便拼接
    lr_t = torch.nn.functional.interpolate(lr_t.unsqueeze(0), size=hr_t.shape[-2:], mode='nearest').squeeze(0)
    
    # 拼接
    grid = vutils.make_grid([lr_t, gen_t, hr_t], nrow=3, normalize=True, value_range=(-1, 1))
    vutils.save_image(grid, os.path.join(save_dir, f"step_{curr_iter}.png"))

class AverageMeter:
    """计算滑动平均值"""
    def __init__(self):
        self.reset()
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count