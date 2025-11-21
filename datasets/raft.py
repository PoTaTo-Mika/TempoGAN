import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.optical_flow import Raft_Large_Weights, raft_large


def compute_optical_flow(img_t, img_t1):
    """
    计算两张连续灰度图像的光流场
    
    Args:
        img_t: t时刻的灰度图像，形状为[512, 512]的numpy数组
        img_t1: t+1时刻的灰度图像，形状为[512, 512]的numpy数组
        
    Returns:
        flow_field: 光流场，形状为[2, 512, 512]的numpy数组，[0,:,:]为X速度分量，[1,:,:]为Y速度分量
    """
    # 检查输入形状
    assert img_t.shape == (512, 512), f"img_t形状应为(512, 512)，实际为{img_t.shape}"
    assert img_t1.shape == (512, 512), f"img_t1形状应为(512, 512)，实际为{img_t1.shape}"
    
    # 将numpy数组转换为torch tensor并添加batch和channel维度
    img_t_tensor = torch.from_numpy(img_t).float().unsqueeze(0).unsqueeze(0)  # [1, 1, 512, 512]
    img_t1_tensor = torch.from_numpy(img_t1).float().unsqueeze(0).unsqueeze(0)  # [1, 1, 512, 512]
    
    # 将单通道灰度图像复制为三通道（RAFT需要RGB输入）
    img_t_rgb = img_t_tensor.repeat(1, 3, 1, 1)  # [1, 3, 512, 512]
    img_t1_rgb = img_t1_tensor.repeat(1, 3, 1, 1)  # [1, 3, 512, 512]
    
    # 加载RAFT模型和预训练权重
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = raft_large(weights=Raft_Large_Weights.DEFAULT, progress=False).to(device)
    model.eval()
    
    # 将图像移动到设备并归一化
    img_t_rgb = img_t_rgb.to(device)
    img_t1_rgb = img_t1_rgb.to(device)
    
    # RAFT模型会自动处理归一化，我们只需要确保输入在[0,1]范围内
    if img_t_rgb.max() > 1.0:
        img_t_rgb = img_t_rgb / 255.0
        img_t1_rgb = img_t1_rgb / 255.0
    
    # 计算光流
    with torch.no_grad():
        flow_predictions = model(img_t_rgb, img_t1_rgb)
        # 取最后的预测结果
        flow = flow_predictions[-1]  # [1, 2, 512, 512]
    
    # 转换为numpy数组并移除batch维度
    flow_np = flow.squeeze(0).cpu().numpy()  # [2, 512, 512]
    
    return flow_np

