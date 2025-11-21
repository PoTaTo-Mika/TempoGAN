import torch
from datasets.raft import compute_optical_flow
import numpy as np

def compute_vorticity_from_flow(flow_field):
    """
    从光流场计算涡量场
    
    Args:
        flow_field: 光流场，形状为[2, H, W]的numpy数组，[0,:,:]为X速度分量，[1,:,:]为Y速度分量
        
    Returns:
        vorticity: 涡量场，形状为[H, W]的numpy数组
    """
    # 提取速度分量
    u = flow_field[0]  # X方向速度
    v = flow_field[1]  # Y方向速度
    
    # 计算速度梯度
    du_dy, du_dx = np.gradient(u)
    dv_dy, dv_dx = np.gradient(v)
    
    # 涡量 = ∂v/∂x - ∂u/∂y
    vorticity = dv_dx - du_dy
    
    return vorticity

def compute_vorticity(img_t, img_t1):
    """
    计算两张连续灰度图像的涡量场
    
    Args:
        img_t: t时刻的灰度图像，形状为[512, 512]的numpy数组
        img_t1: t+1时刻的灰度图像，形状为[512, 512]的numpy数组
        
    Returns:
        vorticity: 涡量场，形状为[512, 512]的numpy数组
    """
    # 首先计算光流场
    flow_field = compute_optical_flow(img_t, img_t1)
    
    # 从光流场计算涡量
    vorticity = compute_vorticity_from_flow(flow_field)
    
    return vorticity

def compute_vorticity_magnitude(vorticity_field):
    """
    计算涡量场的幅值
    
    Args:
        vorticity_field: 涡量场，形状为[H, W]的numpy数组
        
    Returns:
        magnitude: 涡量幅值场，形状为[H, W]的numpy数组
    """
    return np.abs(vorticity_field)

def normalize_vorticity(vorticity_field):
    """
    归一化涡量场到[0, 1]范围
    
    Args:
        vorticity_field: 涡量场，形状为[H, W]的numpy数组
        
    Returns:
        normalized_vorticity: 归一化后的涡量场
    """
    vorticity_abs = np.abs(vorticity_field)
    if np.max(vorticity_abs) > 0:
        normalized_vorticity = vorticity_abs / np.max(vorticity_abs)
    else:
        normalized_vorticity = vorticity_abs
    return normalized_vorticity