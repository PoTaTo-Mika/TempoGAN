import torch
import torch.nn as nn
import torch.nn.functional as F

class AdvectionLayer(nn.Module):
    def __init__(self):
        super(AdvectionLayer, self).__init__()
        
    def forward(self, image, flow):
        """
        使用光流对图像进行warp操作 (Semi-Lagrangian Advection)
        
        Args:
            image: [B, C, H, W] 要被移动的图像 (如 frame t-1)
            flow:  [B, 2, H, W] 速度场 (u, v)。注意：这里通常需要是 "Backward Flow" 
                   或者是从 t 指向 t-1 的位移矢量。
                   Grid Sample 需要归一化到 [-1, 1]。
                   
        Returns:
            warped_image: [B, C, H, W] 对齐到当前时刻的图像
        """
        B, C, H, W = image.size()
        
        # 1. 生成基础网格 (xx, yy)
        xx = torch.arange(0, W).view(1, -1).repeat(H, 1)
        yy = torch.arange(0, H).view(-1, 1).repeat(1, W)
        xx = xx.view(1, 1, H, W).repeat(B, 1, 1, 1)
        yy = yy.view(1, 1, H, W).repeat(B, 1, 1, 1)
        grid = torch.cat((xx, yy), 1).float()  # [B, 2, H, W]
        
        if image.is_cuda:
            grid = grid.cuda()
            
        # 2. 应用流场 (Grid = Base + Flow)
        # 注意：grid_sample 期望的是采样坐标。
        # 如果我们想把 t-1 移到 t。
        # 这里的 flow 应该是 "从 t 像素点 指向 t-1 来源点" 的向量 (Backward Mapping)。
        # 通常简单的 Advection 近似是： pos_src = pos_dst - velocity * dt
        vgrid = grid - flow  # 减去速度 (假设 flow 是速度 * dt)
        
        # 3. 归一化到 [-1, 1]
        vgrid[:, 0, :, :] = 2.0 * vgrid[:, 0, :, :] / max(W - 1, 1) - 1.0
        vgrid[:, 1, :, :] = 2.0 * vgrid[:, 1, :, :] / max(H - 1, 1) - 1.0
        
        # 4. 调整维度为 [B, H, W, 2]
        vgrid = vgrid.permute(0, 2, 3, 1)
        
        # 5. 采样
        warped = F.grid_sample(
            image, 
            vgrid, 
            mode='bilinear', 
            align_corners=False, 
            padding_mode='border'
        )
        
        return warped

# 使用示例函数
def advect_frames(generated_frames, flow_fields):
    """
    将生成的帧序列进行平流对齐
    
    Args:
        generated_frames: List of [B, C, H, W] tensors - [frame_{t-1}, frame_t, frame_{t+1}]
        flow_fields: List of [B, 2, H, W] tensors - [flow_{t-1->t}, flow_{t+1->t}]
        
    Returns:
        aligned_frames: List of [B, C, H, W] tensors - 对齐到时刻t的帧序列
    """
    advection_layer = AdvectionLayer()
    
    frame_t_minus_1, frame_t, frame_t_plus_1 = generated_frames
    flow_backward, flow_forward = flow_fields
    
    # 将 t-1 和 t+1 帧对齐到 t 时刻
    aligned_t_minus_1 = advection_layer(frame_t_minus_1, flow_backward)
    aligned_t_plus_1 = advection_layer(frame_t_plus_1, flow_forward)
    
    return [aligned_t_minus_1, frame_t, aligned_t_plus_1]