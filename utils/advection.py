import torch
import torch.nn as nn
import torch.nn.functional as F

class AdvectionLayer(nn.Module):
    def __init__(self):
        super(AdvectionLayer, self).__init__()
        
    def forward(self, image, flow):
        """
        使用光流对图像进行warp操作
        
        Args:
            image: [B, C, H, W] 源图像 (例如 frame t-1)
            flow:  [B, 2, H, W] 位移场/光流 (从 target 指向 source 的位移)
                   例如：要warp t-1 到 t，需要 flow_curr_to_prev
                   
        Returns:
            warped_image: [B, C, H, W] 对齐后的图像
        """
        B, C, H, W = image.size()
        
        # 1. 生成基础网格 (直接在正确的设备上生成)
        # 使用 linspace 替代 arange 可以处理得更干净，但 arange 更直观
        # Normalized grid: -1 ~ 1
        # 直接生成归一化网格，省去后续的除法计算，稍微快一点
        
        # 这里为了稳健，我们还是生成像素坐标然后归一化，避免对齐问题
        if not hasattr(self, 'grid') or self.grid.size(0) != B or self.grid.size(2) != H or self.grid.size(3) != W:
            xx = torch.arange(0, W, device=image.device).view(1, -1).repeat(H, 1)
            yy = torch.arange(0, H, device=image.device).view(-1, 1).repeat(1, W)
            xx = xx.view(1, 1, H, W).repeat(B, 1, 1, 1)
            yy = yy.view(1, 1, H, W).repeat(B, 1, 1, 1)
            self.grid = torch.cat((xx, yy), 1).float() # [B, 2, H, W]
            
        # 2. 计算采样坐标
        # 关键修正：根据 process_raft.py 的逻辑，flow 是 "从当前像素指向源像素的位移"
        # 所以 Source_Coord = Target_Coord + Flow
        vgrid = self.grid + flow 
        
        # 3. 归一化到 [-1, 1]
        # 公式: 2 * x / (W-1) - 1
        vgrid[:, 0, :, :] = 2.0 * vgrid[:, 0, :, :] / max(W - 1, 1) - 1.0
        vgrid[:, 1, :, :] = 2.0 * vgrid[:, 1, :, :] / max(H - 1, 1) - 1.0
        
        # 4. 调整维度为 [B, H, W, 2] (grid_sample 要求)
        vgrid = vgrid.permute(0, 2, 3, 1)
        
        # 5. 采样
        warped = F.grid_sample(
            image, 
            vgrid, 
            mode='bilinear', 
            align_corners=False, # PyTorch 光流标准通常是 align_corners=True/False 取决于训练，但 False 比较通用
            padding_mode='border' # 边界处理：复用边界像素，防止黑边
        )
        
        return warped

def advect_frames(generated_frames, flow_fields):
    """
    将生成的帧序列进行平流对齐，准备喂给 D_t
    
    Args:
        generated_frames: [gen_prev, gen_curr, gen_next] (都是高分辨率)
        flow_fields: [flow_curr_to_prev, flow_curr_to_next] (高分辨率光流)
                     对应 process_raft.py 中的 backward 和 forward
    """
    advection = AdvectionLayer()
    
    # 1. 解包
    gen_prev, gen_curr, gen_next = generated_frames
    
    # flow_curr_to_prev: 在 t 时刻，指向 t-1 的位移 (backward.npy)
    # flow_curr_to_next: 在 t 时刻，指向 t+1 的位移 (forward.npy)
    flow_c2p, flow_c2n = flow_fields 
    
    # 2. Warp
    # 把 prev 扭曲到 curr
    warped_prev = advection(gen_prev, flow_c2p)
    
    # 把 next 扭曲到 curr
    warped_next = advection(gen_next, flow_c2n)
    
    # 3. 返回堆叠后的结果
    # 顺序：[Warped(t-1), Real(t), Warped(t+1)]
    return [warped_prev, gen_curr, warped_next]