import os
import argparse
import json
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

# 导入模型定义
from models.phy_tempogan import PhyGenerator
from models.tempogan import Generator
# 导入光流计算 (仅 TempoGAN 需要)
from datasets.raft import compute_optical_flow 

def load_image_tensor(path, device):
    """
    加载图片并预处理
    Returns: [1, 1, H, W] in range [-1, 1]
    """
    try:
        img = Image.open(path).convert('L')
        img_np = np.array(img)
        tensor = torch.from_numpy(img_np).float().unsqueeze(0).unsqueeze(0) # [B, C, H, W]
        tensor = (tensor / 127.5) - 1.0
        return tensor.to(device)
    except Exception as e:
        print(f"Error loading {path}: {e}")
        return None

def tensor_to_image(tensor):
    """
    将 [-1, 1] 的 Tensor 转回 uint8 图片
    """
    img = tensor.squeeze().detach().cpu().numpy()
    # Clip to ensure valid range
    img = np.clip(img, -1.0, 1.0)
    img = (img + 1.0) / 2.0 * 255.0
    return img.clip(0, 255).astype(np.uint8)

def downsample(tensor, scale_factor=4, is_flow=False):
    """
    模拟低分辨率输入: HR -> Downsample -> LR
    """
    mode = 'bilinear' if is_flow else 'area'
    
    # F.interpolate 需要 [B, C, H, W]
    lr = F.interpolate(
        tensor, 
        scale_factor=1/scale_factor, 
        mode=mode,
        align_corners=False if mode=='bilinear' else None,
        recompute_scale_factor=True
    )
    
    if is_flow:
        # 光流的数值大小也需要随着分辨率缩小而缩小
        lr = lr / scale_factor
        
    return lr

# ==============================================================================
# 核心推理逻辑：PhyGAN (Sequence Based)
# ==============================================================================
def run_phygan_inference(model, triplets, output_dir, device, scale_factor=4):
    print(f"Mode: PhyTempoGAN (Physics-aware)")
    print(f"Total Triplets: {len(triplets)}")
    
    model.eval()
    
    for i, triplet in enumerate(tqdm(triplets)):
        path_prev, path_curr, path_next = triplet
        
        # 1. 加载原图 (High Res GT)
        hr_prev = load_image_tensor(path_prev, device)
        hr_curr = load_image_tensor(path_curr, device)
        hr_next = load_image_tensor(path_next, device)
        
        if hr_prev is None or hr_curr is None or hr_next is None:
            continue
            
        # 2. 下采样模拟 LR 输入
        # 注意：这里我们模拟的是 "由低分观测重建高分" 的过程
        lr_prev = downsample(hr_prev, scale_factor)
        lr_curr = downsample(hr_curr, scale_factor)
        lr_next = downsample(hr_next, scale_factor)
        
        # 3. 堆叠成序列 [1, 3, 1, H_lr, W_lr]
        # view as [B, T, C, H, W]
        lr_seq = torch.stack([lr_prev, lr_curr, lr_next], dim=1)
        
        # 4. 推理
        with torch.no_grad():
            # Output: [1, 1, H_hr, W_hr]
            gen_hr = model(lr_seq)
        
        # 5. 保存结果 (保留原始文件名)
        save_name = os.path.basename(path_curr)
        save_path = os.path.join(output_dir, save_name)
        Image.fromarray(tensor_to_image(gen_hr)).save(save_path)

# ==============================================================================
# 核心推理逻辑：TempoGAN (Flow Based)
# ==============================================================================
def run_tempogan_inference(model, triplets, output_dir, device, scale_factor=4):
    print(f"Mode: TempoGAN (Classic Flow-guided)")
    print(f"Total Triplets: {len(triplets)}")
    
    model.eval()
    
    for i, triplet in enumerate(tqdm(triplets)):
        _, path_curr, path_next = triplet # TempoGAN需要 t 和 t+1 计算光流
        
        # 1. 加载图像 (需要 numpy 格式计算光流)
        try:
            img_curr_pil = Image.open(path_curr).convert('L')
            img_next_pil = Image.open(path_next).convert('L')
        except Exception as e:
            print(f"Error opening images: {e}")
            continue
            
        img_curr_np = np.array(img_curr_pil)
        img_next_np = np.array(img_next_pil)
        
        # 2. 实时计算 High-Res 光流 (作为 Ground Truth Velocity)
        # TempoGAN 的逻辑是：有了高分图像才有了精确光流，
        # 但在推理时，我们假设只有低分输入。
        # 按照论文/标准流程，我们在这里先计算 HR 光流，然后下采样它，
        # 模拟"假如我们在低分辨率下估计出了光流" (或者直接对 LR 图像算光流也可以，但为了对其训练时的 degrade 过程，通常是 HR Flow -> Downsample)
        flow_hr_np = compute_optical_flow(img_curr_np, img_next_np)
        
        # 3. 准备 Tensor
        hr_curr = load_image_tensor(path_curr, device) # [1, 1, 512, 512]
        hr_flow = torch.from_numpy(flow_hr_np).float().to(device).unsqueeze(0) # [1, 2, 512, 512]
        
        # 4. 下采样生成输入
        # Density: [1, 1, 128, 128]
        lr_curr = downsample(hr_curr, scale_factor)
        # Velocity: [1, 2, 128, 128] (注意 flow 数值也要除以 scale)
        lr_flow = downsample(hr_flow, scale_factor, is_flow=True)
        
        # 5. 归一化 Velocity
        # 这一点至关重要，必须与训练时的 dataset.py 保持一致
        # 假设训练时 max_flow_mag = 50.0
        MAX_FLOW = 50.0 
        lr_flow_norm = torch.clamp(lr_flow / (MAX_FLOW / scale_factor), -1.0, 1.0)
        
        # 6. 拼接输入 [1, 3, 128, 128]
        lr_input = torch.cat([lr_curr, lr_flow_norm], dim=1)
        
        # 7. 推理
        with torch.no_grad():
            gen_hr = model(lr_input)
            
        # 8. 保存
        save_name = os.path.basename(path_curr)
        save_path = os.path.join(output_dir, save_name)
        Image.fromarray(tensor_to_image(gen_hr)).save(save_path)

def main():
    parser = argparse.ArgumentParser(description="TempoGAN/PhyTempoGAN Inference Script")
    parser.add_argument('--config_name', type=str, required=True, 
                       help='Name of the configuration (used for output directory naming)')
    parser.add_argument('--triplets_json', type=str, required=True, 
                       help='Path to the triplets JSON file (generated by shuffle_data.py)')
    parser.add_argument('--model_type', type=str, required=True, choices=['phy', 'tempo'], 
                       help='Choose model architecture')
    parser.add_argument('--checkpoint', type=str, required=True, 
                       help='Path to the trained .pth checkpoint')
    parser.add_argument('--scale_factor', type=int, default=4, 
                       help='Downsampling factor for SR (default: 4)')
    
    args = parser.parse_args()

    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 创建输出目录
    output_dir = os.path.join('outputs', f'inference_results_{args.config_name}')
    os.makedirs(output_dir, exist_ok=True)
    print(f"Results will be saved to: {output_dir}")

    # 1. 加载 Triplet JSON
    print(f"Loading triplets from {args.triplets_json}...")
    with open(args.triplets_json, 'r') as f:
        triplets = json.load(f)
    
    # 2. 初始化模型
    print(f"Loading {args.model_type} model...")
    if args.model_type == 'phy':
        # PhyGAN: Input channels = 1 (Density only)
        model = PhyGenerator(scale_factor=args.scale_factor, in_channels=1, base_channels=64).to(device)
    else:
        # TempoGAN: Input channels = 3 (Density + Velocity)
        model = Generator(scale_factor=args.scale_factor, input_channels=3).to(device)

    # 3. 加载权重
    print(f"Loading weights from {args.checkpoint}...")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    
    # 兼容不同的保存格式 (整个 dict 或者 state_dict_G)
    if isinstance(checkpoint, dict) and 'state_dict_G' in checkpoint:
        state_dict = checkpoint['state_dict_G']
    else:
        state_dict = checkpoint
        
    # 处理 DDP 训练留下的 `module.` 前缀
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('module.'):
            new_state_dict[k[7:]] = v
        else:
            new_state_dict[k] = v
            
    model.load_state_dict(new_state_dict)
    
    # 4. 开始推理
    if args.model_type == 'phy':
        run_phygan_inference(model, triplets, output_dir, device, args.scale_factor)
    else:
        run_tempogan_inference(model, triplets, output_dir, device, args.scale_factor)
        
    print(f"Inference finished. Check {output_dir}")

if __name__ == "__main__":
    main()