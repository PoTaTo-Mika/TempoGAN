import argparse
import json
import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist

# 引入自定义模块
from datasets.dataset import TyphoonDataset
from models.tempogan import Generator, SpatialDiscriminator, TemporalDiscriminator
from utils.loss import TempoGANLoss
from utils.advection import advect_frames
from utils.utils import save_checkpoint, save_sample_images, AverageMeter

def setup_ddp():
    # 从环境变量读取配置 (torchrun 会自动设置这些)
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ["LOCAL_RANK"])
        
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend='nccl', init_method='env://', world_size=world_size, rank=rank)
        return local_rank, rank, world_size
    else:
        # 单机单卡调试模式
        print("Warning: Not using DDP. Running in single process mode.")
        return 0, 0, 1

def cleanup_ddp():
    if dist.is_initialized():
        dist.destroy_process_group()

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config.json', help='Path to config file')
    args = parser.parse_args()

    # 1. 加载配置
    with open(args.config, 'r') as f:
        config = json.load(f)

    # 2. DDP 初始化
    local_rank, rank, world_size = setup_ddp()
    device = torch.device(f"cuda:{local_rank}")

    # 创建目录 (仅主进程)
    if rank == 0:
        os.makedirs(config['paths']['checkpoint_dir'], exist_ok=True)
        os.makedirs(config['paths']['sample_dir'], exist_ok=True)
        os.makedirs(config['paths']['log_dir'], exist_ok=True)
        print(f"Starting training with {world_size} GPUs. Experiment: {config['experiment_name']}")

    # 3. 数据集与DataLoader
    dataset = TyphoonDataset(
        json_path=config['paths']['json_path'],
        root_dir=config['paths']['data_root'],
        flow_dir=config['paths']['flow_root'],
        scale_factor=config['model']['scale_factor']
    )

    sampler = DistributedSampler(dataset, shuffle=True) if world_size > 1 else None
    
    dataloader = DataLoader(
        dataset,
        batch_size=config['training']['batch_size'],
        shuffle=(sampler is None),
        sampler=sampler,
        num_workers=config['training']['num_workers'],
        pin_memory=True,
        drop_last=True
    )

    # 4. 初始化模型
    # 密度(1) + 速度(2) = 3 channels
    G = Generator(input_channels=config['model']['in_channels']).to(device)
    Ds = SpatialDiscriminator().to(device)
    Dt = TemporalDiscriminator().to(device)

    # 权重初始化
    G.apply(weights_init)
    Ds.apply(weights_init)
    Dt.apply(weights_init)

    # SyncBatchNorm (DDP 关键)
    if world_size > 1:
        G = nn.SyncBatchNorm.convert_sync_batchnorm(G)
        Ds = nn.SyncBatchNorm.convert_sync_batchnorm(Ds)
        Dt = nn.SyncBatchNorm.convert_sync_batchnorm(Dt)
        
        # Wrap DDP
        # broadcast_buffers=False 也是 GAN 训练的一个技巧，但 SyncBN 通常处理好了
        G = DDP(G, device_ids=[local_rank], output_device=local_rank)
        Ds = DDP(Ds, device_ids=[local_rank], output_device=local_rank)
        Dt = DDP(Dt, device_ids=[local_rank], output_device=local_rank)

    # 5. 优化器与损失函数
    optimizer_G = optim.Adam(G.parameters(), lr=config['training']['learning_rate']['g'], betas=tuple(config['training']['betas']))
    optimizer_Ds = optim.Adam(Ds.parameters(), lr=config['training']['learning_rate']['d'], betas=tuple(config['training']['betas']))
    optimizer_Dt = optim.Adam(Dt.parameters(), lr=config['training']['learning_rate']['d'], betas=tuple(config['training']['betas']))

    criterion = TempoGANLoss(
        device=device,
        lambda_l1=config['loss_weights']['lambda_l1'],
        lambda_feat=config['loss_weights']['lambda_feat'],
        lambda_adv=config['loss_weights']['lambda_adv']
    )

    # 6. 训练循环
    global_step = 0
    start_time = time.time()
    
    # 记录器
    loss_meter_G = AverageMeter()
    loss_meter_Ds = AverageMeter()
    loss_meter_Dt = AverageMeter()

    for epoch in range(config['training']['num_epochs']):
        if sampler:
            sampler.set_epoch(epoch)
        
        for i, batch in enumerate(dataloader):
            # ---------------------------------------------------------
            # A. 数据解包与重塑
            # ---------------------------------------------------------
            # lr_stack: [B, 3, 3, 128, 128] (t-1, t, t+1)
            # hr_stack: [B, 3, 1, 512, 512]
            lr_stack = batch['lr_stack'].to(device)
            hr_stack = batch['hr_stack'].to(device)
            
            # Flow 用于 Advection (High Res)
            flow_bwd = batch['flow_bwd'].to(device) # t -> t-1
            flow_fwd = batch['flow_fwd'].to(device) # t -> t+1

            B, T, C_in, H_lr, W_lr = lr_stack.shape
            _, _, C_out, H_hr, W_hr = hr_stack.shape

            # Flatten Batch and Time dimensions for Generator inference
            # [B*3, 3, 128, 128]
            lr_input_flat = lr_stack.view(-1, C_in, H_lr, W_lr)
            
            # ---------------------------------------------------------
            # B. 生成器前向传播
            # ---------------------------------------------------------
            gen_hr_flat = G(lr_input_flat) # -> [B*3, 1, 512, 512]
            
            # Reshape back to [B, 3, 1, 512, 512]
            gen_stack = gen_hr_flat.view(B, T, C_out, H_hr, W_hr)
            
            # Split frames
            gen_prev = gen_stack[:, 0] # t-1
            gen_curr = gen_stack[:, 1] # t
            gen_next = gen_stack[:, 2] # t+1
            
            hr_prev = hr_stack[:, 0]
            hr_curr = hr_stack[:, 1]
            hr_next = hr_stack[:, 2]
            
            # G 输入的中间帧 (用于 D_s 和 特征损失)
            lr_curr = lr_stack[:, 1, 0:1] # 取 Density 通道 [B, 1, 128, 128]
            # 实际上 D_s 输入通常是 resize 后的 LR Density
            # dataset 中 lr_stack 的第一个通道就是 density
            
            # ---------------------------------------------------------
            # C. Advection 对齐 (Preparation for D_t)
            # ---------------------------------------------------------
            # 使用 Advection Layer 对齐生成图像
            # 注意：advect_frames 返回 [Warped_prev, Real_curr, Warped_next] 的列表
            # 我们需要把 gen_curr 传进去当做 "Real_curr" 位置来对其进行对齐检查？
            # 不，advect_frames 的逻辑是把 prev 和 next 扭曲到 curr
            fake_triplet_aligned = advect_frames(
                [gen_prev, gen_curr, gen_next], 
                [flow_bwd, flow_fwd]
            )
            
            # 对齐真实图像 (作为 D_t 的正样本)
            # 注意这里要 detach flow 吗？通常 flow 是固定的，没有梯度
            real_triplet_aligned = advect_frames(
                [hr_prev, hr_curr, hr_next],
                [flow_bwd, flow_fwd]
            )
            
            # ---------------------------------------------------------
            # D. 更新 Discriminators (Ds & Dt)
            # ---------------------------------------------------------
            # 1. Update Spatial Discriminator (Ds)
            lr_curr_upsampled = torch.nn.functional.interpolate(
                lr_curr, size=(H_hr, W_hr), mode='nearest'
            )
            
            optimizer_Ds.zero_grad()
            # Ds 判断中间帧 t
            loss_Ds = criterion.calc_ds_loss(Ds, lr_curr_upsampled, hr_curr, gen_curr.detach())
            loss_Ds.backward()
            optimizer_Ds.step()
            
            # 2. Update Temporal Discriminator (Dt)
            optimizer_Dt.zero_grad()
            loss_Dt = criterion.calc_dt_loss(Dt, real_triplet_aligned, fake_triplet_aligned)
            loss_Dt.backward()
            optimizer_Dt.step()
            
            # ---------------------------------------------------------
            # E. 更新 Generator (G)
            # ---------------------------------------------------------
            optimizer_G.zero_grad()
            
            # 重新计算 G 的损失 (Adversarial + Content)
            # 注意：这里需要 advected frames 保持梯度，所以不能 detach
            # calc_g_loss 内部会再次调用 D(G(x)) 来获取梯度
            
            loss_G, logs_G = criterion.calc_g_loss(
                Ds, Dt, 
                lr_curr, hr_curr, gen_curr, 
                fake_triplet_aligned # 带有梯度的 aligned frames
            )
            
            loss_G.backward()
            optimizer_G.step()
            
            # ---------------------------------------------------------
            # F. 日志与保存
            # ---------------------------------------------------------
            loss_meter_G.update(loss_G.item())
            loss_meter_Ds.update(loss_Ds.item())
            loss_meter_Dt.update(loss_Dt.item())
            
            global_step += 1
            
            if rank == 0:
                if global_step % config['training']['log_interval'] == 0:
                    print(f"Epoch [{epoch}/{config['training']['num_epochs']}] "
                          f"Step [{i}/{len(dataloader)}] "
                          f"Loss_G: {loss_meter_G.avg:.4f} "
                          f"Loss_Ds: {loss_meter_Ds.avg:.4f} "
                          f"Loss_Dt: {loss_meter_Dt.avg:.4f} "
                          f"L1: {logs_G['g_l1']:.4f} "
                          f"Time: {time.time() - start_time:.1f}s")
                    loss_meter_G.reset()
                    loss_meter_Ds.reset()
                    loss_meter_Dt.reset()

                if global_step % config['training']['eval_interval'] == 0:
                    save_sample_images(
                        global_step, 
                        lr_stack, hr_stack, gen_stack, 
                        config['paths']['sample_dir']
                    )

                if global_step % config['training']['save_interval'] == 0:
                    save_checkpoint({
                        'epoch': epoch,
                        'global_step': global_step,
                        'state_dict_G': G.module.state_dict(), # 注意 .module
                        'state_dict_Ds': Ds.module.state_dict(),
                        'state_dict_Dt': Dt.module.state_dict(),
                        'optimizer_G': optimizer_G.state_dict(),
                    }, config['paths']['checkpoint_dir'], f'checkpoint_{global_step}.pth')

    cleanup_ddp()

if __name__ == "__main__":
    main()