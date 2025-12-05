import argparse
import json
import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP

# 引入模块
from datasets.dataset import PhyTyphoonDataset
from models.phy_tempogan_v3 import PhyGenerator, SpatialDiscriminator, TemporalDiscriminator
from utils.utils import save_checkpoint, save_sample_images, AverageMeter

# ==============================================================================
# PhyGAN Loss: 专门为 PhyTempoGAN 定制的损失函数
# ==============================================================================
class PhyGANLoss(nn.Module):
    def __init__(self, device, lambda_l1=10.0, lambda_feat=5.0, lambda_adv=1.0):
        super(PhyGANLoss, self).__init__()
        self.device = device
        self.lambda_l1 = lambda_l1
        self.lambda_feat = lambda_feat
        self.lambda_adv = lambda_adv
        
        self.criterion_l1 = nn.L1Loss()
        self.criterion_feat = nn.MSELoss()

    def calc_ds_loss(self, D_s, lr, hr, fake):
        real_pred, _ = D_s(lr, hr)
        fake_pred, _ = D_s(lr, fake.detach())
        
        loss_real = torch.mean(torch.nn.functional.relu(1.0 - real_pred))
        loss_fake = torch.mean(torch.nn.functional.relu(1.0 + fake_pred))
        
        return loss_real + loss_fake

    def calc_dt_loss(self, D_t, real_seq, fake_seq):
        real_pred = D_t(real_seq)
        fake_pred = D_t(fake_seq.detach())
        
        loss_real = torch.mean(torch.nn.functional.relu(1.0 - real_pred))
        loss_fake = torch.mean(torch.nn.functional.relu(1.0 + fake_pred))
        
        return loss_real + loss_fake

    def calc_g_loss(self, D_s, D_t, lr, hr, gen, fake_seq):
        logs = {}
        
        # 1. Adversarial Loss
        pred_fake_s, feat_fake_s = D_s(lr, gen)
        loss_adv_s = -torch.mean(pred_fake_s)
        
        pred_fake_t = D_t(fake_seq)
        loss_adv_t = -torch.mean(pred_fake_t)
        
        # 2. L1 Loss
        loss_l1 = self.criterion_l1(gen, hr)
        
        # 3. Feature Matching Loss
        with torch.no_grad():
            _, feat_real_s = D_s(lr, hr)
            
        loss_feat = 0.0
        weights = [1.0, 1.0, 1.0, 1.0] 
        for i, (f_r, f_f) in enumerate(zip(feat_real_s, feat_fake_s)):
            loss_feat += self.criterion_feat(f_f, f_r) * weights[i]
            
        # Total Loss
        total_loss = (loss_adv_s * self.lambda_adv + 
                      loss_adv_t * self.lambda_adv + 
                      loss_l1 * self.lambda_l1 + 
                      loss_feat * self.lambda_feat)
        
        # Logs
        logs['g_adv_s'] = loss_adv_s.item()
        logs['g_adv_t'] = loss_adv_t.item()
        logs['g_l1'] = loss_l1.item()
        logs['g_feat'] = loss_feat.item()
        
        return total_loss, logs

# ==============================================================================
# Helper Functions
# ==============================================================================

def setup_ddp():
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend='nccl', init_method='env://', world_size=world_size, rank=rank)
        return local_rank, rank, world_size
    else:
        print("Running in single process mode.")
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

# ==============================================================================
# Main Training Loop
# ==============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config.json')
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = json.load(f)

    local_rank, rank, world_size = setup_ddp()
    device = torch.device(f"cuda:{local_rank}")

    if rank == 0:
        os.makedirs(config['paths']['checkpoint_dir'], exist_ok=True)
        os.makedirs(config['paths']['sample_dir'], exist_ok=True)
        os.makedirs(config['paths']['log_dir'], exist_ok=True)
        print(f"Starting PhyGAN Training. Experiment: {config['experiment_name']}")

    # 1. Dataset
    dataset = PhyTyphoonDataset(
        json_path=config['paths']['json_path'],
        root_dir=config['paths']['data_root'],
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

    # 2. Model Initialization
    G = PhyGenerator(
        scale_factor=config['model']['scale_factor'],
        in_channels=1,
        base_channels=64
    ).to(device)
    
    Ds = SpatialDiscriminator(input_channels=2).to(device)
    Dt = TemporalDiscriminator(input_channels=3).to(device)

    G.apply(weights_init)
    Ds.apply(weights_init)
    Dt.apply(weights_init)

    if world_size > 1:
        G = nn.SyncBatchNorm.convert_sync_batchnorm(G)
        Ds = nn.SyncBatchNorm.convert_sync_batchnorm(Ds)
        Dt = nn.SyncBatchNorm.convert_sync_batchnorm(Dt)
        G = DDP(G, device_ids=[local_rank], output_device=local_rank)
        Ds = DDP(Ds, device_ids=[local_rank], output_device=local_rank)
        Dt = DDP(Dt, device_ids=[local_rank], output_device=local_rank)

    # 3. Optimizers
    opt_G = optim.Adam(G.parameters(), lr=config['training']['learning_rate']['g'], betas=tuple(config['training']['betas']))
    opt_Ds = optim.Adam(Ds.parameters(), lr=config['training']['learning_rate']['d'], betas=tuple(config['training']['betas']))
    opt_Dt = optim.Adam(Dt.parameters(), lr=config['training']['learning_rate']['d'], betas=tuple(config['training']['betas']))

    # 4. Loss
    criterion = PhyGANLoss(device, lambda_l1=config['loss_weights']['lambda_l1'], lambda_feat=config['loss_weights']['lambda_feat'], lambda_adv=config['loss_weights']['lambda_adv'])

    # 5. Loop
    global_step = 0
    loss_m_G = AverageMeter()
    loss_m_Ds = AverageMeter()
    loss_m_Dt = AverageMeter()
    
    # 设置 Warmup Epochs 数量
    WARMUP_EPOCHS = 3

    for epoch in range(config['training']['num_epochs']):
        if sampler:
            sampler.set_epoch(epoch)
            
        # 判断当前是否为 Warmup 阶段
        is_warmup = (epoch < WARMUP_EPOCHS)
        
        if rank == 0:
            if is_warmup:
                print(f"Epoch {epoch}: Warmup Mode (L1 Only).")
            else:
                print(f"Epoch {epoch}: Full GAN Training Mode.")
            
        for i, batch in enumerate(dataloader):
            lr_seq = batch['lr_seq'].to(device) 
            hr_seq = batch['hr_seq'].to(device)

            B, T, C, H_lr, W_lr = lr_seq.shape
            _, _, _, H_hr, W_hr = hr_seq.shape
            
            # -------------------------------------------------------
            # Step 1: Run Generator
            # -------------------------------------------------------
            # 无论什么阶段，G 都要跑前向
            gen_hr_center = G(lr_seq)
            
            # 获取 LR 和 HR 的中间帧，用于 loss 计算
            lr_center = lr_seq[:, 1]
            lr_center_up = torch.nn.functional.interpolate(lr_center, size=(H_hr, W_hr), mode='nearest')
            hr_center = hr_seq[:, 1]

            # -------------------------------------------------------
            # 分支逻辑：Warmup vs Normal
            # -------------------------------------------------------
            if is_warmup:
                # ================= Warmup Phase =================
                # 只计算 L1 Loss，不更新判别器，不计算特征损失
                
                loss_l1 = criterion.criterion_l1(gen_hr_center, hr_center)
                
                # 为了保持 loss 数量级一致，建议乘上 config 中的 lambda_l1
                # 如果你想纯粹只是 L1，也可以去掉这个乘数，但为了后续切换顺滑，通常保留权重
                loss_G = loss_l1 * criterion.lambda_l1
                
                opt_G.zero_grad()
                loss_G.backward()
                opt_G.step()
                
                # 记录日志 (其他项置零)
                loss_Ds_val = 0.0
                loss_Dt_val = 0.0
                logs = {
                    'g_l1': loss_l1.item(),
                    'g_feat': 0.0,
                    'g_adv_s': 0.0,
                    'g_adv_t': 0.0
                }

            else:
                # ================= Normal Phase =================
                # 包含 判别器更新 + Full GAN Loss
                
                # 构造序列用于 Dt
                real_seq_3d = hr_seq
                fake_seq_list = [
                    hr_seq[:, 0],   # t-1
                    gen_hr_center,  # Generated t
                    hr_seq[:, 2]    # t+1
                ]
                fake_seq_3d = torch.stack(fake_seq_list, dim=1)

                # --- Update Ds ---
                opt_Ds.zero_grad()
                loss_Ds = criterion.calc_ds_loss(Ds, lr_center_up, hr_center, gen_hr_center.detach())
                loss_Ds.backward()
                opt_Ds.step()
                loss_Ds_val = loss_Ds.item()

                # --- Update Dt ---
                opt_Dt.zero_grad()
                loss_Dt = criterion.calc_dt_loss(Dt, real_seq_3d, fake_seq_3d.detach())
                loss_Dt.backward()
                opt_Dt.step()
                loss_Dt_val = loss_Dt.item()

                # --- Update G ---
                opt_G.zero_grad()
                loss_G, logs = criterion.calc_g_loss(Ds, Dt, lr_center_up, hr_center, gen_hr_center, fake_seq_3d)
                loss_G.backward()
                opt_G.step()

            # -------------------------------------------------------
            # Logging
            # -------------------------------------------------------
            loss_m_G.update(loss_G.item())
            loss_m_Ds.update(loss_Ds_val)
            loss_m_Dt.update(loss_Dt_val)
            global_step += 1

            if rank == 0:
                if global_step % config['training']['log_interval'] == 0:
                    status = "Warmup" if is_warmup else "GAN"
                    print(f"[{status}] Ep [{epoch}/{config['training']['num_epochs']}] "
                          f"Step [{i}/{len(dataloader)}] "
                          f"L_G:{loss_m_G.avg:.3f} L_Ds:{loss_m_Ds.avg:.3f} L_Dt:{loss_m_Dt.avg:.3f} "
                          f"L1:{logs['g_l1']:.3f} Feat:{logs['g_feat']:.3f}")
                    loss_m_G.reset()
                    loss_m_Ds.reset()
                    loss_m_Dt.reset()

                if global_step % config['training']['eval_interval'] == 0:
                    gen_seq_vis = torch.stack([gen_hr_center, gen_hr_center, gen_hr_center], dim=1)
                    save_sample_images(
                        global_step, 
                        lr_seq,
                        hr_seq,
                        gen_seq_vis,
                        config['paths']['sample_dir']
                    )

                if global_step % config['training']['save_interval'] == 0:
                    save_checkpoint({
                        'epoch': epoch,
                        'state_dict_G': G.module.state_dict() if world_size > 1 else G.state_dict(),
                        'optimizer_G': opt_G.state_dict(),
                    }, config['paths']['checkpoint_dir'], f'phygan_step_{global_step}.pth')

    cleanup_ddp()

if __name__ == "__main__":
    main()