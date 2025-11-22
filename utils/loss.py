import torch
import torch.nn as nn
import torch.nn.functional as F

class TempoGANLoss(nn.Module):
    def __init__(self, device, lambda_l1=5.0, lambda_feat=10.0, lambda_adv=1.0):
        super(TempoGANLoss, self).__init__()
        self.device = device
        self.lambda_l1 = lambda_l1
        self.lambda_feat = lambda_feat
        self.lambda_adv = lambda_adv
        
        # 使用 BCEWithLogitsLoss 比手动 Sigmoid + BCELoss 数值更稳定
        self.criterion_gan = nn.BCEWithLogitsLoss()
        self.criterion_l1 = nn.L1Loss()
        self.criterion_feat = nn.MSELoss()

    def get_gan_labels(self, preds, target_is_real):
        """生成真/假标签"""
        # preds 是 logit (未经过 sigmoid)
        if target_is_real:
            labels = torch.ones_like(preds, device=self.device)
        else:
            labels = torch.zeros_like(preds, device=self.device)
        return labels

    def calc_ds_loss(self, D_s, lr_input, hr_real, hr_fake):
        """
        计算空间判别器 D_s 的损失
        Loss = E[-log(D(y))] + E[-log(1-D(G(x)))]
        """
        # 1. Real Loss (D_s 应该把真图判为 1)
        # D_s 输入是 concat(low_res, high_res)
        pred_real, _ = D_s(lr_input, hr_real)
        loss_real = self.criterion_gan(pred_real, self.get_gan_labels(pred_real, True))
        
        # 2. Fake Loss (D_s 应该把生成图判为 0)
        # .detach() 很重要：训练 D 时不需要更新 G 的梯度
        pred_fake, _ = D_s(lr_input, hr_fake.detach())
        loss_fake = self.criterion_gan(pred_fake, self.get_gan_labels(pred_fake, False))
        
        return (loss_real + loss_fake) * 0.5

    def calc_dt_loss(self, D_t, real_triplet, fake_triplet_aligned):
        """
        计算时间判别器 D_t 的损失
        输入已经是经过 Advection 对齐好的 triplet: [Warped_Prev, Curr, Warped_Next]
        """
        # 拼接 triplet 用于输入 D_t (B, 3, H, W)
        real_input = torch.cat(real_triplet, dim=1)
        fake_input = torch.cat(fake_triplet_aligned, dim=1).detach() # 记得 detach
        
        # 1. Real Loss
        pred_real = D_t(real_input)
        loss_real = self.criterion_gan(pred_real, self.get_gan_labels(pred_real, True))
        
        # 2. Fake Loss
        pred_fake = D_t(fake_input)
        loss_fake = self.criterion_gan(pred_fake, self.get_gan_labels(pred_fake, False))
        
        return (loss_real + loss_fake) * 0.5

    # =========================================================================
    # 生成器 Loss (G)
    # =========================================================================

    def calc_g_loss(self, D_s, D_t, lr_input, hr_real, hr_fake, fake_triplet_aligned):
        """
        计算生成器总损失
        Loss_G = L_adv(Ds) + L_adv(Dt) + L1 + L_feat
        """
        logs = {}
        
        # 1. Spatial Adversarial Loss (骗过 D_s)
        pred_fake_s, feat_fake_s = D_s(lr_input, hr_fake)
        loss_adv_s = self.criterion_gan(pred_fake_s, self.get_gan_labels(pred_fake_s, True))
        
        # 2. Temporal Adversarial Loss (骗过 D_t)
        # 输入是对齐后的生成序列
        fake_dt_input = torch.cat(fake_triplet_aligned, dim=1)
        pred_fake_t = D_t(fake_dt_input)
        loss_adv_t = self.criterion_gan(pred_fake_t, self.get_gan_labels(pred_fake_t, True))
        
        # 3. L1 Loss (像素级相似度，针对中间帧 t)
        loss_l1 = self.criterion_l1(hr_fake, hr_real)
        
        # 4. Feature Space Loss (感知损失)
        # 需要计算真实图片的 D_s 特征作为目标
        # 这里 D_s 处于 eval 模式或不计算梯度，我们只需要特征
        with torch.no_grad():
            _, feat_real_s = D_s(lr_input, hr_real)
            
        loss_feat = 0.0
        # 遍历 D_s 的每一层特征 (features_list)
        # 论文 Appendix B 提到权重可能是负的，但通常正权重做 Feature Matching 更稳定
        # 如果你想复现论文的"负权重"技巧，可以在这里乘以 -1，但建议先用正的
        weights = [1.0, 1.0, 1.0, 1.0] # 可以根据层深度调整
        for i, (f_real, f_fake) in enumerate(zip(feat_real_s, feat_fake_s)):
            loss_feat += self.criterion_feat(f_fake, f_real) * weights[i]
            
        # 总损失
        total_loss = (loss_adv_s * self.lambda_adv + 
                      loss_adv_t * self.lambda_adv + 
                      loss_l1 * self.lambda_l1 + 
                      loss_feat * self.lambda_feat)
        
        # 记录日志
        logs['g_adv_s'] = loss_adv_s.item()
        logs['g_adv_t'] = loss_adv_t.item()
        logs['g_l1'] = loss_l1.item()
        logs['g_feat'] = loss_feat.item()
        
        return total_loss, logs