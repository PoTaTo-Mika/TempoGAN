import torch
import torch.nn as nn
import torch.nn.functional as F

class PhyCell_Cell(nn.Module):
    def __init__(self, input_dim, F_hidden_dim, kernel_size, bias=1):
        super(PhyCell_Cell, self).__init__()
        self.input_dim = input_dim
        self.F_hidden_dim = F_hidden_dim
        self.kernel_size = kernel_size
        self.padding = kernel_size[0] // 2, kernel_size[1] // 2
        self.bias = bias

        # F: 物理预测器 (模拟 PDE 算子)
        self.F = nn.Sequential(
            nn.Conv2d(in_channels=input_dim, out_channels=F_hidden_dim, 
                      kernel_size=self.kernel_size, stride=1, padding=self.padding),
            nn.GroupNorm(4, F_hidden_dim), # GroupNorm 比 BatchNorm 对小 Batch 更稳定
            nn.Conv2d(in_channels=F_hidden_dim, out_channels=input_dim, 
                      kernel_size=(1, 1), stride=1, padding=0)
        )

        # K: 校正门 (Kalman Gain)
        self.convgate = nn.Conv2d(in_channels=self.input_dim * 2,
                                  out_channels=self.input_dim,
                                  kernel_size=(3, 3),
                                  padding=1, bias=self.bias)

    def forward(self, x, hidden):
        # x: 当前时刻的输入特征 [B, C, H, W]
        # hidden: 上一时刻的物理状态 [B, C, H, W]
        
        # 1. 计算 Prediction (基于物理规律推演下一步)
        hidden_tilde = hidden + self.F(hidden)
        
        # 2. 计算 Correction (结合当前输入 x 进行修正)
        combined = torch.cat([x, hidden_tilde], dim=1)
        K = torch.sigmoid(self.convgate(combined))
        
        # 3. 更新状态
        next_hidden = hidden_tilde + K * (x - hidden_tilde)
        
        return next_hidden

class PhyCell(nn.Module):
    def __init__(self, input_dim, F_hidden_dims, n_layers, kernel_size):
        super(PhyCell, self).__init__()
        self.input_dim = input_dim
        self.F_hidden_dims = F_hidden_dims
        self.n_layers = n_layers
        self.kernel_size = kernel_size
        
        cell_list = []
        for i in range(self.n_layers):
            cell_list.append(PhyCell_Cell(
                input_dim=input_dim,
                F_hidden_dim=self.F_hidden_dims[i],
                kernel_size=self.kernel_size
            ))
        self.cell_list = nn.ModuleList(cell_list)

    def forward(self, input_tensor, hidden_states=None):
        """
        input_tensor: [B, T, C, H, W] 或者 [B, C, H, W] (单步)
        """
        # 如果输入是 5D Tensor [B, T, C, H, W]，则进行时间步循环
        if input_tensor.dim() == 5:
            B, T, C, H, W = input_tensor.size()
            outputs = []
            
            # 初始化 Hidden States
            if hidden_states is None:
                hidden_states = self.init_hidden(B, H, W, input_tensor.device)
                
            for t in range(T):
                x_t = input_tensor[:, t]
                new_hidden_states = []
                
                # 多层 PhyCell 堆叠
                for i, cell in enumerate(self.cell_list):
                    h_prev = hidden_states[i]
                    
                    # 第一层的输入是图像特征，后续层的输入是上一层的隐状态
                    cell_input = x_t if i == 0 else new_hidden_states[-1]
                    
                    h_next = cell(cell_input, h_prev)
                    new_hidden_states.append(h_next)
                
                hidden_states = new_hidden_states
                outputs.append(hidden_states[-1]) # 取最后一层的输出
            
            # 堆叠时间步 [B, T, C, H, W]
            return torch.stack(outputs, dim=1), hidden_states
            
        else:
            # 单步处理 (用于推理或 simple loop)
            # input_tensor: [B, C, H, W]
            x_t = input_tensor
            B, C, H, W = x_t.size()
            
            if hidden_states is None:
                hidden_states = self.init_hidden(B, H, W, x_t.device)
                
            new_hidden_states = []
            for i, cell in enumerate(self.cell_list):
                h_prev = hidden_states[i]
                cell_input = x_t if i == 0 else new_hidden_states[-1]
                h_next = cell(cell_input, h_prev)
                new_hidden_states.append(h_next)
                
            return new_hidden_states[-1], new_hidden_states

    def init_hidden(self, batch_size, height, width, device):
        hidden = []
        for _ in range(self.n_layers):
            hidden.append(torch.zeros(batch_size, self.input_dim, height, width).to(device))
        return hidden