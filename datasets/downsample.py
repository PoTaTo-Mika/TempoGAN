import torch
import torchvision
import torch.nn.functional as F

def downsample_image(image, downsample_factor):
    """
    对512x512灰度图像进行下采样
    
    Args:
        image: 输入图像，形状为(1, 512, 512)或(512, 512)的tensor
        downsample_factor: 下采样因子，支持2、4、8
    
    Returns:
        下采样后的图像tensor
    """
    assert downsample_factor in [2, 4, 8], "downsample_factor必须是2、4或8"
    
    # 确保图像是3D tensor (channels, height, width)
    if image.dim() == 2:
        image = image.unsqueeze(0)  # 添加通道维度
    
    # 计算目标尺寸
    target_height = 512 // downsample_factor
    target_width = 512 // downsample_factor
    
    # 使用双线性插值进行下采样
    downsampled = F.interpolate(
        image.unsqueeze(0),  # 添加batch维度
        size=(target_height, target_width),
        mode='bilinear',
        align_corners=False
    )
    
    # 移除batch维度，返回与输入相同的维度格式
    result = downsampled.squeeze(0)
    if image.dim() == 2:
        result = result.squeeze(0)  # 如果输入是2D，返回2D
    
    return result

# 使用示例
if __name__ == "__main__":
    # 创建示例图像
    sample_image = torch.randn(512, 512)
    
    # 测试2倍下采样
    downsampled_2x = downsample_image(sample_image, 2)
    print(f"2倍下采样后尺寸: {downsampled_2x.shape}")
    
    # 测试4倍下采样
    downsampled_4x = downsample_image(sample_image, 4)
    print(f"4倍下采样后尺寸: {downsampled_4x.shape}")
    
    # 测试8倍下采样
    downsampled_8x = downsample_image(sample_image, 8)
    print(f"8倍下采样后尺寸: {downsampled_8x.shape}")