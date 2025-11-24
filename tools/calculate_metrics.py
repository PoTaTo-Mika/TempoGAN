import os
import argparse
import numpy as np
import cv2
from tqdm import tqdm
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

def calculate_rase(gt, pred):
    """
    计算 RASE (Relative Average Spectral Error)
    """
    gt = gt.astype(np.float64)
    pred = pred.astype(np.float64)
    rmse = np.sqrt(np.mean((gt - pred) ** 2))
    mean_gt = np.mean(gt)
    if mean_gt == 0:
        return 0.0
    return (rmse / mean_gt) * 100.0

def main():
    parser = argparse.ArgumentParser(description="Calculate PSNR, SSIM, and RASE")
    parser.add_argument('--gt_dir', type=str, required=True, help='Root directory containing GT images (can be nested)')
    parser.add_argument('--pred_dir', type=str, required=True, help='Directory containing Predicted images (flat structure)')
    
    args = parser.parse_args()
    
    psnr_list = []
    ssim_list = []
    rase_list = []
    count = 0
    
    valid_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.tif'}
    
    print(f"Start evaluating...")
    print(f"GT Root:   {args.gt_dir} (Recursive search)")
    print(f"Pred Root: {args.pred_dir} (Flat search)")
    
    # 1. 收集所有待处理的 GT 文件路径
    gt_files_list = []
    for root, dirs, files in os.walk(args.gt_dir):
        for file in files:
            if os.path.splitext(file)[1].lower() in valid_extensions:
                gt_files_list.append(os.path.join(root, file))

    # 2. 遍历 GT 文件并去 Pred 目录查找对应文件
    for gt_path in tqdm(gt_files_list, desc="Evaluating"):
        # 获取纯文件名 (例如 "20190101.png")，忽略 GT 的子目录结构
        filename = os.path.basename(gt_path)
        
        # 构造预测图的路径 (假设所有预测图都在 pred_dir 根目录下)
        pred_path = os.path.join(args.pred_dir, filename)
        
        # 检查是否存在
        if not os.path.exists(pred_path):
            # 很多时候只有中间帧有预测结果，首尾帧没有，这是正常的，直接跳过
            continue
            
        # 读取图像
        img_gt = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
        img_pred = cv2.imread(pred_path, cv2.IMREAD_GRAYSCALE)
        
        if img_gt is None or img_pred is None:
            continue
            
        # 尺寸校验与修正 (防止 padding 导致的微小差异)
        if img_gt.shape != img_pred.shape:
            img_pred = cv2.resize(img_pred, (img_gt.shape[1], img_gt.shape[0]))
            
        # 计算指标
        val_psnr = psnr(img_gt, img_pred, data_range=255)
        val_ssim = ssim(img_gt, img_pred, data_range=255)
        val_rase = calculate_rase(img_gt, img_pred)
        
        psnr_list.append(val_psnr)
        ssim_list.append(val_ssim)
        rase_list.append(val_rase)
        
        count += 1

    if count == 0:
        print("\nError: No matching files found!")
        print(f"Checked {len(gt_files_list)} files in GT.")
        print(f"Please check if filenames match between GT and Pred folders.")
        return

    avg_psnr = np.mean(psnr_list)
    avg_ssim = np.mean(ssim_list)
    avg_rase = np.mean(rase_list)
    
    print("\n" + "="*40)
    print("Evaluation Results")
    print("="*40)
    print(f"Matched Images: {count}")
    print("-" * 40)
    print(f"Average PSNR:  {avg_psnr:.4f} dB")
    print(f"Average SSIM:  {avg_ssim:.4f}")
    print(f"Average RASE:  {avg_rase:.4f} %")
    print("="*40)

if __name__ == "__main__":
    main()