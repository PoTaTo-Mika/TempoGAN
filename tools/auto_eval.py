import os
import glob
import json
import argparse
import subprocess
import shutil
import re
import csv
from natsort import natsorted # 建议安装: pip install natsort，如果不想装，可以用 sorted
import sys

# ================= 配置区域 =================
GT_DIR = "data/test"
CONFIG_NAME = "final_test"
MODEL_TYPE = "phy"
SCALE_FACTOR = 4
TEMP_JSON_NAME = "temp_test_triplets.json"
OUTPUT_DIR = os.path.join("outputs", f"inference_results_{CONFIG_NAME}")
CSV_FILENAME = "evaluation_results.csv"

def generate_test_json(gt_path, output_json):
    """
    遍历 data/test 生成时序三元组 (t-1, t, t+1)
    支持：data/test 下直接是图片，或者 data/test 下包含多个子序列文件夹
    """
    print(f"[Data] Scanning {gt_path} to generate triplets...")
    
    valid_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.tif'}
    triplets = []

    # 检查是否包含子文件夹
    items = sorted(os.listdir(gt_path))
    has_subdirs = any(os.path.isdir(os.path.join(gt_path, i)) for i in items)

    if has_subdirs:
        # 遍历每个子文件夹作为单独的序列
        for subdir in items:
            subdir_path = os.path.join(gt_path, subdir)
            if not os.path.isdir(subdir_path): continue
            
            imgs = sorted([f for f in os.listdir(subdir_path) if os.path.splitext(f)[1].lower() in valid_extensions])
            # 构建滑动窗口
            for i in range(1, len(imgs) - 1):
                prev_p = os.path.join(subdir_path, imgs[i-1])
                curr_p = os.path.join(subdir_path, imgs[i])
                next_p = os.path.join(subdir_path, imgs[i+1])
                triplets.append([prev_p, curr_p, next_p])
    else:
        # 根目录就是一个序列
        imgs = sorted([f for f in items if os.path.splitext(f)[1].lower() in valid_extensions])
        for i in range(1, len(imgs) - 1):
            prev_p = os.path.join(gt_path, imgs[i-1])
            curr_p = os.path.join(gt_path, imgs[i])
            next_p = os.path.join(gt_path, imgs[i+1])
            triplets.append([prev_p, curr_p, next_p])

    if len(triplets) == 0:
        print("Error: No valid triplets found in data/test!")
        sys.exit(1)

    with open(output_json, 'w') as f:
        json.dump(triplets, f, indent=4)
    
    print(f"[Data] Generated {len(triplets)} triplets in {output_json}")

def parse_metrics(output_text):
    """
    从 calculate_metrics.py 的控制台输出中提取数值
    """
    psnr = re.search(r"Average PSNR:\s+([0-9.]+)", output_text)
    ssim = re.search(r"Average SSIM:\s+([0-9.]+)", output_text)
    rase = re.search(r"Average RASE:\s+([0-9.]+)", output_text)
    
    return (
        float(psnr.group(1)) if psnr else 0.0,
        float(ssim.group(1)) if ssim else 0.0,
        float(rase.group(1)) if rase else 0.0
    )

def main():
    parser = argparse.ArgumentParser(description="Auto Evaluate All Checkpoints")
    parser.add_argument('--checkpoint_dir', type=str, required=True, 
                        help='Directory containing .pth files')
    args = parser.parse_args()

    # 1. 准备测试数据 JSON
    generate_test_json(GT_DIR, TEMP_JSON_NAME)

    # 2. 获取所有权重文件
    if not os.path.exists(args.checkpoint_dir):
        print(f"Error: Checkpoint directory {args.checkpoint_dir} not found.")
        return

    checkpoints = glob.glob(os.path.join(args.checkpoint_dir, "*.pth"))
    
    # 尝试自然排序 (Epoch 2 排在 Epoch 10 前面)
    try:
        checkpoints = natsorted(checkpoints)
    except:
        checkpoints.sort()

    print(f"Found {len(checkpoints)} checkpoints to evaluate.")

    # 3. 初始化 CSV
    csv_file = open(CSV_FILENAME, 'w', newline='')
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(['Checkpoint', 'PSNR', 'SSIM', 'RASE'])
    csv_file.flush() # 立即写入表头

    # 4. 循环评测
    for ckpt in checkpoints:
        ckpt_name = os.path.basename(ckpt)
        print("\n" + "#"*60)
        print(f"Processing: {ckpt_name}")
        print("#"*60)

        # A. 清理旧的输出结果 (至关重要，防止文件混淆)
        if os.path.exists(OUTPUT_DIR):
            shutil.rmtree(OUTPUT_DIR)
        
        # B. 运行推理 (inference.py)
        # 命令: python inference.py --config_name final_test --triplets_json temp.json ...
        inf_cmd = [
            "python", "inference.py",
            "--config_name", CONFIG_NAME,
            "--triplets_json", TEMP_JSON_NAME,
            "--model_type", MODEL_TYPE,
            "--checkpoint", ckpt,
            "--scale_factor", str(SCALE_FACTOR)
        ]
        
        try:
            subprocess.run(inf_cmd, check=True)
        except subprocess.CalledProcessError:
            print(f"Inference failed for {ckpt_name}, skipping...")
            continue

        # C. 运行评测 (calculate_metrics.py)
        # 命令: python calculate_metrics.py --gt_dir data/test --pred_dir outputs/...
        eval_cmd = [
            "python", "calculate_metrics.py",
            "--gt_dir", GT_DIR,
            "--pred_dir", OUTPUT_DIR
        ]
        
        # 捕获输出以提取指标
        result = subprocess.run(eval_cmd, capture_output=True, text=True)
        print(result.stdout) # 打印到控制台让用户看到进度

        # D. 解析并保存
        psnr_val, ssim_val, rase_val = parse_metrics(result.stdout)
        
        print(f"Result -> PSNR: {psnr_val}, SSIM: {ssim_val}, RASE: {rase_val}")
        
        csv_writer.writerow([ckpt_name, psnr_val, ssim_val, rase_val])
        csv_file.flush() # 立即写入磁盘，防止程序中断丢失数据

    csv_file.close()
    
    # 清理临时 JSON
    if os.path.exists(TEMP_JSON_NAME):
        os.remove(TEMP_JSON_NAME)
        
    print("\n" + "="*60)
    print(f"All done! Results saved to {CSV_FILENAME}")
    print("="*60)

if __name__ == "__main__":
    main()