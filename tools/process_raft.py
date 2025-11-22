import json
import os
import numpy as np
from PIL import Image
import torch
import torch.multiprocessing as mp
from datasets.raft import compute_optical_flow
from tqdm import tqdm
import time

def process_single_triplet(args):
    """
    处理单个三元组的函数，用于多进程并行处理
    """
    triplet, output_dir, gpu_id, progress_file = args
    
    try:
        # 设置当前进程使用的GPU
        torch.cuda.set_device(gpu_id)
        
        prev_path, curr_path, next_path = triplet
        
        # 检查输出文件是否已存在（断点续推）
        base_name = os.path.basename(curr_path).replace('.png', '')
        forward_output = os.path.join(output_dir, f"{base_name}_forward.npy")
        backward_output = os.path.join(output_dir, f"{base_name}_backward.npy")
        
        if os.path.exists(forward_output) and os.path.exists(backward_output):
            return True, f"Skipped (already exists): {base_name}"
        
        # 加载图像
        prev_img = np.array(Image.open(prev_path).convert('L'))
        curr_img = np.array(Image.open(curr_path).convert('L'))
        next_img = np.array(Image.open(next_path).convert('L'))
        
        # 计算前向光流 (curr -> next)
        forward_flow = compute_optical_flow(curr_img, next_img)
        
        # 计算后向光流 (curr -> prev)
        backward_flow = compute_optical_flow(curr_img, prev_img)
        
        # 保存光流场
        np.save(forward_output, forward_flow)
        np.save(backward_output, backward_flow)
        
        # 更新进度文件
        with open(progress_file, 'a') as f:
            f.write(f"{curr_path}\n")
        
        return True, f"Success: {base_name}"
        
    except Exception as e:
        return False, f"Error processing {curr_path}: {str(e)}"

def process_raft_triplets_parallel(json_file_path, output_dir, num_gpus=4):
    """
    使用多GPU并行处理三元组图像，计算光流场
    
    Args:
        json_file_path: 包含图像三元组的JSON文件路径
        output_dir: 输出光流场的目录
        num_gpus: 使用的GPU数量
    """
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 进度文件路径（用于断点续推）
    progress_file = os.path.join(output_dir, "processed_files.txt")
    processed_set = set()
    
    # 加载已处理的文件（断点续推）
    if os.path.exists(progress_file):
        with open(progress_file, 'r') as f:
            processed_set = set(line.strip() for line in f)
        print(f"找到进度文件，已处理 {len(processed_set)} 个文件")
    
    # 加载JSON文件
    with open(json_file_path, 'r') as f:
        triplets = json.load(f)
    
    # 过滤掉已处理的三元组
    remaining_triplets = []
    for triplet in triplets:
        _, curr_path, _ = triplet
        if curr_path not in processed_set:
            remaining_triplets.append(triplet)
    
    print(f"总三元组数: {len(triplets)}, 剩余待处理: {len(remaining_triplets)}")
    
    if not remaining_triplets:
        print("所有三元组已处理完成！")
        return
    
    # 准备进程参数
    process_args = []
    for i, triplet in enumerate(remaining_triplets):
        gpu_id = i % num_gpus  # 轮询分配GPU
        process_args.append((triplet, output_dir, gpu_id, progress_file))
    
    # 使用多进程并行处理
    print(f"开始使用 {num_gpus} 个GPU并行处理...")
    start_time = time.time()
    
    # 使用进程池
    with mp.Pool(processes=num_gpus) as pool:
        results = list(tqdm(
            pool.imap(process_single_triplet, process_args),
            total=len(process_args),
            desc="Processing triplets"
        ))
    
    # 统计结果
    success_count = sum(1 for success, _ in results if success)
    error_count = len(results) - success_count
    
    end_time = time.time()
    print(f"\n处理完成！")
    print(f"成功: {success_count}, 失败: {error_count}")
    print(f"总耗时: {end_time - start_time:.2f} 秒")
    print(f"输出保存在: {output_dir}")
    
    # 打印错误信息（如果有）
    if error_count > 0:
        print("\n错误详情:")
        for success, message in results:
            if not success:
                print(f"  - {message}")

def process_raft_triplets_sequential(json_file_path, output_dir):
    """
    顺序处理版本（用于调试或小规模数据）
    """
    os.makedirs(output_dir, exist_ok=True)
    progress_file = os.path.join(output_dir, "processed_files.txt")
    processed_set = set()
    
    if os.path.exists(progress_file):
        with open(progress_file, 'r') as f:
            processed_set = set(line.strip() for line in f)
    
    with open(json_file_path, 'r') as f:
        triplets = json.load(f)
    
    remaining_triplets = [
        triplet for triplet in triplets 
        if triplet[1] not in processed_set
    ]
    
    print(f"开始处理 {len(remaining_triplets)} 个三元组...")
    
    for i, triplet in enumerate(tqdm(remaining_triplets)):
        try:
            prev_path, curr_path, next_path = triplet
            
            # 加载图像
            prev_img = np.array(Image.open(prev_path).convert('L'))
            curr_img = np.array(Image.open(curr_path).convert('L'))
            next_img = np.array(Image.open(next_path).convert('L'))
            
            # 计算光流
            forward_flow = compute_optical_flow(curr_img, next_img)
            backward_flow = compute_optical_flow(curr_img, prev_img)
            
            # 保存结果
            base_name = os.path.basename(curr_path).replace('.png', '')
            np.save(os.path.join(output_dir, f"{base_name}_forward.npy"), forward_flow)
            np.save(os.path.join(output_dir, f"{base_name}_backward.npy"), backward_flow)
            
            # 更新进度
            with open(progress_file, 'a') as f:
                f.write(f"{curr_path}\n")
                
        except Exception as e:
            print(f"处理三元组 {i} 时出错: {e}")
            continue
    
    print(f"处理完成！输出保存在: {output_dir}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='处理台风图像三元组的光流计算')
    parser.add_argument('--json_file', type=str, required=True, help='包含图像三元组的JSON文件路径')
    parser.add_argument('--output_dir', type=str, required=True, help='输出光流场的目录')
    parser.add_argument('--num_gpus', type=int, default=4, help='使用的GPU数量（默认4）')
    parser.add_argument('--sequential', action='store_true', help='使用顺序处理（用于调试）')
    
    args = parser.parse_args()
    
    # 设置多进程启动方法
    mp.set_start_method('spawn', force=True)
    
    if args.sequential:
        process_raft_triplets_sequential(args.json_file, args.output_dir)
    else:
        process_raft_triplets_parallel(args.json_file, args.output_dir, args.num_gpus)