import os
import json
import subprocess
import re
from pathlib import Path

def modify_config(config_path, n_segments):
    """修改配置文件中的n_segments值"""
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    config['n_segments'] = [n_segments]
    
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)

def extract_metrics(metrics_path):
    """从error_metrics.txt中提取关键指标"""
    with open(metrics_path, 'r') as f:
        content = f.read()
    
    # 提取训练损失
    train_loss_match = re.search(r'Final Training Loss: ([\d.e+-]+)', content)
    train_loss = float(train_loss_match.group(1)) if train_loss_match else None
    
    # 提取MSE
    mse_match = re.search(r'MSE: ([\d.e+-]+)', content)
    mse = float(mse_match.group(1)) if mse_match else None
    
    # 提取MAE
    mae_match = re.search(r'MAE: ([\d.e+-]+)', content)
    mae = float(mae_match.group(1)) if mae_match else None
    
    # 提取求解时间
    time_match = re.search(r'Total Solution Time: ([\d.]+) seconds', content)
    time = float(time_match.group(1)) if time_match else None
    
    return {
        'train_loss': train_loss,
        'mse': mse,
        'mae': mae,
        'time': time
    }

def update_error_md(error_md_path, n_segments, metrics):
    """更新error.md文件，添加新的结果行"""
    with open(error_md_path, 'r') as f:
        content = f.read()
    
    # 添加新的结果行
    new_line = f" section = {n_segments} train loss: {metrics['train_loss']:.1e}   MSE {metrics['mse']:.1e} MAE {metrics['mae']:.1e} time = {metrics['time']:.0f}s\n"
    
    # 在文件末尾添加新行
    with open(error_md_path, 'a') as f:
        f.write(new_line)

def main():
    # 设置路径
    base_dir = Path("/home/bfly/workspace/computeforcfd/混合网络/算例整合2")
    case_dir = base_dir / "cases/func_fitting_cases/multi_stage_case1"
    config_path = case_dir / "config.json"
    metrics_path = case_dir / "results/error_metrics.txt"
    error_md_path = case_dir / "results/error.md"
    
    # 要测试的n_segments值
    n_segments_list = [1, 2, 4, 8,16,32]
    
    for n_segments in n_segments_list:
        print(f"\n运行实验: n_segments = {n_segments}")
        
        # 1. 修改配置文件
        modify_config(config_path, n_segments)
        
        # 2. 运行main_solver.py
        subprocess.run(['python', str(base_dir / 'src/main_solver.py')], check=True)
        
        # 3. 提取结果
        metrics = extract_metrics(metrics_path)
        
        # 4. 更新error.md
        update_error_md(error_md_path, n_segments, metrics)
        
        print(f"完成 n_segments = {n_segments} 的实验")

if __name__ == "__main__":
    main()