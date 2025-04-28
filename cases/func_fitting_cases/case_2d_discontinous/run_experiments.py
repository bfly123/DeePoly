import os
import json
import subprocess
import re
from pathlib import Path

def modify_config(config_path, n_segments):
    """修改配置文件中的n_segments值，保持二维分段"""
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # 二维分段，同时修改x和y维度
    config['n_segments'] = [n_segments, n_segments]
    
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)

def extract_metrics(metrics_path):
    """从error_metrics.txt中提取训练集和测试集的关键指标"""
    with open(metrics_path, 'r') as f:
        content = f.read()
    
    # 提取训练集指标
    train_mse_match = re.search(r'=== Training Set Errors ===\s+MSE: ([\d.e+-]+)', content)
    train_mse = float(train_mse_match.group(1)) if train_mse_match else None
    
    train_mae_match = re.search(r'=== Training Set Errors ===.*?MAE: ([\d.e+-]+)', content, re.DOTALL)
    train_mae = float(train_mae_match.group(1)) if train_mae_match else None
    
    # 提取测试集指标
    test_mse_match = re.search(r'=== Test Set Errors ===\s+MSE: ([\d.e+-]+)', content)
    test_mse = float(test_mse_match.group(1)) if test_mse_match else None
    
    test_mae_match = re.search(r'=== Test Set Errors ===.*?MAE: ([\d.e+-]+)', content, re.DOTALL)
    test_mae = float(test_mae_match.group(1)) if test_mae_match else None
    
    # 提取训练损失
    train_loss_match = re.search(r'Final Training Loss: ([\d.e+-]+)', content)
    train_loss = float(train_loss_match.group(1)) if train_loss_match else None
    
    # 提取求解时间
    time_match = re.search(r'Total Solution Time: ([\d.]+) seconds', content)
    time = float(time_match.group(1)) if time_match else None
    
    return {
        'train_loss': train_loss,
        'train_mse': train_mse,
        'train_mae': train_mae,
        'test_mse': test_mse,
        'test_mae': test_mae,
        'time': time
    }

def update_error_md(error_md_path, n_segments, metrics):
    """更新error.md文件，添加新的结果行包含训练和测试的MSE和MAE"""
    # 检查文件是否存在，如果不存在则创建并添加表头
    if not os.path.exists(error_md_path):
        with open(error_md_path, 'w') as f:
            f.write("# 实验结果\n\n")
            f.write("| 分段数(每维) | 总分段数 | 训练损失 | 训练MSE | 训练MAE | 测试MSE | 测试MAE | 求解时间(s) |\n")
            f.write("|------------|---------|----------|---------|---------|---------|---------|-------------|\n")
    
    # 计算总分段数
    total_segments = n_segments * n_segments
    
    # 格式化新的结果行
    new_line = f"| {n_segments}×{n_segments} | {total_segments} | {metrics['train_loss']:.3e} | {metrics['train_mse']:.3e} | {metrics['train_mae']:.3e} | {metrics['test_mse']:.3e} | {metrics['test_mae']:.3e} | {metrics['time']:.1f} |\n"
    
    # 在文件末尾添加新行
    with open(error_md_path, 'a') as f:
        f.write(new_line)

def main():
    # 设置路径
    base_dir = Path("/home/bfly/workspace/computeforcfd/混合网络/算例整合2")
    case_dir = base_dir / "cases/func_fitting_cases/case_2d_discontinous"
    config_path = case_dir / "config.json"
    metrics_path = case_dir / "results/error_metrics.txt"
    error_md_path = case_dir / "results/error.md"
    
    # 要测试的n_segments值（每个维度）
    n_segments_list = [1, 2, 4, 8, 10]
    
    for n_segments in n_segments_list:
        print(f"\n运行实验: n_segments = {n_segments}×{n_segments} (总分段数: {n_segments*n_segments})")
        
        # 1. 修改配置文件
        modify_config(config_path, n_segments)
        
        # 2. 运行main_solver.py，添加--case_path参数
        cmd = [
            'python', 
            str(base_dir / 'src/main_solver.py'), 
            '--case_path', 
            str(case_dir)
        ]
        print(f"执行命令: {' '.join(cmd)}")
        subprocess.run(cmd, check=True)
        
        # 3. 提取结果
        metrics = extract_metrics(metrics_path)
        
        # 4. 更新error.md
        update_error_md(error_md_path, n_segments, metrics)
        
        print(f"完成 n_segments = {n_segments}×{n_segments} 的实验")

if __name__ == "__main__":
    main()