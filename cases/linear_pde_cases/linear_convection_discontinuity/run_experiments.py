import os
import json
import subprocess
import re
from pathlib import Path

def modify_config(config_path, n_segments):
    """修改配置文件中的n_segments值"""
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # 二维分段设置
    config['n_segments'] = [n_segments, n_segments]
    
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)

def extract_metrics(metrics_path):
    """从error_metrics.txt中提取训练集和测试集的关键指标"""
    with open(metrics_path, 'r') as f:
        content = f.read()
    
    # 提取训练集指标
    train_mse_match = re.search(r'=== 训练集误差 ===\s+MSE: ([\d.e+-]+)', content)
    train_mse = float(train_mse_match.group(1)) if train_mse_match else None
    
    train_rel_l2_match = re.search(r'=== 训练集误差 ===.*?RELATIVE_L2: ([\d.e+-]+)', content, re.DOTALL)
    train_rel_l2 = float(train_rel_l2_match.group(1)) if train_rel_l2_match else None
    
    # 提取测试集指标
    test_mse_match = re.search(r'=== 测试集误差 ===\s+MSE: ([\d.e+-]+)', content)
    test_mse = float(test_mse_match.group(1)) if test_mse_match else None
    
    test_rel_l2_match = re.search(r'=== 测试集误差 ===.*?RELATIVE_L2: ([\d.e+-]+)', content, re.DOTALL)
    test_rel_l2 = float(test_rel_l2_match.group(1)) if test_rel_l2_match else None
    
    # 提取训练损失
    train_loss_match = re.search(r'最终训练损失: ([\d.e+-]+)', content)
    train_loss = float(train_loss_match.group(1)) if train_loss_match else None
    
    # 提取求解时间
    time_match = re.search(r'总求解时间: ([\d.]+) 秒', content)
    time = float(time_match.group(1)) if time_match else None
    
    return {
        'train_loss': train_loss,
        'train_mse': train_mse,
        'train_rel_l2': train_rel_l2,
        'test_mse': test_mse,
        'test_rel_l2': test_rel_l2,
        'time': time
    }

def update_error_md(error_md_path, n_segments, metrics):
    """使用简单格式更新error.md文件，包含训练集和测试集的MSE和相对L2误差"""
    # 添加新的结果行
    new_line = f" segments = {n_segments}x{n_segments} train loss: {metrics['train_loss']:.1e}   "
    new_line += f"train MSE: {metrics['train_mse']:.1e} "
    new_line += f"train L2: {metrics['train_rel_l2']:.1e} "
    new_line += f"test MSE: {metrics['test_mse']:.1e} "
    new_line += f"test L2: {metrics['test_rel_l2']:.1e} "
    new_line += f"time = {metrics['time']:.0f}s\n"
    
    # 在文件末尾添加新行
    with open(error_md_path, 'a') as f:
        f.write(new_line)

def update_error_table(error_table_path, n_segments, metrics):
    """创建和更新表格格式的error_table.md文件"""
    # 检查文件是否存在，如果不存在则创建并添加表头
    if not os.path.exists(error_table_path):
        with open(error_table_path, 'w') as f:
            f.write("# 二维泊松方程分段实验结果表格\n\n")
            f.write("| 分段数 | 训练损失 | 训练MSE | 训练相对L2 | 测试MSE | 测试相对L2 | 求解时间(s) |\n")
            f.write("|--------|----------|---------|------------|---------|------------|-------------|\n")
    
    # 格式化新的结果行
    new_line = f"| {n_segments}x{n_segments} | {metrics['train_loss']:.3e} | {metrics['train_mse']:.3e} | {metrics['train_rel_l2']:.3e} | {metrics['test_mse']:.3e} | {metrics['test_rel_l2']:.3e} | {metrics['time']:.1f} |\n"
    
    # 在文件末尾添加新行
    with open(error_table_path, 'a') as f:
        f.write(new_line)

def main():
    # 设置路径
    base_dir = Path("/home/bfly/workspace/computeforcfd/混合网络/算例整合2")
    case_dir = base_dir / "cases/linear_pde_cases/poisson_2d"
    config_path = case_dir / "config.json"
    metrics_path = case_dir / "results/error_metrics.txt"
    error_md_path = case_dir / "results/error.md"
    error_table_path = case_dir / "results/error_table.md"
    
    # 要测试的n_segments值
    # 从小到大测试不同分段数
    n_segments_list = [2, 4, 8, 16]
    
    # 确保结果目录存在
    os.makedirs(case_dir / "results", exist_ok=True)
    
    # 创建README文件，说明实验内容
    with open(case_dir / "results/README.md", "w") as f:
        f.write("# 二维泊松方程分段实验\n\n")
        f.write("本实验测试了不同分段数对求解二维泊松方程的影响。\n\n")
        f.write("泊松方程：$-\\Delta u = f$ 在 $\\Omega = [0,1]\\times[0,1]$\n\n")
        f.write("边界条件：$u = 0$ 在 $\\partial\\Omega$\n\n")
        f.write("源项：$f = 2\\pi^2\\sin(\\pi x)\\sin(\\pi y)$\n\n")
        f.write("精确解：$u(x,y) = \\sin(\\pi x)\\sin(\\pi y)$\n\n")
    
    for n_segments in n_segments_list:
        print(f"\n运行实验: n_segments = {n_segments}x{n_segments}")
        
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
        
        # 4. 同时更新两种格式的文件
        update_error_md(error_md_path, n_segments, metrics)
        update_error_table(error_table_path, n_segments, metrics)
        
        print(f"完成 n_segments = {n_segments}x{n_segments} 的实验")

if __name__ == "__main__":
    main() 