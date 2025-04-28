import matplotlib.pyplot as plt
import numpy as np
import os
import time
import torch
from typing import Dict, Optional, List, Any
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import griddata
from data_generate import generate_global_field, exact_solution

def calculate_errors(true_values, predictions):
    """计算各种误差指标"""
    mse = np.mean((predictions - true_values)**2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(predictions - true_values))
    max_error = np.max(np.abs(predictions - true_values))
    
    relative_l2 = np.sqrt(np.sum((predictions - true_values)**2)) / np.sqrt(np.sum(true_values**2))
    
    return {
        'MSE': mse,
        'RMSE': rmse,
        'MAE': mae,
        'MAX_ERROR': max_error,
        'RELATIVE_L2': relative_l2
    }

def plot_2d_solution(data: Dict, prediction: np.ndarray, save_path: Optional[str] = None):
    """绘制二维泊松方程解的等高线图和3D曲面图
    
    Args:
        data: 包含输入数据的字典
        prediction: 预测结果
        save_path: 保存路径
    """
    print("使用PDE专用绘图函数...")
    
    # 提取数据
    x_test = data['x']  # 测试点坐标
    u_true = data['u']  # 真实值
    u_pred = prediction  # 预测值
    
    # 计算误差
    abs_error = np.abs(u_true - u_pred)
    
    # 创建网格用于绘图
    n_grid = 100
    x_min, x_max = np.min(x_test[:, 0]), np.max(x_test[:, 0])
    y_min, y_max = np.min(x_test[:, 1]), np.max(x_test[:, 1])
    
    xi = np.linspace(x_min, x_max, n_grid)
    yi = np.linspace(y_min, y_max, n_grid)
    Xi, Yi = np.meshgrid(xi, yi)
    
    # 使用散点插值到网格上
    Ui_true = griddata((x_test[:, 0], x_test[:, 1]), u_true.flatten(), (Xi, Yi), method='cubic')
    Ui_pred = griddata((x_test[:, 0], x_test[:, 1]), u_pred.flatten(), (Xi, Yi), method='cubic')
    
    # 计算误差场
    Ei = np.abs(Ui_true - Ui_pred)
    
    # 1. 创建等高线图
    fig1, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 绘制真实解等高线
    cs1 = axes[0, 0].contourf(Xi, Yi, Ui_true, 50, cmap='viridis')
    axes[0, 0].set_title('True Solution')
    axes[0, 0].set_xlabel('x')
    axes[0, 0].set_ylabel('y')
    fig1.colorbar(cs1, ax=axes[0, 0])
    
    # 绘制预测解等高线
    cs2 = axes[0, 1].contourf(Xi, Yi, Ui_pred, 50, cmap='viridis')
    axes[0, 1].set_title('Predicted Solution')
    axes[0, 1].set_xlabel('x')
    axes[0, 1].set_ylabel('y')
    fig1.colorbar(cs2, ax=axes[0, 1])
    
    # 绘制误差等高线
    cs3 = axes[1, 0].contourf(Xi, Yi, Ei, 50, cmap='hot')
    axes[1, 0].set_title('Absolute Error')
    axes[1, 0].set_xlabel('x')
    axes[1, 0].set_ylabel('y')
    fig1.colorbar(cs3, ax=axes[1, 0])
    
    # 绘制误差剖面
    mid_idx = n_grid // 2
    axes[1, 1].semilogy(xi, Ei[mid_idx, :], 'b-', label='Error at y=0.5')
    axes[1, 1].semilogy(yi, Ei[:, mid_idx], 'r--', label='Error at x=0.5')
    axes[1, 1].set_title('Error Profiles (log scale)')
    axes[1, 1].set_xlabel('Coordinate')
    axes[1, 1].set_ylabel('Absolute Error')
    axes[1, 1].legend()
    axes[1, 1].grid(True)
    
    plt.tight_layout()
    
    # 2. 创建3D曲面图
    fig2 = plt.figure(figsize=(18, 6))
    
    # 真实解的3D曲面
    ax1 = fig2.add_subplot(131, projection='3d')
    surf1 = ax1.plot_surface(Xi, Yi, Ui_true, cmap=cm.viridis, linewidth=0, antialiased=False)
    ax1.set_title('True Solution')
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.set_zlabel('u(x,y)')
    fig2.colorbar(surf1, ax=ax1, shrink=0.5, aspect=5)
    
    # 预测解的3D曲面
    ax2 = fig2.add_subplot(132, projection='3d')
    surf2 = ax2.plot_surface(Xi, Yi, Ui_pred, cmap=cm.viridis, linewidth=0, antialiased=False)
    ax2.set_title('Predicted Solution')
    ax2.set_xlabel('x')
    ax2.set_ylabel('y')
    ax2.set_zlabel('u(x,y)')
    fig2.colorbar(surf2, ax=ax2, shrink=0.5, aspect=5)
    
    # 误差的3D曲面
    ax3 = fig2.add_subplot(133, projection='3d')
    surf3 = ax3.plot_surface(Xi, Yi, Ei, cmap=cm.hot, linewidth=0, antialiased=False)
    ax3.set_title('Absolute Error')
    ax3.set_xlabel('x')
    ax3.set_ylabel('y')
    ax3.set_zlabel('|Error|')
    fig2.colorbar(surf3, ax=ax3, shrink=0.5, aspect=5)
    
    plt.tight_layout()
    
    # 保存图像
    if save_path:
        # 保存等高线图
        contour_path = os.path.join(os.path.dirname(save_path), 'contour_plots.png')
        fig1.savefig(contour_path)
        print(f"等高线图已保存至 {contour_path}")
        
        # 保存3D曲面图
        surface_path = os.path.join(os.path.dirname(save_path), '3d_plots.png')
        fig2.savefig(surface_path)
        print(f"3D曲面图已保存至 {surface_path}")
        
        # 保存原来的测试结果图
        fig1.savefig(save_path)
        
        # 保存误差数据
        error_data_path = os.path.splitext(save_path)[0] + '_error.dat'
        error_data = np.column_stack((x_test[:, 0], x_test[:, 1], abs_error.flatten()))
        np.savetxt(error_data_path, error_data, header='x y error', comments='')
        print(f"误差数据已保存至 {error_data_path}")
    else:
        plt.show()
    
    plt.close(fig1)
    plt.close(fig2)

def generate_output(output_data: Dict[str, Any]):
    """生成输出结果和可视化
    
    Args:
        output_data: 包含所有必要数据的字典
    """
    # 解包数据
    train_data = output_data['train_data']
    test_data = output_data['test_data']
    train_predictions = output_data['train_predictions']
    test_predictions = output_data['test_predictions']
    train_segments = output_data['train_segments']
    test_segments = output_data['test_segments']
    coeffs = output_data['coeffs']
    model = output_data['model']
    config = output_data['config']
    result_dir = output_data['result_dir']
    
    # 获取最终损失值和求解时间
    final_loss = output_data.get('final_loss', None)
    solution_time = output_data.get('solution_time', None)
    
    # 确保结果目录存在
    os.makedirs(result_dir, exist_ok=True)
    
    # 1. 计算误差指标
    train_errors = calculate_errors(train_data['u'], train_predictions)
    test_errors = calculate_errors(test_data['u'], test_predictions)
    
    # 2. 保存误差指标到文本文件
    with open(os.path.join(result_dir, "error_metrics.txt"), "w") as f:
        f.write("=== 训练集误差 ===\n")
        for name, value in train_errors.items():
            f.write(f"{name}: {value:.8e}\n")
        
        f.write("\n=== 测试集误差 ===\n")
        for name, value in test_errors.items():
            f.write(f"{name}: {value:.8e}\n")
        
        # 添加最终损失和求解时间信息
        f.write("\n=== 求解信息 ===\n")
        if final_loss is not None:
            f.write(f"最终训练损失: {final_loss:.8e}\n")
        if solution_time is not None:
            f.write(f"总求解时间: {solution_time:.4f} 秒\n")
    
    # 3. 生成可视化结果
    # 只绘制测试集结果
    test_vis_path = os.path.join(result_dir, "test_results.png")
    plot_2d_solution(test_data, test_predictions, test_vis_path)
    
    # 4. 保存系数和模型
    np.save(os.path.join(result_dir, "coefficients.npy"), coeffs)
    torch.save(model.state_dict(), os.path.join(result_dir, "model.pt"))
    
    # 5. 保存源数据
    try:
        np.save(os.path.join(result_dir, "train_data.npy"), train_data)
        np.save(os.path.join(result_dir, "test_data.npy"), test_data)
    except Exception as e:
        print(f"警告: 保存数据时出错: {e}")
    
    # 6. 输出结果摘要
    print("\n=== 泊松方程求解结果摘要 ===")
    print(f"训练集 MSE: {train_errors['MSE']:.6e}")
    print(f"测试集 MSE: {test_errors['MSE']:.6e}")
    print(f"测试集相对L2误差: {test_errors['RELATIVE_L2']:.6e}")
    
    # 打印最终损失和求解时间信息
    if final_loss is not None:
        print(f"最终训练损失: {final_loss:.6e}")
    if solution_time is not None:
        print(f"总求解时间: {solution_time:.4f} 秒")
        
    print(f"结果已保存至: {result_dir}") 