import matplotlib.pyplot as plt
import numpy as np
import os
import time
import torch
from typing import Dict, Optional, List, Any
from data_generate import generate_global_field
from mpl_toolkits.mplot3d import Axes3D

def calculate_errors(true_values, predictions):
    """Calculate various error metrics"""
    mse = np.mean((predictions - true_values)**2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(predictions - true_values))
    max_error = np.max(np.abs(predictions - true_values))
    
    return {
        'MSE': mse,
        'RMSE': rmse,
        'MAE': mae,
        'MAX_ERROR': max_error
    }

def plot_solution(data: Dict, prediction: np.ndarray, save_path: Optional[str] = None):
    """为随机采样数据绘制三维散点图
    
    Args:
        data: Dictionary containing input data
        prediction: Prediction results
        save_path: Save path
    """
    print("使用3D散点图绘图函数...")
    x_test = data['x']  # 测试点坐标 (n_test, 2)
    y_true = data['u']  # 真实值 (n_test, 1)
    
    # 获取真实无噪声函数值
    y_exact = generate_global_field(x_test)

    # 计算绝对误差
    abs_error = np.abs(y_exact - prediction)

    # 创建图形，1x2子图布局
    fig = plt.figure(figsize=(15, 6))
    
    # 1. 预测结果的散点图
    ax1 = fig.add_subplot(121, projection='3d')
    scatter1 = ax1.scatter(x_test[:, 0], x_test[:, 1], prediction, 
                          c=prediction, cmap='plasma', s=30, alpha=0.6)
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.set_zlabel('预测值')
    ax1.set_title('预测结果散点图')
    plt.colorbar(scatter1, ax=ax1, label='函数值')
    ax1.grid(True)
    ax1.view_init(elev=20, azim=45)

    # 2. 误差散点图 (使用对数坐标)
    ax2 = fig.add_subplot(122, projection='3d')
    # 添加小的常数避免log(0)错误
    error_safe = np.maximum(abs_error, 1e-15)
    # 使用log10缩放误差值
    scatter2 = ax2.scatter(x_test[:, 0], x_test[:, 1], np.log10(error_safe), 
                          c=np.log10(error_safe), cmap='hot', s=30, alpha=0.6)
    ax2.set_xlabel('x')
    ax2.set_ylabel('y')
    ax2.set_zlabel('log10(误差)')
    ax2.set_title('误差散点图 (对数坐标)')
    plt.colorbar(scatter2, ax=ax2, label='log10(|误差|)')
    ax2.grid(True)
    ax2.view_init(elev=20, azim=45)

    # 调整布局
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"图像已保存至 {save_path}")
        
        # 保存误差数据到dat文件
        error_data_path = os.path.splitext(save_path)[0] + '_error.dat'
        error_data = np.column_stack((x_test, abs_error))
        np.savetxt(error_data_path, error_data, header='x y error', comments='')
        print(f"误差数据已保存至 {error_data_path}")
    else:
        plt.show()
    plt.close()

def plot_solution_mesh(data: Dict, prediction: np.ndarray, save_path: Optional[str] = None):
    """Plot 3D surface plots for mesh data
    
    Args:
        data: Dictionary containing input data
        prediction: Prediction results
        save_path: Save path
    """
    print("Using 3D mesh plotting function...")
    x_test = data['x']  # test points coordinates (n_test, 2)
    y_true = data['u']  # true values (n_test, 1)
    
    # Get exact values
    y_exact = generate_global_field(x_test)

    # Calculate absolute error
    abs_error = np.abs(y_exact - prediction)

    # Reconstruct original grid points
    x_points = np.unique(x_test[:, 0])
    y_points = np.unique(x_test[:, 1])
    nx, ny = len(x_points), len(y_points)
    
    # Create meshgrid
    X, Y = np.meshgrid(x_points, y_points)
    
    # Initialize grid data
    Z_pred = np.zeros((ny, nx))
    Z_error = np.zeros((ny, nx))
    
    # Fill grid data
    for i, x in enumerate(x_points):
        for j, y in enumerate(y_points):
            idx = np.where((x_test[:, 0] == x) & (x_test[:, 1] == y))[0][0]
            Z_pred[j, i] = prediction[idx]
            Z_error[j, i] = abs_error[idx]

    # Create figure with 1x2 subplots
    fig = plt.figure(figsize=(15, 6))
    
    # 1. Prediction surface plot
    ax1 = fig.add_subplot(121, projection='3d')
    surf1 = ax1.plot_surface(X, Y, Z_pred, cmap='plasma', alpha=0.8)
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.set_zlabel('Prediction')
    ax1.set_title('Prediction Results')
    plt.colorbar(surf1, ax=ax1, label='Value')
    ax1.grid(True)
    ax1.view_init(elev=20, azim=45)

    # 2. Error surface plot (with log scale for z-axis)
    ax2 = fig.add_subplot(122, projection='3d')
    # 添加小的常数避免log(0)错误
    Z_error_safe = np.maximum(Z_error, 1e-15)  
    # 使用log10缩放误差值
    surf2 = ax2.plot_surface(X, Y, np.log10(Z_error_safe), cmap='hot', alpha=0.8)
    ax2.set_xlabel('x')
    ax2.set_ylabel('y')
    ax2.set_zlabel('log10(Error)')
    ax2.set_title('Absolute Error (Log10 Scale)')
    plt.colorbar(surf2, ax=ax2, label='log10(|Error|)')
    ax2.grid(True)
    ax2.view_init(elev=20, azim=45)

    # Adjust layout
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to: {save_path}")
        
        # Save error data
        error_data_path = os.path.splitext(save_path)[0] + '_error.dat'
        error_data = np.column_stack((x_test, abs_error))
        np.savetxt(error_data_path, error_data, header='x y error', comments='')
        print(f"Error data saved to: {error_data_path}")
    else:
        plt.show()
    plt.close()

def generate_output(output_data: Dict[str, Any]):
    """Generate all output results and visualizations
    
    Args:
        output_data: Dictionary containing all required data
    """
    # Unpack data
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
    
    # Get final loss value and solution time if available
    final_loss = output_data.get('final_loss', None)
    solution_time = output_data.get('solution_time', None)
    
    # Ensure result directory exists
    os.makedirs(result_dir, exist_ok=True)
    
    # 1. Calculate error metrics
    # Check if u exists in the data, if not try to build it from u_segments
   
    train_errors = calculate_errors(train_data['u'], train_predictions)
    test_errors = calculate_errors(test_data['u'], test_predictions)
    
    # 2. Save error metrics to text file
    with open(os.path.join(result_dir, "error_metrics.txt"), "w") as f:
        f.write("=== Training Set Errors ===\n")
        for name, value in train_errors.items():
            f.write(f"{name}: {value:.8e}\n")
        
        f.write("\n=== Test Set Errors ===\n")
        for name, value in test_errors.items():
            f.write(f"{name}: {value:.8e}\n")
        
        # Add final loss and solution time information
        f.write("\n=== Solution Information ===\n")
        if final_loss is not None:
            f.write(f"Final Training Loss: {final_loss:.8e}\n")
        if solution_time is not None:
            f.write(f"Total Solution Time: {solution_time:.4f} seconds\n")
    
    # 3. Generate visualization results
    # Only plot test set results
    test_vis_path = os.path.join(result_dir, "test_results.png")
    if isinstance(config.points_domain_test, int):
        plot_solution(test_data, test_predictions, test_vis_path)
    else:
        plot_solution_mesh(test_data, test_predictions, test_vis_path)
    
    # 4. Save coefficients and model
    np.save(os.path.join(result_dir, "coefficients.npy"), coeffs)
    torch.save(model.state_dict(), os.path.join(result_dir, "model.pt"))
    
    # 5. Save source data
    try:
        np.save(os.path.join(result_dir, "train_data.npy"), train_data)
        np.save(os.path.join(result_dir, "test_data.npy"), test_data)
    except Exception as e:
        print(f"Warning: Error saving data: {e}")
    
    # 6. Output result summary
    print("\n=== Fitting Results Summary ===")
    print(f"Training Set MSE: {train_errors['MSE']:.6e}")
    print(f"Test Set MSE: {test_errors['MSE']:.6e}")
    print(f"Test Set RMSE: {test_errors['RMSE']:.6e}")
    
    # Print final loss and solution time information
    if final_loss is not None:
        print(f"Final Training Loss: {final_loss:.6e}")
    if solution_time is not None:
        print(f"Total Solution Time: {solution_time:.4f} seconds")
        
    print(f"Results saved to: {result_dir}")

def plot_results(X_test, u_pred, u_true, save_path=None):
    """
    绘制二维函数拟合结果
    Args:
        X_test: 测试点坐标 (n_test, 2)
        u_pred: 预测值 (n_test, 1)
        u_true: 真实值 (n_test, 1)
        save_path: 保存路径
    """
    # 转换为numpy数组
    X_test = X_test.detach().cpu().numpy()
    u_pred = u_pred.detach().cpu().numpy()
    u_true = u_true.detach().cpu().numpy()
    
    # 创建图形
    fig = plt.figure(figsize=(15, 5))
    
    # 绘制预测结果
    ax1 = fig.add_subplot(131)
    scatter1 = ax1.scatter(X_test[:, 0], X_test[:, 1], c=u_pred, cmap='viridis')
    ax1.set_title('预测解')
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    plt.colorbar(scatter1, ax=ax1)
    
    # 绘制真实解
    ax2 = fig.add_subplot(132)
    scatter2 = ax2.scatter(X_test[:, 0], X_test[:, 1], c=u_true, cmap='viridis')
    ax2.set_title('真实解')
    ax2.set_xlabel('x')
    ax2.set_ylabel('y')
    plt.colorbar(scatter2, ax=ax2)
    
    # 绘制误差
    ax3 = fig.add_subplot(133)
    error = np.abs(u_pred - u_true)
    scatter3 = ax3.scatter(X_test[:, 0], X_test[:, 1], c=error, cmap='viridis')
    ax3.set_title('绝对误差')
    ax3.set_xlabel('x')
    ax3.set_ylabel('y')
    plt.colorbar(scatter3, ax=ax3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
    plt.close()

def calculate_metrics(u_pred, u_true):
    """
    计算误差指标
    Args:
        u_pred: 预测值
        u_true: 真实值
    Returns:
        dict: 包含各种误差指标的字典
    """
    # 转换为numpy数组
    u_pred = u_pred.detach().cpu().numpy()
    u_true = u_true.detach().cpu().numpy()
    
    # 计算各种误差
    mse = np.mean((u_pred - u_true)**2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(u_pred - u_true))
    max_error = np.max(np.abs(u_pred - u_true))
    
    return {
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'max_error': max_error
    }

def save_metrics(metrics, save_path):
    """
    保存误差指标到文件
    Args:
        metrics: 误差指标字典
        save_path: 保存路径
    """
    with open(save_path, 'w') as f:
        f.write("=== Training Set Errors ===\n")
        f.write(f"MSE: {metrics['mse']:.8e}\n")
        f.write(f"RMSE: {metrics['rmse']:.8e}\n")
        f.write(f"MAE: {metrics['mae']:.8e}\n")
        f.write(f"MAX_ERROR: {metrics['max_error']:.8e}\n")