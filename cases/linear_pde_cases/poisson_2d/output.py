import matplotlib.pyplot as plt
import numpy as np
import os
import time
import torch
from typing import Dict, Optional, List, Any
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import griddata
from data_generate import generate_global_field, exact_solution, generate_reference_solution
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

def plot_2d_solution(data: Dict, prediction: np.ndarray, config: Any, save_path: Optional[str] = None):
    """绘制二维泊松方程解的等高线图和3D曲面图
    
    Args:
        data: 包含输入数据的字典
        prediction: 预测结果
        config: 配置对象
        save_path: 保存路径
    """
    print("使用PDE专用绘图函数...")
    
    # 提取数据
    x_test = data['x']  # 测试点坐标
    exact_test = generate_reference_solution(x_test)
    u_pred = prediction  # 预测值
    
    # 获取网格点数
    nx, ny = config.points_domain_test
    x_min, x_max = np.min(x_test[:, 0]), np.max(x_test[:, 0])
    y_min, y_max = np.min(x_test[:, 1]), np.max(x_test[:, 1])
    
    # 创建网格
    xi = np.linspace(x_min, x_max, nx)
    yi = np.linspace(y_min, y_max, ny)
    Xi, Yi = np.meshgrid(xi, yi)
    
    # 重塑预测值和精确解为网格形状
    Ui_pred = u_pred.reshape(ny, nx)
    Ui_exact = exact_test.reshape(ny, nx)
    
    # 计算误差
    error = np.abs(Ui_pred - Ui_exact)
    
    # 1. 创建预测解等高线图
    fig1, ax = plt.subplots(figsize=(8, 6))
    cs = ax.contourf(Xi, Yi, Ui_pred, 50, cmap='viridis')
    ax.set_title('Predicted Solution')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    fig1.colorbar(cs, ax=ax)
    plt.tight_layout()
    
    # 2. 创建精确解等高线图
    fig2, ax = plt.subplots(figsize=(8, 6))
    cs = ax.contourf(Xi, Yi, Ui_exact, 50, cmap='viridis')
    ax.set_title('Exact Solution')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    fig2.colorbar(cs, ax=ax)
    plt.tight_layout()
    
    # 3. 创建误差等高线图
    fig3, ax = plt.subplots(figsize=(8, 6))
    cs = ax.contourf(Xi, Yi, error, 50, cmap='hot')
    ax.set_title('Absolute Error')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    fig3.colorbar(cs, ax=ax)
    plt.tight_layout()
    
    # 4. 创建预测解3D曲面图
    fig4 = plt.figure(figsize=(8, 6))
    ax = fig4.add_subplot(111, projection='3d')
    surf = ax.plot_surface(Xi, Yi, Ui_pred, cmap=cm.viridis, linewidth=0, antialiased=False)
    ax.set_title('Predicted Solution')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('u(x,y)')
    fig4.colorbar(surf, ax=ax, shrink=0.5, aspect=5)
    plt.tight_layout()
    
    # 保存图像
    if save_path:
        # 保存预测解等高线图
        contour_pred_path = os.path.join(os.path.dirname(save_path), 'contour_pred.png')
        fig1.savefig(contour_pred_path)
        print(f"预测解等高线图已保存至 {contour_pred_path}")
        
        # 保存精确解等高线图
        contour_exact_path = os.path.join(os.path.dirname(save_path), 'contour_exact.png')
        fig2.savefig(contour_exact_path)
        print(f"精确解等高线图已保存至 {contour_exact_path}")
        
        # 保存误差等高线图
        contour_error_path = os.path.join(os.path.dirname(save_path), 'contour_error.png')
        fig3.savefig(contour_error_path)
        print(f"误差等高线图已保存至 {contour_error_path}")
        
        # 保存3D曲面图
        surface_path = os.path.join(os.path.dirname(save_path), '3d_plot.png')
        fig4.savefig(surface_path)
        print(f"3D曲面图已保存至 {surface_path}")
        
        # 保存原来的测试结果图
        fig1.savefig(save_path)
        
        # 保存误差数据
        error_data_path = os.path.splitext(save_path)[0] + '_error.dat'
        error_data = np.column_stack((x_test, error.flatten()))
        np.savetxt(error_data_path, error_data, header='x y error', comments='')
        print(f"误差数据已保存至 {error_data_path}")
    else:
        plt.show()
    
    plt.close(fig1)
    plt.close(fig2)
    plt.close(fig3)
    plt.close(fig4)

def generate_output(output_data: Dict[str, Any]):
    """Generate output results and visualizations for Poisson equation solution
    
    Args:
        output_data: Dictionary containing all necessary data including:
            - test_data: Test dataset
            - test_predictions: Model predictions on test data
            - test_segments: Test data segments
            - coeffs: Model coefficients
            - model: Trained model
            - config: Configuration parameters
            - result_dir: Directory to save results
            - final_loss: Final training loss (optional)
            - solution_time: Total solution time (optional)
    """
    # Unpack data
    test_data = output_data['test_data']
    test_predictions = output_data['test_predictions']
    test_segments = output_data['test_segments']
    coeffs = output_data['coeffs']
    model = output_data['model']
    config = output_data['config']
    result_dir = output_data['result_dir']
    
    # Get optional metrics
    final_loss = output_data.get('final_loss', None)
    solution_time = output_data.get('solution_time', None)
    
    # Create result directory
    os.makedirs(result_dir, exist_ok=True)
    
    # 1. Generate visualizations
    # Plot test results
    test_vis_path = os.path.join(result_dir, "test_results.png")
    plot_2d_solution(test_data, test_predictions, config, test_vis_path)
    
    # 2. Save model and coefficients
    model_path = os.path.join(result_dir, "model.pt")
    coeffs_path = os.path.join(result_dir, "coefficients.npy")
    torch.save(model.state_dict(), model_path)
    np.save(coeffs_path, coeffs)
    
    # 3. Save raw data
    try:
        np.save(os.path.join(result_dir, "test_data.npy"), test_data)
    except Exception as e:
        print(f"Warning: Error saving data: {e}")
    
    # 4. Print summary
    print("\n=== Poisson Equation Solution Summary ===")
    if final_loss is not None:
        print(f"Final Training Loss: {final_loss:.6e}")
    if solution_time is not None:
        print(f"Total Solution Time: {solution_time:.4f} seconds")
    
    print(f"Results saved to: {result_dir}") 