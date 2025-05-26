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

def generate_output(config, data_train, data_test, fitter, model, coeffs, result_dir):
    x_train = data_train["x"]
    x_test = data_test["x"]
    nx, ny = config.points_domain_test
    x_coords = np.unique(x_test[:,0])
    y_coords = np.unique(x_test[:,1])

    _, net_train = model(torch.tensor(x_train, dtype=torch.float64))
    _, net_test = model(torch.tensor(x_test, dtype=torch.float64))
    net_train = net_train.detach().numpy()
    net_test = net_test.detach().numpy()

    exact_train = generate_reference_solution(x_train)
    exact_test = generate_reference_solution(x_test)

    deePoly_train, _ = fitter.construct(data_train, model, coeffs)
    deePoly_test, _ = fitter.construct(data_test, model, coeffs)

    # Calculate errors
    def calculate_errors(pred, exact):
        mse = np.mean((pred - exact) ** 2)
        mae = np.mean(np.abs(pred - exact))
        max_error = np.max(np.abs(pred - exact))
        return mse, mae, max_error

    # Calculate errors for both methods
    net_train_errors = calculate_errors(net_train, exact_train)
    net_test_errors = calculate_errors(net_test, exact_test)
    deePoly_train_errors = calculate_errors(deePoly_train, exact_train)
    deePoly_test_errors = calculate_errors(deePoly_test, exact_test)

    # Print error statistics
    print("\nError Statistics:")
    print("Training Data:")
    print(f"Net Method - MSE: {net_train_errors[0]:.2e}, MAE: {net_train_errors[1]:.2e}, Max Error: {net_train_errors[2]:.2e}")
    print(f"DeePoly Method - MSE: {deePoly_train_errors[0]:.2e}, MAE: {deePoly_train_errors[1]:.2e}, Max Error: {deePoly_train_errors[2]:.2e}")
    print("\nTest Data:")
    print(f"Net Method - MSE: {net_test_errors[0]:.2e}, MAE: {net_test_errors[1]:.2e}, Max Error: {net_test_errors[2]:.2e}")
    print(f"DeePoly Method - MSE: {deePoly_test_errors[0]:.2e}, MAE: {deePoly_test_errors[1]:.2e}, Max Error: {deePoly_test_errors[2]:.2e}")

    # Create visualization directory if it doesn't exist
    vis_dir = os.path.join(result_dir, "visualizations")
    os.makedirs(vis_dir, exist_ok=True)

    # 添加训练数据的对比图
    fig = plt.figure(figsize=(15, 5))
    
    # Plot Net Method (Train)
    ax1 = fig.add_subplot(131)
    scatter1 = ax1.scatter(x_train[:, 0], x_train[:, 1], 
                          c=net_train, 
                          cmap=cm.coolwarm,
                          s=50)
    ax1.set_title('Net Method (Train)')
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    fig.colorbar(scatter1, ax=ax1, shrink=0.5)

    # Plot DeePoly Method (Train)
    ax2 = fig.add_subplot(132)
    scatter2 = ax2.scatter(x_train[:, 0], x_train[:, 1], 
                          c=deePoly_train, 
                          cmap=cm.coolwarm,
                          s=50)
    ax2.set_title('DeePoly Method (Train)')
    ax2.set_xlabel('x')
    ax2.set_ylabel('y')
    fig.colorbar(scatter2, ax=ax2, shrink=0.5)

    # Plot Exact Solution (Train)
    ax3 = fig.add_subplot(133)
    scatter3 = ax3.scatter(x_train[:, 0], x_train[:, 1], 
                          c=exact_train, 
                          cmap=cm.coolwarm,
                          s=50)
    ax3.set_title('Exact Solution (Train)')
    ax3.set_xlabel('x')
    ax3.set_ylabel('y')
    fig.colorbar(scatter3, ax=ax3, shrink=0.5)

    plt.tight_layout()
    plt.savefig(os.path.join(vis_dir, 'train_solution_comparison.png'))
    plt.close()

    # Reshape data for plotting
    X, Y = np.meshgrid(x_coords, y_coords)
    net_test_reshaped = net_test.reshape(nx, ny)
    deePoly_test_reshaped = deePoly_test.reshape(nx, ny)
    exact_test_reshaped = exact_test.reshape(nx, ny)

    # Create comparison plots
    fig = plt.figure(figsize=(15, 5))
    
    # Plot Net Method
    ax1 = fig.add_subplot(131)
    contour1 = ax1.contourf(X, Y, net_test_reshaped, levels=20, cmap=cm.coolwarm)
    ax1.set_title('Net Method')
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    fig.colorbar(contour1, ax=ax1, shrink=0.5)

    # Plot DeePoly Method
    ax2 = fig.add_subplot(132)
    contour2 = ax2.contourf(X, Y, deePoly_test_reshaped, levels=20, cmap=cm.coolwarm)
    ax2.set_title('DeePoly Method')
    ax2.set_xlabel('x')
    ax2.set_ylabel('y')
    fig.colorbar(contour2, ax=ax2, shrink=0.5)

    # Plot Exact Solution
    ax3 = fig.add_subplot(133)
    contour3 = ax3.contourf(X, Y, exact_test_reshaped, levels=20, cmap=cm.coolwarm)
    ax3.set_title('Exact Solution')
    ax3.set_xlabel('x')
    ax3.set_ylabel('y')
    fig.colorbar(contour3, ax=ax3, shrink=0.5)

    plt.tight_layout()
    plt.savefig(os.path.join(vis_dir, 'solution_comparison.png'))
    plt.close()

    # Plot error distributions
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Net Method Error
    net_error = np.abs(net_test - exact_test)
    im1 = ax1.contourf(X, Y, net_error.reshape(nx, ny), levels=20, cmap='hot')
    ax1.set_title('Net Method Error')
    plt.colorbar(im1, ax=ax1)

    # DeePoly Method Error
    deePoly_error = np.abs(deePoly_test - exact_test)
    im2 = ax2.contourf(X, Y, deePoly_error.reshape(nx, ny), levels=20, cmap='hot')
    ax2.set_title('DeePoly Method Error')
    plt.colorbar(im2, ax=ax2)

    plt.tight_layout()
    plt.savefig(os.path.join(vis_dir, 'error_distribution.png'))
    plt.close()

    # Save error statistics to file
    with open(os.path.join(result_dir, 'error_statistics.txt'), 'w') as f:
        f.write("Error Statistics:\n")
        f.write("Training Data:\n")
        f.write(f"Net Method - MSE: {net_train_errors[0]:.2e}, MAE: {net_train_errors[1]:.2e}, Max Error: {net_train_errors[2]:.2e}\n")
        f.write(f"DeePoly Method - MSE: {deePoly_train_errors[0]:.2e}, MAE: {deePoly_train_errors[1]:.2e}, Max Error: {deePoly_train_errors[2]:.2e}\n")
        f.write("\nTest Data:\n")
        f.write(f"Net Method - MSE: {net_test_errors[0]:.2e}, MAE: {net_test_errors[1]:.2e}, Max Error: {net_test_errors[2]:.2e}\n")
        f.write(f"DeePoly Method - MSE: {deePoly_test_errors[0]:.2e}, MAE: {deePoly_test_errors[1]:.2e}, Max Error: {deePoly_test_errors[2]:.2e}\n")

