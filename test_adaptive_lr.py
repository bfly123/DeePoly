"""
测试自适应学习率版本的PINN
"""
import torch
import torch.nn as nn
import numpy as np
import time
import warnings
warnings.filterwarnings('ignore')

# 导入改进版模型
import sys
sys.path.insert(0, '.')

from standalone_pinn_ac_improved import (
    PINN_AC_Improved,
    generate_training_data,
    ac_equation
)

if __name__ == "__main__":
    print("=" * 80)
    print("测试自适应学习率PINN")
    print("=" * 80)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"设备: {device}\n", flush=True)

    # 创建模型
    model = PINN_AC_Improved(layers=[2, 128, 128, 128, 128, 1]).to(device)
    print(f"参数量: {sum(p.numel() for p in model.parameters())}\n", flush=True)

    # 生成数据
    print("生成数据...", flush=True)
    x_bc, t_bc, u_bc, x_pde, t_pde = generate_training_data(100, 256, device)
    print(f"BC点: {len(x_bc)}, PDE点: {len(x_pde)}\n", flush=True)

    # 自适应学习率配置 - 从1e-2快速衰减
    initial_lr = 1e-2
    lr_thresholds = [1.0, 0.5, 1e-2, 1e-3, 1e-4]
    lr_values = [initial_lr, 0.005, 0.003, 0.001, 0.0005, 0.0001]

    def adjust_learning_rate(optimizer, current_loss):
        """只降不升，避免振荡"""
        current_lr = optimizer.param_groups[0]['lr']
        target_lr = lr_values[0]
        for i, threshold in enumerate(lr_thresholds):
            if current_loss < threshold:
                target_lr = lr_values[i + 1]
        # 只允许下降
        target_lr = min(target_lr, current_lr)
        if abs(current_lr - target_lr) > 1e-8:
            for param_group in optimizer.param_groups:
                param_group['lr'] = target_lr
            return True, target_lr
        return False, current_lr

    # 训练
    n_epochs = 10000
    optimizer = torch.optim.Adam(model.parameters(), lr=initial_lr, betas=(0.999, 0.999))

    print("开始训练（自适应学习率）...")
    print("=" * 80)

    x_bc = x_bc.clone().detach()
    t_bc = t_bc.clone().detach()
    u_bc = u_bc.clone().detach()

    start_time = time.time()
    best_loss = float('inf')

    for epoch in range(n_epochs):
        optimizer.zero_grad()

        # BC loss
        u_bc_pred = model(x_bc, t_bc)
        loss_bc = torch.mean((u_bc_pred - u_bc)**2)

        # PDE loss
        x_pde.requires_grad_(True)
        t_pde.requires_grad_(True)
        u_pde = model(x_pde, t_pde)
        residual = ac_equation(u_pde, x_pde, t_pde)
        loss_pde = torch.mean(residual**2)

        loss = loss_pde + loss_bc
        loss.backward()
        optimizer.step()

        current_loss = loss.item()

        # 自适应调整学习率
        lr_changed, current_lr = adjust_learning_rate(optimizer, current_loss)

        if current_loss < best_loss:
            best_loss = current_loss

        if (epoch + 1) % 1000 == 0 or epoch == 0:
            print(f"Epoch {epoch+1:5d}/{n_epochs} | Loss: {current_loss:.6e} | "
                  f"PDE: {loss_pde.item():.6e} | BC: {loss_bc.item():.6e} | "
                  f"LR: {current_lr:.6e} | Time: {time.time()-start_time:.1f}s", flush=True)

        if lr_changed:
            print(f"   → 学习率调整至 {current_lr:.6e} (loss={current_loss:.6e})", flush=True)

    print("=" * 80)
    print(f"✓ 训练完成! 总用时: {time.time() - start_time:.2f}s")
    print(f"  最佳Loss: {best_loss:.6e}")
    print("=" * 80)
