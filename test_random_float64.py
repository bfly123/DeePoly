"""
测试随机采样 + float64 + 自适应学习率版本
"""
import torch
import sys
sys.path.insert(0, '.')

from standalone_pinn_ac_improved import PINN_AC_Improved, generate_training_data, train_pinn_improved

if __name__ == "__main__":
    print("=" * 80)
    print("测试：随机采样 + float64 + 自适应学习率")
    print("=" * 80)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"设备: {device}\n", flush=True)

    # 创建模型
    model = PINN_AC_Improved(layers=[2, 128, 128, 128, 128, 1]).to(device)
    print(f"参数量: {sum(p.numel() for p in model.parameters())}\n", flush=True)

    # 检查数据类型
    print("检查模型数据类型:", flush=True)
    for name, param in model.named_parameters():
        print(f"  {name}: {param.dtype}", flush=True)
        break
    print()

    # 生成数据
    print("生成随机采样数据...", flush=True)
    x_bc, t_bc, u_bc, x_pde, t_pde = generate_training_data(
        n_interior=10000, n_boundary=1000, device=device
    )
    print(f"BC点: {len(x_bc)}, PDE点: {len(x_pde)}", flush=True)
    print(f"数据类型: x_bc={x_bc.dtype}, x_pde={x_pde.dtype}\n", flush=True)

    # 训练10k epochs
    print("开始训练...", flush=True)
    loss_history, pde_loss_history, bc_loss_history = train_pinn_improved(
        model, x_bc, t_bc, u_bc, x_pde, t_pde,
        n_epochs=10000,
        lr=1e-2,
        device=device,
        print_every=1000
    )

    print("\n" + "=" * 80)
    print("✓ 测试完成!")
    print("=" * 80)
