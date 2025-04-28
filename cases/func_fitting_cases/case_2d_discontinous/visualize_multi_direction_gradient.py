import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib.gridspec as gridspec
from data_generate import generate_multi_direction_gradient_field

# 创建网格数据
n_points = 200
x_1d = np.linspace(-2, 2, n_points)
y_1d = np.linspace(-2, 2, n_points)
x_grid, y_grid = np.meshgrid(x_1d, y_1d)

# 将网格数据转换为输入格式
xy_points = np.column_stack((x_grid.flatten(), y_grid.flatten()))

# 计算函数值
z_values = generate_multi_direction_gradient_field(xy_points)
z_grid = z_values.reshape(n_points, n_points)

# 计算梯度
dx, dy = np.gradient(z_grid, x_1d, y_1d)
gradient_magnitude = np.sqrt(dx**2 + dy**2)

# 创建图像布局
plt.figure(figsize=(18, 12))
gs = gridspec.GridSpec(2, 3, height_ratios=[1, 1])

# 1. 3D表面图
ax1 = plt.subplot(gs[0, 0:2], projection='3d')
surf = ax1.plot_surface(x_grid, y_grid, z_grid, cmap=cm.viridis, 
                       linewidth=0, antialiased=True, alpha=0.8)
ax1.set_xlabel('X')
ax1.set_ylabel('Y')
ax1.set_zlabel('Z')
ax1.set_title('多方向大梯度函数 3D 表面图')
plt.colorbar(surf, ax=ax1, shrink=0.6, aspect=10, pad=0.1)

# 2. 函数等高线图
ax2 = plt.subplot(gs[0, 2])
contour = ax2.contourf(x_grid, y_grid, z_grid, 50, cmap='viridis')
ax2.set_xlabel('X')
ax2.set_ylabel('Y')
ax2.set_title('函数等高线图')
ax2.set_aspect('equal')
plt.colorbar(contour, ax=ax2)

# 3. 梯度场图
ax3 = plt.subplot(gs[1, 0])
# 为了可视化效果，对梯度进行降采样
step = 10
quiver = ax3.quiver(x_grid[::step, ::step], y_grid[::step, ::step], 
                   dx[::step, ::step], dy[::step, ::step],
                   gradient_magnitude[::step, ::step],
                   cmap='plasma', scale=50, width=0.001)
ax3.set_xlabel('X')
ax3.set_ylabel('Y')
ax3.set_title('梯度场')
ax3.set_aspect('equal')
plt.colorbar(quiver, ax=ax3)

# 4. 梯度强度图
ax4 = plt.subplot(gs[1, 1])
gradient_plot = ax4.contourf(x_grid, y_grid, gradient_magnitude, 50, cmap='plasma')
ax4.set_xlabel('X')
ax4.set_ylabel('Y')
ax4.set_title('梯度强度图')
ax4.set_aspect('equal')
plt.colorbar(gradient_plot, ax=ax4)

# 5. 梯度强度热点图
ax5 = plt.subplot(gs[1, 2])
# 使用对数尺度突出显示梯度高低
log_gradient = np.log1p(gradient_magnitude)
hotspot = ax5.imshow(log_gradient, extent=[-2, 2, -2, 2], 
                    origin='lower', cmap='hot', aspect='auto')
ax5.set_xlabel('X')
ax5.set_ylabel('Y')
ax5.set_title('梯度强度热点图 (对数尺度)')
plt.colorbar(hotspot, ax=ax5)

plt.tight_layout()
plt.savefig('multi_direction_gradient_visualization.png', dpi=300, bbox_inches='tight')
plt.show()

# 输出一些梯度统计信息
print(f"平均梯度强度: {np.mean(gradient_magnitude)}")
print(f"最大梯度强度: {np.max(gradient_magnitude)}")
print(f"最小梯度强度: {np.min(gradient_magnitude)}")
print(f"梯度强度标准差: {np.std(gradient_magnitude)}")

# 显示梯度强度分布
plt.figure(figsize=(10, 6))
plt.hist(gradient_magnitude.flatten(), bins=100, alpha=0.7)
plt.title('梯度强度分布直方图')
plt.xlabel('梯度强度')
plt.ylabel('频率')
plt.grid(alpha=0.3)
plt.savefig('gradient_magnitude_histogram.png', dpi=300, bbox_inches='tight')
plt.show() 