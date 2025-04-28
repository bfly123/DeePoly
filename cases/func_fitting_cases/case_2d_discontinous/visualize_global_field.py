import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib.gridspec as gridspec
from data_generate import generate_global_field

# 创建网格数据，限制在[-1, 1]×[-1, 1]范围内
n_points = 200
x_1d = np.linspace(-1, 1, n_points)
y_1d = np.linspace(-1, 1, n_points)
x_grid, y_grid = np.meshgrid(x_1d, y_1d)

# 将网格数据转换为输入格式
xy_points = np.column_stack((x_grid.flatten(), y_grid.flatten()))

# 计算函数值
z_values = generate_global_field(xy_points)
z_grid = z_values.reshape(n_points, n_points)

# 计算梯度
dx, dy = np.gradient(z_grid, x_1d, y_1d)
gradient_magnitude = np.sqrt(dx**2 + dy**2)

# 创建图像布局
plt.figure(figsize=(16, 12))
gs = gridspec.GridSpec(2, 2, height_ratios=[1, 1])

# 1. 3D表面图
ax1 = plt.subplot(gs[0, 0], projection='3d')
surf = ax1.plot_surface(x_grid, y_grid, z_grid, cmap=cm.viridis, 
                       linewidth=0, antialiased=True, alpha=0.8)
ax1.set_xlabel('X')
ax1.set_ylabel('Y')
ax1.set_zlabel('Z')
ax1.set_title('圆形大梯度区域函数 3D 表面图')
plt.colorbar(surf, ax=ax1, shrink=0.6, aspect=10, pad=0.1)

# 2. 函数等高线图
ax2 = plt.subplot(gs[0, 1])
contour = ax2.contourf(x_grid, y_grid, z_grid, 30, cmap='viridis')
ax2.set_xlabel('X')
ax2.set_ylabel('Y')
ax2.set_title('函数等高线图')
ax2.set_aspect('equal')
plt.colorbar(contour, ax=ax2)

# 3. 梯度场图
ax3 = plt.subplot(gs[1, 0])
# 为了可视化效果，对梯度进行降采样
step = 8
quiver = ax3.quiver(x_grid[::step, ::step], y_grid[::step, ::step], 
                   dx[::step, ::step], dy[::step, ::step],
                   gradient_magnitude[::step, ::step],
                   cmap='plasma', scale=30, width=0.002)
ax3.set_xlabel('X')
ax3.set_ylabel('Y')
ax3.set_title('梯度场')
ax3.set_aspect('equal')
plt.colorbar(quiver, ax=ax3)

# 4. 梯度强度图
ax4 = plt.subplot(gs[1, 1])
# 使用对数尺度更好地显示梯度变化
log_gradient = np.log1p(gradient_magnitude)
gradient_plot = ax4.contourf(x_grid, y_grid, log_gradient, 30, cmap='plasma')
ax4.set_xlabel('X')
ax4.set_ylabel('Y')
ax4.set_title('梯度强度图 (对数尺度)')
ax4.set_aspect('equal')
plt.colorbar(gradient_plot, ax=ax4)

plt.tight_layout()
plt.savefig('global_field_visualization.png', dpi=300, bbox_inches='tight')
plt.show()

# 输出梯度统计信息
print(f"平均梯度强度: {np.mean(gradient_magnitude)}")
print(f"最大梯度强度: {np.max(gradient_magnitude)}")
print(f"最小梯度强度: {np.min(gradient_magnitude)}")
print(f"梯度强度标准差: {np.std(gradient_magnitude)}")

# 沿着特定剖面显示函数值和梯度变化
plt.figure(figsize=(14, 6))

# 1. 沿x轴的剖面
plt.subplot(1, 2, 1)
x_slice = x_1d
y_slice = np.zeros_like(x_slice)
z_slice = np.array([generate_global_field(np.array([[x, 0]]))[0, 0] for x in x_slice])
dx_slice = np.gradient(z_slice, x_slice)

plt.plot(x_slice, z_slice, 'b-', linewidth=2, label='函数值')
plt.plot(x_slice, dx_slice, 'r--', linewidth=2, label='梯度')
plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
plt.xlabel('x坐标')
plt.ylabel('值')
plt.title('沿x轴的函数值和梯度')
plt.legend()
plt.grid(alpha=0.3)

# 2. 沿半径方向的剖面
plt.subplot(1, 2, 2)
theta = np.linspace(0, 2*np.pi, 8)
r_values = np.linspace(0, 1, 100)
center_x, center_y = 0.0, 0.0

for angle in [0, np.pi/4, np.pi/2, 3*np.pi/4]:
    # 计算沿径向的点
    rx = center_x + r_values * np.cos(angle)
    ry = center_y + r_values * np.sin(angle)
    
    # 计算这些点上的函数值
    z_radial = np.array([generate_global_field(np.array([[x, y]]))[0, 0] 
                        for x, y in zip(rx, ry)])
    
    # 计算梯度（近似）
    dr = r_values[1] - r_values[0]
    dz_dr = np.gradient(z_radial, dr)
    
    plt.plot(r_values, dz_dr, '-', linewidth=2, 
            label=f'梯度 ({int(angle*180/np.pi)}°)')

plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
plt.xlabel('到圆心的距离')
plt.ylabel('梯度大小')
plt.title('沿不同角度的径向梯度分布')
plt.legend()
plt.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('global_field_profiles.png', dpi=300, bbox_inches='tight')
plt.show() 