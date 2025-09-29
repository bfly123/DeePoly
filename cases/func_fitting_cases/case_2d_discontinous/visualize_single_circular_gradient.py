import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib.gridspec as gridspec
from data_generate import generate_single_circular_gradient

# 创建网格数据，限制在[-1, 1]×[-1, 1]范围内
n_points = 200
x_1d = np.linspace(-1, 1, n_points)
y_1d = np.linspace(-1, 1, n_points)
x_grid, y_grid = np.meshgrid(x_1d, y_1d)

# 将网格数据转换为输入格式
xy_points = np.column_stack((x_grid.flatten(), y_grid.flatten()))

# 计算函数值
z_values = generate_single_circular_gradient(xy_points)
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
ax1.set_title('单圆形大梯度区域函数 3D 表面图')
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
plt.savefig('single_circular_gradient_visualization.png', dpi=300, bbox_inches='tight')
plt.show()

# 输出梯度统计信息
print(f"平均梯度强度: {np.mean(gradient_magnitude)}")
print(f"最大梯度强度: {np.max(gradient_magnitude)}")
print(f"最小梯度强度: {np.min(gradient_magnitude)}")
print(f"梯度强度标准差: {np.std(gradient_magnitude)}")

# 显示梯度强度沿着特定路径的变化
# 从圆心向外的径向剖面
center_x, center_y = 0.2, -0.3
radius = 0.4
theta = np.linspace(0, 2*np.pi, 100)
n_samples = 50
radii = np.linspace(0, 0.8, n_samples)

plt.figure(figsize=(12, 5))

# 绘制剖面位置示意图
plt.subplot(1, 2, 1)
plt.contourf(x_grid, y_grid, z_grid, 30, cmap='viridis')
for angle in [0, np.pi/4, np.pi/2, 3*np.pi/4]:
    rx = center_x + radii * np.cos(angle)
    ry = center_y + radii * np.sin(angle)
    plt.plot(rx, ry, 'r-', linewidth=1.5)
    plt.plot(rx[-1], ry[-1], 'ro')
plt.plot(center_x, center_y, 'ro', markersize=5)
plt.xlabel('X')
plt.ylabel('Y')
plt.title('径向剖面位置')
plt.axis('equal')
plt.grid(alpha=0.3)

# 绘制剖面梯度强度
plt.subplot(1, 2, 2)
for i, angle in enumerate([0, np.pi/4, np.pi/2, 3*np.pi/4]):
    # 计算剖面上各点的坐标
    rx = center_x + radii * np.cos(angle)
    ry = center_y + radii * np.sin(angle)
    
    # 使用线性插值获取这些点上的梯度值
    from scipy.interpolate import interp2d
    gradient_interp = interp2d(x_1d, y_1d, gradient_magnitude, kind='linear')
    grad_profile = np.array([gradient_interp(x, y)[0] for x, y in zip(rx, ry)])
    
    # 绘制梯度剖面
    plt.plot(radii, grad_profile, '-', linewidth=2, 
             label=f'角度 {int(angle*180/np.pi)}°')

plt.xlabel('到圆心的距离')
plt.ylabel('梯度强度')
plt.title('不同角度径向剖面的梯度强度')
plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig('gradient_profiles.png', dpi=300, bbox_inches='tight')
plt.show() 