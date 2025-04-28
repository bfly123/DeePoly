import numpy as np

def generate_global_field(x: np.ndarray) -> np.ndarray:
    """生成一个在[-1,1]*[-1,1]区域内具有大梯度过渡的二维场
    
    函数在sin曲线边界附近有非常陡峭的梯度变化，而不是直接的间断
    
    Args:
        x: 输入坐标 (n_points, 2)
    Returns:
        y: 函数值 (n_points, 1)
    """
    def sine_large_gradient_function(x, y):
        # 定义sin曲线
        amplitude = 0.4      # 振幅
        frequency = 2.0      # 频率
        
        # 计算sin曲线的值
        sin_curve = amplitude * np.sin(frequency * np.pi * x)
        
        # 计算每个点到sin曲线的距离 (y - sin(x))
        distance_to_curve = y - sin_curve
        
        # 使用双曲正切函数创建陡峭的过渡区
        # tanh函数会在0附近产生最大梯度，远离0时逐渐平缓
        steepness = 15.0     # 控制过渡区的宽度，值越大过渡越陡
        gradient_height = 8.0  # 控制过渡两侧的函数值差异，越大梯度越大
        
        # 创建大梯度过渡
        z = gradient_height * np.tanh(steepness * distance_to_curve)
        
        # 添加一些局部变化以增强视觉效果
        # 振幅设置得比较小，以免掩盖主要的梯度特征
        local_variation = 0.4 * np.sin(3 * x) * np.cos(2 * y)
        
        # 返回最终结果
        return z + local_variation

    # 处理输入数据
    x_coords = x[:, 0]
    y_coords = x[:, 1]
    
    # 计算函数值
    z = sine_large_gradient_function(x_coords, y_coords)
    
    return z.reshape(-1, 1)

def generate_large_gradient_field(x: np.ndarray) -> np.ndarray:
    """生成一个在两个方向都具有大梯度的二维场
    
    函数在x和y方向上都有显著的梯度变化，包含多个大梯度区域
    
    Args:
        x: 输入坐标 (n_points, 2)
    Returns:
        y: 函数值 (n_points, 1)
    """
    def large_gradient_function(x, y):
        # 1. 多个陡峭的高斯函数
        gaussian1 = 3.0 * np.exp(-10.0 * ((x-0.5)**2 + (y-0.5)**2))  # 右上方峰值
        gaussian2 = 2.5 * np.exp(-8.0 * ((x+0.6)**2 + (y+0.7)**2))   # 左下方峰值
        
        # 2. 锐利的正弦波（高频）- 创建在特定区域的高梯度
        sine_x = 0.8 * np.sin(8.0 * x) * np.exp(-2.0 * (y**2))       # x方向的大梯度
        sine_y = 0.8 * np.sin(8.0 * y) * np.exp(-2.0 * (x**2))       # y方向的大梯度
        
        # 3. 双曲正切函数 - 在特定位置创建更陡峭的变化
        tanh_x = 1.5 * np.tanh(5.0 * (x - 0.1))                      # x方向阶跃状大梯度
        tanh_y = 1.5 * np.tanh(5.0 * (y + 0.2))                      # y方向阶跃状大梯度
        
        # 4. 多项式项 - 在特定区域添加非线性梯度
        poly_term = 0.3 * x**3 - 0.3 * y**3
        
        # 组合所有项
        return gaussian1 + gaussian2 + sine_x + sine_y + tanh_x + tanh_y + poly_term
    
    # 处理输入数据
    x_coords = x[:, 0]
    y_coords = x[:, 1]
    
    # 计算函数值
    z = large_gradient_function(x_coords, y_coords)
    
    return z.reshape(-1, 1)

def generate_multi_direction_gradient_field(x: np.ndarray) -> np.ndarray:
    """生成一个在多个任意方向上都具有大梯度的二维场
    
    函数在各个方向上都设计了梯度变化，不仅限于x和y轴方向
    
    Args:
        x: 输入坐标 (n_points, 2)
    Returns:
        y: 函数值 (n_points, 1)
    """
    def multi_direction_gradient_function(x, y):
        # 1. 创建多个方向的大梯度带（径向方向）
        # 计算到不同中心点的距离
        center_points = [
            (0.0, 0.0),   # 原点
            (1.0, 1.0),   # 右上
            (-1.0, 1.0),  # 左上
            (1.0, -1.0),  # 右下
            (-1.0, -1.0)  # 左下
        ]
        
        # 初始化函数值
        z = np.zeros_like(x)
        
        # 在每个中心点周围创建梯度环
        for cx, cy in center_points:
            # 计算到中心点的距离
            r = np.sqrt((x-cx)**2 + (y-cy)**2)
            
            # 创建环形梯度，在特定半径周围有大梯度
            # 控制环的厚度和梯度大小
            ring_radius = 0.3
            ring_thickness = 15.0  # 越大环越薄，梯度越陡
            
            # 在环上创建高梯度
            ring = np.exp(-ring_thickness * (r - ring_radius)**2)
            z += ring
            
        # 2. 创建沿着不同角度的大梯度带（角度方向）
        angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]  # 0°, 45°, 90°, 135°方向
        
        for angle in angles:
            # 计算沿特定角度的投影
            projection = x * np.cos(angle) + y * np.sin(angle)
            
            # 创建高频正弦波形成大梯度
            wave_freq = 6.0  # 频率越高梯度越大
            wave_amp = 0.5   # 振幅
            decay = 0.8     # 衰减因子
            
            # 创建沿特定方向的高梯度带
            # 使用exp函数让波形在垂直于指定方向上逐渐衰减
            ortho_projection = -x * np.sin(angle) + y * np.cos(angle)
            directional_decay = np.exp(-decay * ortho_projection**2)
            
            # 添加到函数值
            z += wave_amp * np.sin(wave_freq * projection) * directional_decay
            
        # 3. 添加旋转场梯度（螺旋状大梯度）
        # 计算与原点的极坐标角度
        theta = np.arctan2(y, x)
        r = np.sqrt(x**2 + y**2)
        
        # 创建螺旋状梯度
        spiral_freq = 5.0  # 角度频率
        radial_freq = 3.0  # 径向频率
        spiral = 0.4 * np.sin(spiral_freq * theta + radial_freq * r) * np.exp(-0.5 * r)
        z += spiral
        
        # 4. 添加一些局部的高梯度奇点
        singularity_points = [
            (0.5, -0.5),
            (-0.7, 0.3),
            (0.3, 0.8)
        ]
        
        for sx, sy in singularity_points:
            # 计算到奇点的距离
            d = np.sqrt((x-sx)**2 + (y-sy)**2)
            
            # 在奇点附近创建尖锐的梯度
            # 使用一个较陡的函数，在接近奇点处创建非常大的梯度
            singularity_strength = 0.25
            z += singularity_strength * np.log(0.1 + d) * np.exp(-8.0 * d)
        
        return z
    
    # 处理输入数据
    x_coords = x[:, 0]
    y_coords = x[:, 1]
    
    # 计算函数值
    z = multi_direction_gradient_function(x_coords, y_coords)
    
    return z.reshape(-1, 1)

def generate_single_circular_gradient(x: np.ndarray) -> np.ndarray:
    """在[-1,1]*[-1,1]区域内生成一个只有单个圆形大梯度区域的二维场
    
    函数在圆环边界处有大梯度，其余区域平滑变化
    
    Args:
        x: 输入坐标 (n_points, 2)
    Returns:
        y: 函数值 (n_points, 1)
    """
    def circular_gradient_function(x, y):
        # 平滑背景场
        background = 0.1 * (x**2 + y**2)
        
        # 计算到圆心(0.2, -0.3)的距离
        center_x, center_y = 0.2, -0.3
        r = np.sqrt((x - center_x)**2 + (y - center_y)**2)
        
        # 设置圆环参数
        ring_radius = 0.4    # 圆环半径
        ring_width = 25.0    # 控制圆环的宽度，值越大圆环越窄，梯度越陡
        ring_height = 2.0    # 控制圆环的高度，值越大梯度越大
        
        # 创建圆环函数 - 在r=ring_radius处有最大值，形成环形大梯度
        ring = ring_height * np.exp(-ring_width * (r - ring_radius)**2)
        
        # 组合背景和圆环
        z = background + ring
        
        # 为了增强圆环处的大梯度效果，可以添加一个非线性变换
        # 这会在圆环周围创建更陡峭的梯度变化
        z = z + 0.5 * np.tanh(6.0 * (z - 0.5))
        
        return z
    
    # 处理输入数据
    x_coords = x[:, 0]
    y_coords = x[:, 1]
    
    # 计算函数值
    z = circular_gradient_function(x_coords, y_coords)
    
    return z.reshape(-1, 1)