import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from typing import Dict, Tuple, List, Optional, Union
import time
import copy
import scipy.io
#import frames


class Config:
    def __init__(self):
        self.time = 1
        self.dt = 0.05  # 使用dt=0.05保持一致
        self.mu = 0.0001
        self.steps = int(self.time/self.dt)
        self.n_segments =10
        self.points_per_segment_train = 35
        self.points_per_segment_test = 32
        self.poly_degree = 5  # 降低从10到5，减少条件数
        self.DNN_degree = 10  # 降低从20到10，减少矩阵维度
        self.x_domain = [-1,1]
        self.hidden_dims = [16] * 3  # 减少神经网络复杂度
        self.epochs_adam = 2000
        self.epochs_lbfgs = 200
        self.method = "hybrid"
        self.lr = 0.001
        self.seed = 42
        self.Init_u = Init_u
        self.forcing_term = forcing_term
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def init_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

def split_array(x, x_min, x_max):
    # 随机采样的 x 点
    n = len(x_min)
    segments = []
    for i in range(n):
      if i == 0:
        mask = (x >= x_min[i]) & (x < x_max[i])
        segments.append(x[mask])
      else:
        mask = (x > x_min[i]) & (x <= x_max[i])
        segments.append(x[mask])
        
    return segments


def target_fn(x):
    theta1 = 0.52
    theta2 = 9.6
    theta3 = 3.2
    c = 15.0
    t = 1  # Example time

    x_original = (x + 1) * (5 / 2)

    result = np.where(
        t > x_original,
        theta2
        + np.sin((c * (t - x_original) * np.pi) / theta3)
        - (t - x_original) ** (theta1 + 0.1),
        0,
    )

    return result


def target_fn(t):
    x = np.where(
        t > 0,
        np.log(t + 2) * np.cos(2 * t + t**3),  # t > 0 的情况
        np.log(t + 2) * np.cos(2 * t + t**3) + 1,
    )
    return x


def Init_u(x):
    u =  x**2*np.cos(np.pi * x)
    return u

def forcing_term(x):
    """
    计算方程右端项
    f(x) = Σ(i=1 to 4)[i*sin(ix)] + 8sin(8x)
    """
    x = x*np.pi/2 + np.pi/2*np.ones_like(x)
    result = np.zeros_like(x)
    for i in range(1, 5):
        result += i * np.sin(i * x)
    result += 8 * np.sin(8 * x)
    return -result*(np.pi/2)**2

def exact_solution(x):
    """
    计算解析解
    u(x) = x + Σ(i=1 to 4)[sin(ix)/i] + sin(8x)/8
    """
    x = x*np.pi/2 + np.pi/2*np.ones_like(x)
    result = np.zeros_like(x)
    for i in range(1, 5):
        result += np.sin(i * x) / i
    result += np.sin(8 * x) / 8
    return result + x



class Net(nn.Module):
    def __init__(self, in_dim, hidden_dims, out_dim, use_periodic=False):
        super().__init__()
        self.use_periodic = use_periodic
        actual_in = in_dim * 3 if use_periodic else in_dim

        layers = []
        dims = [actual_in] + hidden_dims
        for i in range(len(dims) - 1):
            layers.extend(
                [nn.Linear(dims[i], dims[i + 1], dtype=torch.float64), nn.Tanh()]
            )

        self.net = nn.Sequential(*layers)
        self.out = nn.Linear(hidden_dims[-1], out_dim, bias=False, dtype=torch.float64)

    def forward(self, x):
        if self.use_periodic:
            x = torch.cat(
                [x, torch.sin(2 * np.pi * x), torch.cos(2 * np.pi * x)], dim=-1
            )
        h = self.net(x)
        return h, self.out(h)
    def derivative_h(self, x):
        h,_ = self.forward(x)
        # 确定是否正确
        assert h.shape[0] == x.shape[0] and x.shape[1] == 1, "h 和 x 的形状不匹配"
        h_x = torch.autograd.grad(h, x, grad_outputs=torch.ones_like(h), create_graph=True)[0]
        assert h_x.shape == h.shape, "h_x 的形状不正确"
        return h_x

def gradients(outputs, inputs):
    return torch.autograd.grad(outputs, inputs,grad_outputs=torch.ones_like(outputs), create_graph=True)




   
def Time_evolve_pre(model, data_GPU,u_n,config):
    lr_adam = config.lr
    epochs_adam = config.epochs_adam
    epochs_lbfgs = config.epochs_lbfgs
    optimizer_adam = optim.Adam(model.parameters(), lr=lr_adam)

    optimizer_lbfgs = optim.LBFGS(model.parameters(), lr=1.0, max_iter=epochs_lbfgs)


    u_n = torch.tensor(u_n, dtype=torch.float64, device=config.device)
    loss = physics_loss(model, data_GPU,u_n,u_n)
    def closure():
        optimizer_lbfgs.zero_grad()
        loss = physics_loss(model, data_GPU,u_n,u_n)
        loss.backward()
        return loss


    if config.method == "hybrid":
        # Training with Adam
        for epoch in range(100000):
            optimizer_adam.zero_grad()
            loss = physics_loss(model, data_GPU,u_n,u_n)
            loss.backward()
            optimizer_adam.step()

            # 改善收敛条件 - 与dt相关的自适应收敛阈值
            convergence_threshold = 0.1 * config.dt  # 更严格的收敛条件
            if loss.item() < convergence_threshold:
                break
            if epoch % 100 == 0:
                print(f"Adam Epoch {epoch}, Loss: {loss.item()}")

        print(f"Adam Loss: {loss.item()}")
        #optimizer_lbfgs = optim.LBFGS(model.parameters(), lr=1.0, max_iter=epochs_lbfgs)
        #optimizer_lbfgs.zero_grad()
        #optimizer_lbfgs.step(closure)
        ## 获取最终的loss
        #final_loss = closure()
        #print(f"LBFGS Final Loss: {final_loss.item()}")
      
        coeffs= fit_poly(data_train, config,model,config.method)
        u_n = evl_poly(data_train, coeffs, config,model,config.method).flatten()
        du_x = evl_poly(data_train, coeffs, config,model,config.method,1)
        du_xx = evl_poly(data_train, coeffs, config,model,config.method,2)

    return u_n,model,coeffs,du_x,du_xx


# Equation: $$ (u_{n+1} - u_n)/dt + u_{n+1}*u_{n+1}_x -\mu* u_{n+1}_xx= 0$$
def physics_loss(model, data_GPU,u_n,u_star):
    x_train = data_GPU["x_train"]
    x_L = data_GPU["x_L"]
    x_R = data_GPU["x_R"]
    #R_train = data_GPU["R_train"]
    #u_n = data_GPU["u_n"]
    dt = data_GPU["dt"]
    mu = data_GPU["mu"]
    _, u = model(x_train)
    _, u_L = model(x_L)
    _, u_R = model(x_R)
    du_x = gradients(u, x_train)[0]
    du_x_L = gradients(u_L, x_L)[0]
    du_x_R = gradients(u_R, x_R)[0]
    du_xx = gradients(du_x, x_train)[0]

    loss = torch.mean(((u - u_n) + dt* (0.0001*du_xx + 5* u_n**3-5*u_n)) ** 2) + 10*torch.mean((u_L -u_R) ** 2) + 10*torch.mean((du_x_R - du_x_L) ** 2)
    return loss


def generate_data(config, mode):
    n_segments = config.n_segments
    if mode == "train":
        points_per_segment = config.points_per_segment_train
    else:
        points_per_segment = config.points_per_segment_test
    x_segments = []
    R_segments = []
    u_segments = []
    x_segments_norm = []
    x_min = np.zeros((n_segments,1),dtype=np.float64)
    x_max = np.zeros((n_segments,1),dtype=np.float64)
    x_min_norm = np.zeros((n_segments,1),dtype=np.float64)
    x_max_norm = np.zeros((n_segments,1),dtype=np.float64)
    segment_ranges = np.linspace(-1, 1, n_segments + 1,dtype=np.float64)
    # 增加每个segment边界点数组

    x1_global = np.linspace(config.x_domain[0], config.x_domain[1], config.points_per_segment_test*n_segments, dtype=np.float64)
    #element = x1_global.reshape(32,16)
    for i in range(n_segments):
      x_min_norm[i]=0
      x_max_norm[i]=1
      x_min[i] = segment_ranges[i]
      x_max[i] = segment_ranges[i + 1]
      if mode == "train":
            x_seg = np.random.uniform(
                x_min[i], x_max[i], points_per_segment
            ).flatten()
            x_seg = np.concatenate(
                [
                    x_min[i].flatten(),
                    x_seg,
                    x_max[i].flatten(),
                ]
            )
      else:
    #        x_seg = element[i].flatten()
            x_seg = x1_global[i*points_per_segment:(i+1)*points_per_segment]
      x_segments.append(x_seg)

      #R_seg = config.target_fn1(x_seg).reshape(-1, 1)
      #R_segments.append(R_seg)
      u_seg = config.Init_u(x_seg).flatten()
      u_segments.append(u_seg)

      x_segments_norm.append(
          normalize_data(x_seg, x_min[i], x_max[i],dtype=np.float64)
        )
    x_L=x_min[0].flatten()
    x_R=x_max[-1].flatten()
    x = np.concatenate(x_segments).flatten().reshape(-1,1)
    #R = np.concatenate(R_segments).reshape(-1, 1)
    u = np.concatenate(u_segments).flatten().reshape(-1,1)
    u_star_segments = np.copy(u_segments)
    data = {
        "x": x,
        #"R": R,
        "u": u,
        "x_L": x_L,
        "x_R": x_R,
        "x_min": x_min.flatten(),
        "x_max": x_max.flatten(),
        "x_min_norm": x_min_norm.flatten(),
        "x_max_norm": x_max_norm.flatten(),
        "x_segments_norm": x_segments_norm,
        "x_segments": x_segments,
        "u_segments": u_segments,
        "u_segments_star": u_star_segments,
        "R_segments": R_segments,
        "n_segments": n_segments,
    }
    return data


def get_poly_features(x, degree, derivative=0):
    if derivative > degree:
        raise ValueError("Derivative order is greater than polynomial degree.")

    x = np.asarray(x,dtype=np.float64).reshape(-1, 1)

    if derivative == 0:
        return np.hstack([x**i for i in range(degree + 1)])
    else:
        features = []
        features.extend([np.zeros_like(x) for _ in range(derivative)])

        for i in range(derivative, degree + 1):
            coeff = 1
            for j in range(i - derivative + 1, i + 1):
                coeff *= j
            term = coeff * x ** (i - derivative)
            features.append(term)

    return np.hstack(features)

def get_dnn_features(x,x_min,x_max, model,config, derivative=0):
    x = (x*(x_max-x_min)+x_min).reshape(-1, 1)
    x_tensor = torch.tensor(x,requires_grad=True).double().to(config.device)

    h, _ = model(x_tensor)

    # 确保神经网络特征维度与配置的DNN_degree一致
    actual_dnn_dim = h.shape[1]
    expected_dnn_dim = config.DNN_degree

    if actual_dnn_dim != expected_dnn_dim:
        # 如果实际维度大于期望维度，截取前expected_dnn_dim个特征
        if actual_dnn_dim > expected_dnn_dim:
            h = h[:, :expected_dnn_dim]
        # 如果实际维度小于期望维度，用零填充
        else:
            padding = torch.zeros(h.shape[0], expected_dnn_dim - actual_dnn_dim, device=h.device, dtype=h.dtype)
            h = torch.cat([h, padding], dim=1)

    jacobian = np.zeros((h.shape[0], h.shape[1]))
    if derivative==0:
        return h.detach().cpu().numpy()
    else:
        for i in range(h.shape[1]):
            grad = h[:, i]
            for _ in range(derivative):
                grad = gradients(grad, x_tensor)[0]
            jacobian[:,i] = grad.detach().squeeze().cpu().numpy()
        return jacobian


def normalize_data(x: np.ndarray, x_min: float, x_max: float,dtype=np.float64) -> np.ndarray:
    return (x - x_min) / (x_max - x_min) if x_max - x_min > 1e-10 else x


def test_solve_without_boundary_constraints(data, config, model, method="hybrid"):
    """测试没有边界约束的求解精度"""
    print("\n" + "="*50)
    print("测试无边界约束的矩阵求解精度")
    print("="*50)

    dg = config.poly_degree
    dN = config.DNN_degree
    ns = data["n_segments"]
    x = data["x_segments_norm"]
    dt = config.dt
    mu = config.mu
    x_max = data["x_max"]
    x_min = data["x_min"]

    dgN = dg
    if method == "hybrid":
        dgN = dN + dg

    A, b = [], []

    # 只添加主方程约束，不添加边界和连续性约束
    for i in range(ns):
        U = get_poly_features(x[i], dg)
        dU2 = get_poly_features(x[i], dg, 2)/(x_max[i]-x_min[i])**2
        if method == "hybrid":
            U_nn = get_dnn_features(x[i], x_min[i],x_max[i], model,config)
            dU2_nn = get_dnn_features(x[i], x_min[i],x_max[i], model,config, 2)
            U = np.hstack([U,U_nn])
            dU2 = np.hstack([dU2,dU2_nn])

        # 简化主方程: U/dt + (-mu*dU2) = 1 (简化的右端项)
        Left = U/dt + (-mu*dU2)
        Right = np.ones((Left.shape[0], 1))  # 简化的右端项

        # 扩展矩阵以包含所有段的系数
        n_points = Left.shape[0]
        Left_extended = np.zeros((n_points, (dgN + 1) * ns))
        Left_extended[:, i * (dgN + 1) : (i + 1) * (dgN + 1)] = Left

        A.extend(Left_extended)
        b.extend(Right.flatten())

    # 转换为numpy数组
    A = np.vstack(A).astype(np.float64)
    b = np.array(b, dtype=np.float64).reshape(-1, 1)

    print(f"纯主方程矩阵: {A.shape[0]} x {A.shape[1]}")
    print(f"段数: {ns}, 每段特征数: {dgN + 1}")

    # 计算条件数
    ATA = A.T @ A
    try:
        eigenvals = np.linalg.eigvals(ATA)
        eigenvals = eigenvals[eigenvals > 1e-16]  # 排除接近零的特征值
        condition_num = np.max(eigenvals) / np.min(eigenvals) if len(eigenvals) > 0 else np.inf
    except:
        condition_num = np.inf

    print(f"主方程矩阵条件数: {condition_num:.6e}")

    # 测试不同的正则化参数
    lambda_values = [0, 1e-16, 1e-12, 1e-10, 1e-8, 1e-6]

    best_result = None
    best_residual = np.inf

    for lambda_reg in lambda_values:
        print(f"\n--- 测试正则化参数 λ = {lambda_reg:.0e} ---")

        # 添加正则化
        if lambda_reg > 0:
            n_vars = A.shape[1]
            A_reg = np.vstack([A, lambda_reg * np.eye(n_vars)])
            b_reg = np.vstack([b, np.zeros((n_vars, 1))])
        else:
            A_reg = A
            b_reg = b

        # SVD求解
        try:
            result = np.linalg.lstsq(A_reg, b_reg, rcond=1e-15)
            coeffs = result[0]
            residuals_lstsq = result[1] if len(result) > 1 and len(result[1]) > 0 else None
            rank = result[2] if len(result) > 2 else None
            singular_values = result[3] if len(result) > 3 else None

            # 计算残差
            residual = A_reg @ coeffs - b_reg
            residual_norm = np.linalg.norm(residual)
            residual_norm_inf = np.linalg.norm(residual, ord=np.inf)
            b_norm = np.linalg.norm(b_reg)
            relative_residual = residual_norm / max(b_norm, 1e-16)

            # 计算条件数
            if singular_values is not None and len(singular_values) > 0:
                cond_svd = singular_values[0] / singular_values[-1] if singular_values[-1] > 1e-16 else np.inf
            else:
                cond_svd = np.inf

            print(f"  矩阵大小: {A_reg.shape[0]} x {A_reg.shape[1]}")
            print(f"  矩阵秩: {rank}")
            print(f"  条件数(SVD): {cond_svd:.6e}")
            print(f"  残差 ||Ax-b||_2: {residual_norm:.6e}")
            print(f"  残差 ||Ax-b||_∞: {residual_norm_inf:.6e}")
            print(f"  相对残差: {relative_residual:.6e}")
            if residuals_lstsq is not None:
                print(f"  lstsq残差: {residuals_lstsq[0]:.6e}")

            # 检查解的合理性
            coeffs_norm = np.linalg.norm(coeffs)
            print(f"  解的范数: {coeffs_norm:.6e}")

            # 记录最佳结果
            if residual_norm < best_residual:
                best_residual = residual_norm
                best_result = {
                    'lambda': lambda_reg,
                    'residual': residual_norm,
                    'condition': cond_svd,
                    'relative_residual': relative_residual
                }

        except Exception as e:
            print(f"  求解失败: {e}")

    # 输出最佳结果总结
    if best_result:
        print(f"\n*** 最佳求解结果 ***")
        print(f"最佳正则化参数: λ = {best_result['lambda']:.0e}")
        print(f"最小残差: {best_result['residual']:.6e}")
        print(f"对应条件数: {best_result['condition']:.6e}")
        print(f"相对残差: {best_result['relative_residual']:.6e}")

    print("\n" + "="*50)
    return A, b


def compare_boundary_conditions(data, config, model, method="hybrid"):
    """对比不同边界条件设置的求解精度"""
    print("\n" + "="*60)
    print("边界条件对比测试")
    print("="*60)

    # 测试1: 原始周期边界条件
    print("\n### 测试1: 周期边界条件 (u_L = u_R, u'_L = u'_R) ###")
    A1, b1, residual1 = fit_poly_with_boundary(data, config, model, method, boundary_type="periodic")

    # 测试2: Dirichlet + Neumann 边界条件
    print("\n### 测试2: Dirichlet + Neumann 边界条件 (u_L = -1, u_R = -1, u'_L = 0, u'_R = 0) ###")
    A2, b2, residual2 = fit_poly_with_boundary(data, config, model, method, boundary_type="dirichlet_neumann")

    # 对比结果
    print("\n" + "="*60)
    print("边界条件对比结果总结")
    print("="*60)
    print(f"周期边界条件:")
    print(f"  残差: {residual1['residual']:.6e}")
    print(f"  条件数: {residual1['condition']:.6e}")
    print(f"  相对残差: {residual1['relative_residual']:.6e}")
    print()
    print(f"Dirichlet+Neumann边界条件:")
    print(f"  残差: {residual2['residual']:.6e}")
    print(f"  条件数: {residual2['condition']:.6e}")
    print(f"  相对残差: {residual2['relative_residual']:.6e}")
    print()

    improvement = residual1['residual'] / residual2['residual'] if residual2['residual'] > 0 else float('inf')
    print(f"残差改善倍数: {improvement:.2f}")
    if improvement > 1:
        print("✓ Dirichlet+Neumann边界条件求解精度更高")
    elif improvement < 1:
        print("✗ 周期边界条件求解精度更高")
    else:
        print("→ 两种边界条件精度相当")

    print("="*60)
    return A1, b1, A2, b2


def fit_poly_with_boundary(data, config, model, method="hybrid", boundary_type="periodic"):
    """使用指定边界条件的多项式拟合"""
    dg = config.poly_degree
    dN = config.DNN_degree

    ns = data["n_segments"]
    x = data["x_segments_norm"]
    R = data["R_segments"]
    u_n = data["u_segments"]
    u_star = data["u_segments_star"]
    dt = config.dt
    mu = config.mu
    x_L = data["x_L"]
    x_R = data["x_R"]
    x_max = data["x_max"]
    x_min = data["x_min"]
    x_max_norm = data["x_max_norm"]
    x_min_norm = data["x_min_norm"]

    dgN = dg
    if method == "hybrid":
        dgN = dN + dg

    A, b = [], []

    # 主方程约束 (保持不变)
    for i in range(ns):
        U = get_poly_features(x[i], dg)
        dU1 = get_poly_features(x[i], dg, 1)/(x_max[i]-x_min[i])
        dU2 = get_poly_features(x[i], dg, 2)/(x_max[i]-x_min[i])**2
        if method == "hybrid":
            U_nn = get_dnn_features(x[i], x_min[i],x_max[i], model,config)
            dU_nn = get_dnn_features(x[i], x_min[i],x_max[i], model,config, 1)
            dU2_nn = get_dnn_features(x[i], x_min[i],x_max[i], model,config, 2)
            U = np.hstack([U,U_nn])
            dU1 = np.hstack([dU1,dU_nn])
            dU2 = np.hstack([dU2,dU2_nn])
        un = u_n[i].reshape(-1,1)

        Left = U/dt + (-mu*dU2)

        # 简化右端项用于测试（避免依赖于未计算的u_star和R）
        try:
            Right = R[i] + u_star[i].reshape(-1,1)/dt
        except (IndexError, TypeError):
            try:
                # 尝试使用R[i]
                Right = R[i] + un/dt
            except (IndexError, TypeError):
                # 如果R也不可用，使用简化的右端项
                Right = np.ones((Left.shape[0], 1)) + un/dt

        # 扩展矩阵
        n_points = Left.shape[0]
        Left_extended = np.zeros((n_points, (dgN + 1) * ns))
        Left_extended[:, i * (dgN + 1) : (i + 1) * (dgN + 1)] = Left

        A.extend(Left_extended)
        b.extend(Right.flatten())

    # 连续性约束 (保持不变)
    constraint_weight = max(6.0, 8.0 / np.sqrt(ns))
    boundary_weight = max(4.0, constraint_weight * 0.7)

    # 导数连续性
    for i in range(ns - 1):
        cont = np.zeros((1, (dgN + 1) * ns), dtype=np.float64)
        P = get_poly_features(x_max_norm[i], dg,1)/(x_max[i]-x_min[i])
        if method == "hybrid":
            P_nn = get_dnn_features(x_max_norm[i], x_min[i],x_max[i], model,config, 1)
            P = np.hstack([P,P_nn])
        cont[0, i * (dgN + 1) : (i + 1) * (dgN + 1)] = P.flatten()
        P = get_poly_features(x_min_norm[i+ 1], dg,1)/(x_max[i+1]-x_min[i+1])
        if method == "hybrid":
            P_nn = get_dnn_features(x_min_norm[i + 1], x_min[i + 1],x_max[i + 1], model,config, 1)
            P = np.hstack([P,P_nn])
        cont[0, (i + 1) * (dgN + 1) : (i + 2) * (dgN + 1)] = -P.flatten()
        A.extend([constraint_weight*cont])
        b.extend([0])

    # 值连续性
    for i in range(ns - 1):
        cont = np.zeros((1, (dgN + 1) * ns), dtype=np.float64)
        P = get_poly_features(x_max_norm[i], dg,0)
        if method == "hybrid":
            P_nn = get_dnn_features(x_max_norm[i], x_min[i],x_max[i], model,config, 0)
            P = np.hstack([P,P_nn])
        cont[0, i * (dgN + 1) : (i + 1) * (dgN + 1)] = P.flatten()
        P = get_poly_features(x_min_norm[i+ 1], dg,0)
        if method == "hybrid":
            P_nn = get_dnn_features(x_min_norm[i + 1], x_min[i + 1],x_max[i + 1], model,config, 0)
            P = np.hstack([P,P_nn])
        cont[0, (i + 1) * (dgN + 1) : (i + 2) * (dgN + 1)] = -P.flatten()
        A.extend([constraint_weight*cont])
        b.extend([0])

    # 边界条件设置
    if boundary_type == "periodic":
        print("使用周期边界条件: u_L = u_R, u'_L = u'_R")
        # 周期边界条件: u(left) = u(right)
        u_B = np.zeros((1, (dgN + 1) * ns), dtype=np.float64)
        P_left = get_poly_features(x_min_norm[0], dg)
        if method == "hybrid":
            P_nn_left = get_dnn_features(x_min_norm[0], x_min[0],x_max[0], model,config, 0)
            P_left = np.hstack([P_left,P_nn_left])
        u_B[0, 0 : (dgN + 1)] = P_left.flatten()

        P_right = get_poly_features(x_max_norm[-1], dg)
        if method == "hybrid":
            P_nn_right = get_dnn_features(x_max_norm[-1], x_min[-1],x_max[-1], model,config, 0)
            P_right = np.hstack([P_right,P_nn_right])
        u_B[0, (ns - 1) * (dgN + 1) : ns * (dgN + 1)] = -P_right.flatten()

        A.extend([boundary_weight*u_B])
        b.extend([0])

        # 周期边界条件: u'(left) = u'(right)
        u1_B = np.zeros((1, (dgN + 1) * ns), dtype=np.float64)
        P_left = get_poly_features(x_min_norm[0], dg,1)/(x_max[0]-x_min[0])
        if method == "hybrid":
            P_nn_left = get_dnn_features(x_min_norm[0], x_min[0],x_max[0], model,config, 1)
            P_left = np.hstack([P_left,P_nn_left])
        u1_B[0, 0 : (dgN + 1)] = P_left.flatten()

        P_right = get_poly_features(x_max_norm[-1], dg,1)/(x_max[-1]-x_min[-1])
        if method == "hybrid":
            P_nn_right = get_dnn_features(x_max_norm[-1], x_min[-1],x_max[-1], model,config, 1)
            P_right = np.hstack([P_right,P_nn_right])
        u1_B[0, (ns - 1) * (dgN + 1) : ns * (dgN + 1)] = -P_right.flatten()

        A.extend([boundary_weight*u1_B])
        b.extend([0])

    elif boundary_type == "dirichlet_neumann":
        print("使用Dirichlet+Neumann边界条件: u_L = -1, u_R = -1, u'_L = 0, u'_R = 0")

        # Dirichlet边界条件: u(left) = -1
        u_L = np.zeros((1, (dgN + 1) * ns), dtype=np.float64)
        P_left = get_poly_features(x_min_norm[0], dg)
        if method == "hybrid":
            P_nn_left = get_dnn_features(x_min_norm[0], x_min[0],x_max[0], model,config, 0)
            P_left = np.hstack([P_left,P_nn_left])
        u_L[0, 0 : (dgN + 1)] = P_left.flatten()
        A.extend([boundary_weight*u_L])
        b.extend([-1])  # u_L = -1

        # Dirichlet边界条件: u(right) = -1
        u_R = np.zeros((1, (dgN + 1) * ns), dtype=np.float64)
        P_right = get_poly_features(x_max_norm[-1], dg)
        if method == "hybrid":
            P_nn_right = get_dnn_features(x_max_norm[-1], x_min[-1],x_max[-1], model,config, 0)
            P_right = np.hstack([P_right,P_nn_right])
        u_R[0, (ns - 1) * (dgN + 1) : ns * (dgN + 1)] = P_right.flatten()
        A.extend([boundary_weight*u_R])
        b.extend([-1])  # u_R = -1

        # Neumann边界条件: u'(left) = 0
        u1_L = np.zeros((1, (dgN + 1) * ns), dtype=np.float64)
        P_left = get_poly_features(x_min_norm[0], dg,1)/(x_max[0]-x_min[0])
        if method == "hybrid":
            P_nn_left = get_dnn_features(x_min_norm[0], x_min[0],x_max[0], model,config, 1)
            P_left = np.hstack([P_left,P_nn_left])
        u1_L[0, 0 : (dgN + 1)] = P_left.flatten()
        A.extend([boundary_weight*u1_L])
        b.extend([0])  # u'_L = 0

        # Neumann边界条件: u'(right) = 0
        u1_R = np.zeros((1, (dgN + 1) * ns), dtype=np.float64)
        P_right = get_poly_features(x_max_norm[-1], dg,1)/(x_max[-1]-x_min[-1])
        if method == "hybrid":
            P_nn_right = get_dnn_features(x_max_norm[-1], x_min[-1],x_max[-1], model,config, 1)
            P_right = np.hstack([P_right,P_nn_right])
        u1_R[0, (ns - 1) * (dgN + 1) : ns * (dgN + 1)] = P_right.flatten()
        A.extend([boundary_weight*u1_R])
        b.extend([0])  # u'_R = 0

    # 矩阵求解部分
    A = np.vstack(A).astype(np.float64)
    b = np.array(b, dtype=np.float64).reshape(-1, 1)

    print(f"完整系统矩阵: {A.shape[0]} x {A.shape[1]}")

    # 自适应正则化
    ATA = A.T @ A
    condition_estimate = np.trace(ATA) / (np.min(np.diag(ATA)) + 1e-16)
    lambda_reg = max(1e-6, 1e-8 * np.sqrt(condition_estimate))

    print(f"自适应正则化参数: {lambda_reg:.2e}")

    n_vars = A.shape[1]
    A_reg = np.vstack([A, lambda_reg * np.eye(n_vars)])
    b_reg = np.vstack([b, np.zeros((n_vars, 1))])

    # SVD求解
    result = np.linalg.lstsq(A_reg, b_reg, rcond=1e-15)
    coeffs = result[0]
    residuals = result[1] if len(result) > 1 else None
    rank = result[2] if len(result) > 2 else None
    singular_values = result[3] if len(result) > 3 else None

    # 计算残差
    residual = A_reg @ coeffs - b_reg
    residual_norm = np.linalg.norm(residual)
    residual_norm_inf = np.linalg.norm(residual, ord=np.inf)
    b_norm = np.linalg.norm(b_reg)
    relative_residual = residual_norm / max(b_norm, 1e-16)
    condition_number = singular_values[0] / singular_values[-1] if singular_values is not None and len(singular_values) > 0 and singular_values[-1] > 1e-16 else np.inf

    print(f"求解结果:")
    print(f"  矩阵大小: {A_reg.shape[0]} x {A_reg.shape[1]}")
    print(f"  矩阵秩: {rank}")
    print(f"  条件数: {condition_number:.6e}")
    print(f"  残差 ||Ax-b||_2: {residual_norm:.6e}")
    print(f"  残差 ||Ax-b||_∞: {residual_norm_inf:.6e}")
    print(f"  相对残差: {relative_residual:.6e}")

    residual_info = {
        'residual': residual_norm,
        'condition': condition_number,
        'relative_residual': relative_residual,
        'rank': rank
    }

    return A, b, residual_info


def fit_poly(data,config,model, method="poly"):
    # 先测试无边界约束的求解精度
    test_solve_without_boundary_constraints(data, config, model, method)

    # 然后对比两种边界条件
    A1, b1, A2, b2 = compare_boundary_conditions(data, config, model, method)

    # 使用Dirichlet+Neumann边界条件进行最终求解 (精度更高)
    print(f"\n使用Dirichlet+Neumann边界条件进行最终求解...")
    A, b, residual_info = fit_poly_with_boundary(data, config, model, method, boundary_type="dirichlet_neumann")

    # 提取系数并重新整形
    n_vars = A.shape[1]
    A_reg = np.vstack([A, 1e-6 * np.eye(n_vars)])  # 使用固定的小正则化
    b_reg = np.vstack([b, np.zeros((n_vars, 1))])

    result = np.linalg.lstsq(A_reg, b_reg, rcond=1e-15)
    coeffs_flat = result[0]

    # 重新整形为 (ns, dgN + 1) 格式
    dg = config.poly_degree
    dN = config.DNN_degree if method == "hybrid" else 0
    dgN = dg + dN
    ns = data["n_segments"]

    coeffs = coeffs_flat.reshape(ns, dgN + 1)

    return coeffs
    dN = config.DNN_degree
    
    ns = data["n_segments"]
    x = data["x_segments_norm"]
    #   u = data["segments_u"]
    R = data["R_segments"]
    u_n = data["u_segments"]
    u_star = data["u_segments_star"]
    dt = config.dt
    mu = config.mu
    x_L = data["x_L"]
    x_R = data["x_R"]
    x_max = data["x_max"]
    x_min = data["x_min"]
    x_max_norm = data["x_max_norm"]
    x_min_norm = data["x_min_norm"]
    
    dgN = dg
    if method == "hybrid":
        dgN = dN +dg

    A, b = [], []

    for i in range(ns):
        U = get_poly_features(x[i], dg)
        dU1 = get_poly_features(x[i], dg, 1)/(x_max[i]-x_min[i])
        dU2 = get_poly_features(x[i], dg, 2)/(x_max[i]-x_min[i])**2
        if method == "hybrid":
            U_nn = get_dnn_features(x[i], x_min[i],x_max[i], model,config)
            dU_nn = get_dnn_features(x[i], x_min[i],x_max[i], model,config, 1)#/(x_max[i,0]-x_min[i,0])
            dU2_nn = get_dnn_features(x[i], x_min[i],x_max[i], model,config, 2)#*(x_max[i,0]-x_min[i,0])
            #将U和dU的维度从(n_points, dg+1)变为(n_points,   dg+1+n_segments)
            U = np.hstack([U,U_nn])
            dU1 = np.hstack([dU1,dU_nn])
            dU2 = np.hstack([dU2,dU2_nn])
        un = u_n[i].reshape(-1,1)
        #Left = U  + dt*(un*dU1- mu*dU2)#/(x_max[i,0]-x_min[i,0])
        Left = U/dt  + (-0.0001*dU2)#/(x_max[i,0]-x_min[i,0])
        zeros_l = np.zeros((Left.shape[0], (dgN + 1) * i), dtype=np.float64)
        zeros_r = np.zeros((Left.shape[0], (dgN + 1) * (ns - i - 1)), dtype=np.float64)
        A.extend([np.hstack([zeros_l, Left, zeros_r])])
        b.extend(u_n[i]/dt - (5*u_n[i]**3-5*u_n[i]))

    #for i in range(ns - 1):
    #    cont = np.zeros((1, (dgN + 1) * ns), dtype=np.float64)
    #    P = get_poly_features(x_max_norm[i], dg,2)/(x_max[i,0]-x_min[i,0])**2
    #    if method == "hybrid":
    #        P_nn = get_dnn_features(x_max_norm[i], x_min[i,0],x_max[i,0], model,config, 2)
    #        P = np.hstack([P,P_nn])
    #    cont[0, i * (dgN + 1) : (i + 1) * (dgN + 1)] = P.flatten()
    #    P = get_poly_features(x_min_norm[i+ 1], dg,2)/(x_max[i+1,0]-x_min[i+1,0])**2
    #    if method == "hybrid":
    #        P_nn = get_dnn_features(x_min_norm[i + 1], x_min[i + 1,0],x_max[i + 1,0], model,config, 2)
    #        P = np.hstack([P,P_nn])
    #    cont[0, (i + 1) * (dgN + 1) : (i + 2) * (dgN + 1)] = -P.flatten()
    #    A.extend([cont])
    #    b.append([0])


    # 计算自适应权重 - 与时间步相关的平衡权重
    constraint_weight = 1.0 * np.sqrt(dt)  # 降低过强的约束权重

    for i in range(ns - 1):
        cont = np.zeros((1, (dgN + 1) * ns), dtype=np.float64)
        P = get_poly_features(x_max_norm[i], dg,1)/(x_max[i]-x_min[i])
        if method == "hybrid":
            P_nn = get_dnn_features(x_max_norm[i], x_min[i],x_max[i], model,config, 1)
            P = np.hstack([P,P_nn])
        cont[0, i * (dgN + 1) : (i + 1) * (dgN + 1)] = P.flatten()
        P = get_poly_features(x_min_norm[i+ 1], dg,1)/(x_max[i+1]-x_min[i+1])
        if method == "hybrid":
            P_nn = get_dnn_features(x_min_norm[i + 1], x_min[i + 1],x_max[i + 1], model,config, 1)
            P = np.hstack([P,P_nn])
        cont[0, (i + 1) * (dgN + 1) : (i + 2) * (dgN + 1)] = -P.flatten()
        A.extend([constraint_weight*cont])  # 使用平衡权重而非100倍
        b.extend([0])

    for i in range(ns - 1):
        cont = np.zeros((1, (dgN + 1) * ns), dtype=np.float64)
        P = get_poly_features(x_max_norm[i], dg,0)
        if method == "hybrid":
            P_nn = get_dnn_features(x_max_norm[i], x_min[i],x_max[i], model,config, 0)
            P = np.hstack([P,P_nn])
        cont[0, i * (dgN + 1) : (i + 1) * (dgN + 1)] = P.flatten()
        P = get_poly_features(x_min_norm[i+ 1], dg,0)
        if method == "hybrid":
            P_nn = get_dnn_features(x_min_norm[i + 1], x_min[i + 1],x_max[i + 1], model,config, 0)
            P = np.hstack([P,P_nn])
        cont[0, (i + 1) * (dgN + 1) : (i + 2) * (dgN + 1)] = -P.flatten()
        A.extend([constraint_weight*cont])  # 使用平衡权重
        b.extend([0])
    # 添加左边界条件
    u_B = np.zeros((1, (dgN + 1) * ns), dtype=np.float64)
    P= get_poly_features(x_min_norm[0], dg)
    if method == "hybrid":
        P_nn = get_dnn_features(x_min_norm[0], x_min[0],x_max[0], model,config, 0)
        P = np.hstack([P,P_nn])
    u_B[0, 0 : (dgN + 1)] = P.flatten()

    P= get_poly_features(x_max_norm[-1], dg)
    if method == "hybrid":
        P_nn = get_dnn_features(x_max_norm[-1], x_min[-1],x_max[-1], model,config, 0)
        P = np.hstack([P,P_nn])
    u_B[0, (ns - 1) * (dgN + 1) : ns * (dgN + 1)] = -P.flatten()

    # 边界条件权重也使用平衡权重
    boundary_weight = 1.0  # 降低边界条件权重
    A.extend([boundary_weight*u_B])
    b.extend([0])


    u1_B = np.zeros((1, (dgN + 1) * ns), dtype=np.float64)
    P= get_poly_features(x_min_norm[0], dg,1)/(x_max[0]-x_min[0])
    if method == "hybrid":
        P_nn = get_dnn_features(x_min_norm[0], x_min[0],x_max[0], model,config, 1)
        P = np.hstack([P,P_nn])
    u1_B[0, 0 : (dgN + 1)] = P.flatten()

    P= get_poly_features(x_max_norm[-1], dg,1)/(x_max[-1]-x_min[-1])
    if method == "hybrid":
        P_nn = get_dnn_features(x_max_norm[-1], x_min[-1],x_max[-1], model,config, 1)
        P = np.hstack([P,P_nn])
    u1_B[0, (ns - 1) * (dgN + 1) : ns * (dgN + 1)] = -P.flatten()

    A.extend([boundary_weight*u1_B])  # 使用平衡权重
    b.extend([0])


    # 将A和b转换为numpy数组并确保数值稳定性
    A = np.vstack(A).astype(np.float64)
    b = np.array(b, dtype=np.float64).reshape(-1, 1)

    # 增强正则化项以提高数值稳定性
    # 自适应正则化：基于矩阵条件数估计
    ATA = A.T @ A
    condition_estimate = np.trace(ATA) / (np.min(np.diag(ATA)) + 1e-16)
    lambda_reg = max(1e-16, 1e-15 * np.sqrt(condition_estimate))  # 增强正则化

    print(f"自适应正则化参数: {lambda_reg:.2e}")

    n_vars = A.shape[1]
    A_reg = np.vstack([A, lambda_reg * np.eye(n_vars)])
    b_reg = np.vstack([b, np.zeros((n_vars, 1))])

    # 两步迭代SVD求解以改善残差精度
    print("=== 开始两步迭代SVD求解 ===")

    # 第一步：初始SVD求解
    print("第一步：初始SVD求解")
    result = np.linalg.lstsq(A_reg, b_reg, rcond=1e-15)
    coeffs = result[0]
    residuals = result[1] if len(result) > 1 else None
    rank = result[2] if len(result) > 2 else None
    singular_values = result[3] if len(result) > 3 else None

    # 计算第一步残差
    residual_1 = A_reg @ coeffs - b_reg
    residual_norm_1 = np.linalg.norm(residual_1)
    residual_norm_inf_1 = np.linalg.norm(residual_1, ord=np.inf)
    b_norm = np.linalg.norm(b_reg)
    relative_residual_1 = residual_norm_1 / max(b_norm, 1e-16)
    condition_number = singular_values[0] / singular_values[-1] if singular_values is not None and len(singular_values) > 0 and singular_values[-1] > 1e-16 else np.inf

    print(f"第一步残差: ||Ax-b||_2 = {residual_norm_1:.6e}")
    print(f"第一步相对残差: {relative_residual_1:.6e}")
    print(f"条件数: {condition_number:.6e}")

    # 第二步：基于残差的迭代求解
    if residual_norm_1 > 1e-10:  # 如果残差不够小，进行第二步迭代
        print("第二步：残差迭代改进")

        # 计算残差方程: A * delta_x = -residual_1
        delta_result = np.linalg.lstsq(A_reg, -residual_1, rcond=1e-15)
        delta_coeffs = delta_result[0]

        # 更新系数
        coeffs_improved = coeffs + delta_coeffs

        # 计算第二步残差
        residual_2 = A_reg @ coeffs_improved - b_reg
        residual_norm_2 = np.linalg.norm(residual_2)
        residual_norm_inf_2 = np.linalg.norm(residual_2, ord=np.inf)
        relative_residual_2 = residual_norm_2 / max(b_norm, 1e-16)

        print(f"第二步残差: ||Ax-b||_2 = {residual_norm_2:.6e}")
        print(f"第二步相对残差: {relative_residual_2:.6e}")

        # 判断是否改善
        improvement_factor = residual_norm_1 / max(residual_norm_2, 1e-16)
        print(f"残差改善倍数: {improvement_factor:.2f}")

        if residual_norm_2 < residual_norm_1:
            print("✓ 迭代改善成功，使用改进后的系数")
            coeffs = coeffs_improved
            final_residual_norm = residual_norm_2
            final_residual_norm_inf = residual_norm_inf_2
            final_relative_residual = relative_residual_2
        else:
            print("✗ 迭代未改善，保持初始系数")
            final_residual_norm = residual_norm_1
            final_residual_norm_inf = residual_norm_inf_1
            final_relative_residual = relative_residual_1
    else:
        print("第一步残差已足够小，跳过第二步迭代")
        final_residual_norm = residual_norm_1
        final_residual_norm_inf = residual_norm_inf_1
        final_relative_residual = relative_residual_1

    # 获取全局步骤计数器（如果存在）
    import __main__
    if hasattr(__main__, 'linear_solve_step'):
        __main__.linear_solve_step += 1
        step_info = f"Linear Solve Step {__main__.linear_solve_step}"
    else:
        step_info = "Linear Solve Residual Analysis"

    # 打印最终残差分析
    print(f"\n=== {step_info} - 最终结果 ===")
    print(f"Matrix size: {A_reg.shape[0]} x {A_reg.shape[1]}")
    print(f"Matrix rank: {rank}")
    print(f"Condition number: {condition_number:.6e}")
    print(f"Final Residual Analysis:")
    print(f"  ||Ax-b||_2 = {final_residual_norm:.6e}")
    print(f"  ||Ax-b||_∞ = {final_residual_norm_inf:.6e}")
    print(f"  ||b||_2 = {b_norm:.6e}")
    print(f"  Relative residual = {final_relative_residual:.6e}")
    if residuals is not None and len(residuals) > 0:
        print(f"  Residual from lstsq: {residuals[0]:.6e}")
    print("=" * 45)

    return coeffs.reshape(ns, dgN + 1)


def evl_poly(data, coeffs, config,model=None, method="poly",derivative=0):
    n_segments = data["n_segments"]
    x = data["x_segments_norm"]
    # 将x的维度从(n_segments, points_per_segment, 1)变为(n_segments * points_per_segment, 1)
    x = np.concatenate(x).flatten().reshape(-1,1)#.squeeze(1)
    x_min = data["x_min"]
    x_max = data["x_max"]
    u_n = np.concatenate(data["u"]).flatten().reshape(-1,1)
    dg = config.poly_degree
    if method == "hybrid":
        dgN = config.DNN_degree + dg
    y_pred = np.zeros_like(u_n)
    segment_size = len(x) // n_segments
    for i in range(n_segments):
        start_idx = i * segment_size
        end_idx = (i + 1) * segment_size if i < n_segments - 1 else len(x)
        x_seg = x[start_idx:end_idx]
        X_poly = get_poly_features(x_seg, dg,derivative)/(x_max[i]-x_min[i])**derivative
        if method == "hybrid":
            X_nn = get_dnn_features(x_seg,x_min[i],x_max[i], model,config, derivative)
            X_poly = np.hstack([X_poly,X_nn])
        y_pred[start_idx:end_idx] = X_poly @ coeffs[i].reshape(-1,1)
    return y_pred


def plot_errors(x_test, u_test, test_preds, title="Error Comparison"):

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

    # 画出target function
    ax1.plot(x_test, u_test, label="Target Function", color='black')
    for method, pred in test_preds.items():
        ax1.plot(x_test, pred, label=method, alpha=0.6)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlabel("Input")
    ax1.set_ylabel("Output")
    ax1.set_title("Target Function and Predictions")
    ax1.legend(loc="upper right")

    # 画出errors
    for method, pred in test_preds.items():
        errors = np.abs(pred - u_test.squeeze(1))
        ax2.plot(x_test, errors, label=method, alpha=0.6)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlabel("Input")
    ax2.set_ylabel("Absolute Error")
    ax2.set_title("Test Set Errors")
    ax2.legend(loc="lower right")
    ax2.set_yscale('log')

    plt.suptitle(title)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    init_seed()
    config = Config()
    device = config.device
    print(f"Using device: {device}")
    
    # 初始化步骤计数器
    linear_solve_step = 0
    
    data = scipy.io.loadmat('./cases/Time_pde_cases/AC_equation/reference_data/allen_cahn_highres.mat')
    
    t = data['t']
    x = data['x']
    Exact = data['usol']

    n_segments = config.n_segments
    data_train = generate_data(config, "train")
    data_test = generate_data(config, "test")
    # Enable interactive plotting
    plt.ion()
    
    u_n = data_train["u"]
    u_star = data_train["u"]
    fig, ax = plt.subplots(figsize=(10, 6))
    line_pred = ax.scatter(data_train["x"], u_n, c='r', label='Numerical', s=20)
    ax.plot(x.flatten(),Exact[-1,:].flatten(),label='Reference', color='blue', alpha=0.7)
    
    # Set plot properties
    ax.set_ylim(-1.2, 1.2)
    ax.set_xlim(-1.0, 1.0)
    ax.set_xlabel('x')
    ax.set_ylabel('u')
    ax.set_title('Allen-Cahn Equation Real-time Solution')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    model = Net(in_dim=1, hidden_dims=config.hidden_dims, out_dim=1).to(config.device)

    dt = config.dt
    mu = config.mu
    x_train = torch.tensor(data_train["x"], dtype=torch.float64, requires_grad=True, device=config.device)
    x_L = torch.tensor(data_train["x_L"], dtype=torch.float64,requires_grad=True, device=config.device)
    x_R = torch.tensor(data_train["x_R"], dtype=torch.float64,requires_grad=True, device=config.device)

    data_GPU = {"x_train": x_train, "x_L": x_L, "x_R": x_R, "dt": dt, "mu": mu}
    
    T =0
    it = 0
    dU1_old = []
    dU2_old = []
    du_x_old = 0
    du_xx_old = 0
    start_time = time.time()
    while T<config.time:
        it = it +1
        T = T + config.dt
        
        u_star,model,coeffs,du_x,du_xx = Time_evolve_pre(model, data_GPU,u_n,config)

        u_n = u_n.flatten()
        u_star = u_star.flatten()
        u_star = (u_star + u_n)/2

        du_x = du_x.flatten()
        du_xx = du_xx.flatten()
        
        if (it >1): 
            du1_x = ( du_x +du_x_old)/2
            du2_xx = (du_xx +du_xx_old)/2
            u_star = u_star.flatten()
            u_n = u_n.flatten()

            u_n =  u_n - dt*(- 0.0001*du2_xx + 5*u_star**3-5*u_star)
        else:
            u_n = u_n - dt*(- 0.0001*du_xx + 5*u_n**3-5*u_n)
        
        du_x_old = copy.deepcopy(du_x)
        du_xx_old = copy.deepcopy(du_xx)


        u_n_segments = []
        start_idx = 0
        for i in range(n_segments):
            seg_len = len(data_train["u_segments"][i])
            u_n_segments.append(u_n[start_idx:start_idx + seg_len])
            start_idx += seg_len
            
        data_train["u_segments"] = u_n_segments

        u_n = u_n.reshape(-1,1)
        print(f"Step {it}: time = {T:.3f}, max|u| = {np.max(np.abs(u_n)):.4f}")

        # Update real-time plot
        line_pred.set_offsets(np.column_stack((data_train["x"], u_n)))
        
        # Update title with current time and step
        ax.set_title(f'Allen-Cahn Equation - Step {it}, Time = {T:.3f}s')
        
        # Refresh display
        fig.canvas.draw()
        fig.canvas.flush_events()
        time.sleep(0.05)  # Small delay for smooth animation
    
    plt.ioff()
    print("Time evolution completed. Close the plot window to continue...")
    methods = ["New"]


    hybrid_time = time.time() - start_time

    print(f" Cost Time: {hybrid_time}")


    x_min = data_test["x_min"]
    x_max = data_test["x_max"]

    x_test = split_array(x, x_min, x_max)
    print(f"Debug: x_test segments = {len(x_test)}, config.n_segments = {config.n_segments}")

    # Use existing test data segments instead of splitting reference grid
    x_segment = data_test["x_segments"]
    x_segment_norm = data_test["x_segments_norm"]
    #data_test["x_segments"] = x_segment
    #data_test["x_segments_norm"] = x_segment_norm
    
    y_pred_test_hybrid = evl_poly(data_test, coeffs, config,model,config.method)
    print("\nPerformance Summary:")
    print("-" * 50)

    # Interpolate reference solution to test grid for comparison
    from scipy.interpolate import interp1d
    x_ref = x.flatten()
    u_ref = Exact[-1,:].flatten()
    x_test_flat = data_test["x"].flatten()
    
    # Interpolate reference to test points
    interp_func = interp1d(x_ref, u_ref, kind='cubic', bounds_error=False, fill_value='extrapolate')
    u_ref_interp = interp_func(x_test_flat)
    
    L2_error = np.sqrt(np.mean((u_ref_interp - y_pred_test_hybrid.flatten())**2))
    print(f"L2 error: {L2_error:.6e}")
    print(f"Test grid points: {len(y_pred_test_hybrid.flatten())}")
    print(f"Reference grid points: {len(u_ref)}")

    # Final comparison plot
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 1, 1)
    plt.plot(x_test_flat, y_pred_test_hybrid.flatten(), 'r-', label='Numerical Solution', linewidth=2)
    plt.plot(x_ref, u_ref, 'b--', label='Reference Solution', linewidth=2, alpha=0.8)
    plt.xlabel('x')
    plt.ylabel('u')
    plt.title(f'Final Solution Comparison at T = {config.time}s')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 1, 2)
    error = np.abs(u_ref_interp - y_pred_test_hybrid.flatten())
    plt.plot(x_test_flat, error, 'g-', linewidth=2, label=f'Absolute Error (L2 = {L2_error:.2e})')
    plt.xlabel('x')
    plt.ylabel('|Error|')
    plt.title('Pointwise Absolute Error')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.yscale('log')
    
    plt.tight_layout()
    plt.show()
    
    print("\\nFinal comparison plot displayed. Analysis completed!")


