import numpy as np
import torch
from concurrent.futures import ThreadPoolExecutor
from typing import Optional, Union, Tuple, Dict, Callable
import scipy.sparse as sp
from scipy.sparse import linalg as splinalg
from scipy import linalg
import time
import warnings
import os
import multiprocessing
import psutil

# 尝试导入CuPy，如果不可用则提供警告
try:
    import cupy as cp
    from cupyx.scipy.sparse import linalg as cplinalg
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False
    warnings.warn("CuPy未安装，GPU加速功能将不可用。使用 pip install cupy-cuda11x 安装（根据CUDA版本选择）")


class LinearSolver:
    """过定线性系统求解器，用于解决Ax=b问题（A的行数大于列数），支持CuPy GPU加速和CPU并行计算"""

    def __init__(self, verbose: bool = False, use_gpu: bool = True, 
                 n_jobs: int = None, chunk_size: int = 1000, 
                 gpu_memory_fraction: float = 0.9, 
                 gpu_batch_size: int = None,
                 use_mixed_precision: bool = False,
                 performance_tracking: bool = False,
                 config: object = None):
        """初始化求解器
        
        Args:
            verbose: 是否打印详细信息
            use_gpu: 是否使用GPU（如果可用）, True表示默认使用GPU
            n_jobs: 并行CPU核心数，None表示使用全部可用核心
            chunk_size: 矩阵分块大小，用于并行计算
            gpu_memory_fraction: GPU内存使用比例，防止占用全部显存
            gpu_batch_size: GPU批处理大小，None表示自动确定
            use_mixed_precision: 是否使用混合精度（对某些操作使用float16/32）
            performance_tracking: 是否跟踪性能指标
            config: 配置对象，用于覆盖默认设置
        """
        self.verbose = verbose
        self.performance_tracking = performance_tracking
        self.gpu_memory_fraction = gpu_memory_fraction
        self.gpu_batch_size = gpu_batch_size
        self.use_mixed_precision = use_mixed_precision
        
        # 检查config.linear_device
        if config is not None and hasattr(config, 'linear_device'):
            if getattr(config, 'linear_device', '').lower() == 'cpu':
                use_gpu = False
        
        # 检查实际CUDA设备
        if use_gpu and CUPY_AVAILABLE:
            try:
                n_devices = cp.cuda.runtime.getDeviceCount()
                if n_devices == 0:
                    warnings.warn("未检测到CUDA设备，将回退到CPU计算")
                    use_gpu = False
            except Exception:
                warnings.warn("检测CUDA设备时出错，将回退到CPU计算")
                use_gpu = False
        
        # 设置CPU并行参数
        self.n_jobs = n_jobs if n_jobs is not None else max(1, os.cpu_count() - 1)  # 保留一个核心给系统
        self.chunk_size = chunk_size
        
        # 检测是否可以使用GPU
        self.gpu_available = CUPY_AVAILABLE
        
        # 如果设置使用GPU但GPU不可用，显示警告并回退到CPU
        if use_gpu and not self.gpu_available:
            warnings.warn("请求使用GPU但CuPy不可用，将回退到CPU计算")
            self.use_gpu = False
        else:
            self.use_gpu = use_gpu and self.gpu_available
            
        if self.verbose:
            print(f"计算后端: {'GPU (CuPy)' if self.use_gpu else f'CPU ({self.n_jobs}核心)'}")
            if self.use_mixed_precision:
                print("使用混合精度计算")
            
        # 方法注册
        self.methods = {
            "direct": self._solve_direct,
            "svd": self._solve_svd,
            "qr": self._solve_qr,
            "ridge": self._solve_ridge,
            "lsqr": self._solve_lsqr,
            "lsmr": self._solve_lsmr,
            "parallel_direct": self._solve_parallel_direct,
            "parallel_svd": self._solve_parallel_svd,
            "parallel_ridge": self._solve_parallel_ridge
        }
        
        # GPU特定方法
        if self.use_gpu:
            self.methods.update({
                "gpu_direct": self._solve_gpu_direct,
                "gpu_svd": self._solve_gpu_svd,
                "gpu_qr": self._solve_gpu_qr,
                "gpu_lstsq": self._solve_gpu_lstsq,
                "gpu_batched_lstsq": self._solve_gpu_batched_lstsq
            })
            
        # 创建线程池
        self.executor = ThreadPoolExecutor(max_workers=self.n_jobs)
        
        # 配置GPU内存使用
        if self.use_gpu:
            try:
                # 限制内存使用比例
                mempool = cp.get_default_memory_pool()
                mempool.set_limit(fraction=self.gpu_memory_fraction)
                if self.verbose:
                    print(f"GPU内存使用比例设置为: {self.gpu_memory_fraction:.2f}")
                    free_mem = self._get_available_gpu_memory()
                    if free_mem:
                        print(f"可用GPU内存: {free_mem:.2f} GB")
            except Exception as e:
                warnings.warn(f"设置GPU内存限制失败: {str(e)}")
        
        # 性能跟踪
        self.performance_history = [] if performance_tracking else None
    
    def _check_gpu_capability(self) -> bool:
        """检查GPU是否适合计算（性能、内存等）"""
        if not CUPY_AVAILABLE:
            return False
            
        try:
            # 检查GPU内存
            free_memory_gb = self._get_available_gpu_memory()
            if free_memory_gb is None or free_memory_gb < 1.0:  # 至少需要1GB可用内存
                if self.verbose:
                    print("GPU内存不足，切换到CPU模式")
                return False
                
            # 简单性能测试
            test_size = 1000
            test_a = np.random.rand(test_size, test_size // 2).astype(np.float32)
            test_b = np.random.rand(test_size, 1).astype(np.float32)
            
            # CPU计时
            start = time.time()
            np.linalg.lstsq(test_a, test_b, rcond=None)
            cpu_time = time.time() - start
            
            # GPU计时
            a_gpu = cp.asarray(test_a)
            b_gpu = cp.asarray(test_b)
            start = time.time()
            cp.linalg.lstsq(a_gpu, b_gpu, rcond=None)
            gpu_time = time.time() - start
            
            # 释放测试内存
            del a_gpu, b_gpu
            cp.get_default_memory_pool().free_all_blocks()
            
            # 如果GPU比CPU快，使用GPU
            use_gpu = gpu_time < cpu_time
            if self.verbose:
                print(f"性能测试: CPU={cpu_time:.4f}秒, GPU={gpu_time:.4f}秒")
                print(f"自动选择: {'GPU' if use_gpu else 'CPU'}")
            
            return use_gpu
            
        except Exception as e:
            if self.verbose:
                print(f"GPU检测失败: {str(e)}，切换到CPU模式")
            return False

    def solve(self, 
              A: Union[np.ndarray, sp.spmatrix], 
              b: np.ndarray,
              method: str = "gpu_auto", 
              fallback: bool = True,
              **kwargs) -> Tuple[np.ndarray, Dict]:
        """求解线性系统Ax=b
        
        Args:
            A: 系统矩阵，形状为(m,n)，m>n（行数大于列数）
            b: 右侧向量，形状为(m,)或(m,1)
            method: 求解方法，默认为"gpu_auto"优先选择GPU方法
            fallback: 如果选定方法失败，是否尝试回退到更安全的方法
            **kwargs: 特定方法的额外参数
        
        Returns:
            Tuple[np.ndarray, Dict]: 解向量和求解信息字典
        """
        # 检查输入
        m, n = A.shape
        if m < n:
            raise ValueError(f"输入矩阵不是过定系统：行数({m}) < 列数({n})")
            
        # 确保b是正确的形状
        if b.ndim == 1:
            b = b.reshape(-1, 1)
        
        # 自动选择方法
        original_method = method
        if method == "auto" or method == "gpu_auto":
            method = self._auto_select_method(A)
            # 如果请求为gpu_auto且有GPU，但选择了CPU方法，则尝试找替代GPU方法
            if method.startswith("gpu_") == False and self.use_gpu and original_method == "gpu_auto":
                # 尝试选择对应的GPU方法
                if method == "direct":
                    method = "gpu_direct"
                elif method == "svd":
                    method = "gpu_svd"
                elif method == "qr":
                    method = "gpu_qr"
                else:
                    # 默认使用gpu_direct
                    method = "gpu_direct"
                    
            if self.verbose:
                print(f"自动选择方法: {method}")
        
        # 检查方法是否存在
        if method not in self.methods:
            raise ValueError(f"未知的求解方法: {method}。可用方法: {list(self.methods.keys())}")
        
        # 计时开始
        start_time = time.time()
        
        try:
            # 调用对应的方法
            x, info = self.methods[method](A, b, **kwargs)
        except Exception as e:
            if fallback and method != "direct":
                # 出现异常，尝试回退到更安全的方法
                # 如果GPU方法失败，首先尝试其他GPU方法，然后再回退到CPU
                if method.startswith("gpu_") and self.use_gpu:
                    fallback_method = "gpu_batched_lstsq"
                    warnings.warn(f"方法 {method} 失败: {str(e)}，尝试回退到GPU批处理方法")
                else:
                    fallback_method = "direct"
                    warnings.warn(f"方法 {method} 失败: {str(e)}，尝试回退到直接法")
                    
                return self.solve(A, b, method=fallback_method, fallback=False, **kwargs)
            else:
                raise RuntimeError(f"求解失败且无法回退: {str(e)}")
        
        # 计算求解时间
        solve_time = time.time() - start_time
        info['solve_time'] = solve_time
        info['method'] = method
        info['original_method'] = original_method
        
        # 计算残差范数（在CPU上）
        if not sp.issparse(A):
            residual_norm = np.linalg.norm(A @ x - b)
        else:
            residual_norm = np.linalg.norm(A.dot(x) - b)
        info['residual_norm'] = residual_norm
        
        if self.verbose:
            print(f"求解方法: {method}")
            print(f"求解时间: {solve_time:.6f}秒")
            print(f"残差范数: {residual_norm:.6e}")
            if 'iterations' in info:
                print(f"迭代次数: {info['iterations']}")
                
        # 记录性能
        if self.performance_tracking:
            self.performance_history.append({
                'method': method,
                'matrix_shape': A.shape,
                'solve_time': solve_time,
                'residual_norm': residual_norm,
                'timestamp': time.time()
            })
        
        return x, info
    
    def _auto_select_method(self, A: Union[np.ndarray, sp.spmatrix]) -> str:
        """自动选择最佳求解方法，基于矩阵大小、条件数和可用计算资源"""
        m, n = A.shape
        is_sparse = sp.issparse(A)
        
        # 对于极大规模问题或稀疏矩阵，自动选择迭代解法
        if m > 1e6 or n > 1e5 or (is_sparse and A.nnz / (m*n) < 0.05):
            return "lsmr"
        
        # 估计问题规模和内存需求
        problem_size = m * n
        element_size = 4 if self.use_mixed_precision else 8  # float32或float64
        memory_needed_gb = problem_size * element_size * 3 / (1024**3)  # 额外系数是为安全裕度
        
        # 如果GPU可用，优先考虑GPU方法
        if self.use_gpu and not is_sparse:
            # 检查可用GPU内存
            gpu_mem_gb = self._get_available_gpu_memory()
            
            if gpu_mem_gb:
                # 根据GPU内存选择方法
                if memory_needed_gb > gpu_mem_gb * 0.8:
                    # GPU内存接近极限，使用批处理模式
                    if self.verbose:
                        print(f"矩阵需要{memory_needed_gb:.2f}GB，GPU内存{gpu_mem_gb:.2f}GB，切换到批处理")
                    
                    if memory_needed_gb > gpu_mem_gb * 0.95:
                        # 内存需求几乎等于或超过GPU内存，回退到CPU并行
                        if problem_size > 5e6 and self.n_jobs > 1:
                            return "parallel_direct"
                        else:
                            return "direct"
                    else:
                        # 使用批处理模式
                        return "gpu_batched_lstsq"
                else:
                    # GPU内存充足，根据问题特性选择GPU方法
                    if problem_size > 1e7:
                        return "gpu_lstsq"
                    elif self._is_ill_conditioned(A):
                        return "gpu_svd"
                    else:
                        return "gpu_direct"
            return "gpu_direct"  # 默认情况下尝试直接GPU方法
                
        # 检查CPU内存（简化估计）
        try:
            cpu_memory_gb = psutil.virtual_memory().available / (1024**3)
        except:
            cpu_memory_gb = 8.0  # 保守估计
            
        # 如果CPU内存不足，尝试迭代解法
        if memory_needed_gb > cpu_memory_gb * 0.8 and not is_sparse:
            if self.verbose:
                print(f"内存需求({memory_needed_gb:.2f}GB)超过可用CPU内存的80%，使用迭代解法")
            return "lsmr"
        
        # CPU方法选择
        # 对于大矩阵使用并行方法
        if not is_sparse:
            if problem_size > 5e6 and self.n_jobs > 1:
                if self._is_ill_conditioned(A):
                    return "parallel_svd" if problem_size < 1e8 else "parallel_ridge"
                else:
                    return "parallel_direct"
            elif problem_size < 1e6:
                if self._is_ill_conditioned(A):
                    return "svd"
                else:
                    return "qr" if m > 3*n else "direct"
        
        # 对于稀疏矩阵
        if is_sparse:
            density = A.nnz / (m * n)
            if density < 0.01:
                return "lsqr"
            else:
                return "lsmr"
                
        # 默认情况
        return "direct"
    
    def _is_ill_conditioned(self, A: np.ndarray, sample_size: int = 1000, threshold: float = 1e6) -> bool:
        """估计矩阵是否病态（可能有较大的条件数）"""
        if sp.issparse(A):
            return False  # 稀疏矩阵暂不估计条件数
            
        m, n = A.shape
        
        # 对于小矩阵，直接计算条件数
        if m < 5000 and n < 1000:
            try:
                return np.linalg.cond(A) > threshold
            except:
                return False
        
        # 对于大矩阵，采样估计
        if m > sample_size:
            # 随机采样行
            indices = np.random.choice(m, sample_size, replace=False)
            A_sample = A[indices]
            try:
                return np.linalg.cond(A_sample) > threshold / 10  # 降低阈值，因为采样可能低估条件数
            except:
                return False
                
        return False
    
    def _solve_direct(self, A: Union[np.ndarray, sp.spmatrix], b: np.ndarray, **kwargs) -> Tuple[np.ndarray, Dict]:
        """直接最小二乘法，求解(A^T A)x = A^T b"""
        if sp.issparse(A):
            x, istop, itn, normr = splinalg.lsqr(A, b.flatten(), **kwargs)[:4]
            info = {'iterations': itn, 'status': istop}
            return x.reshape(-1, 1), info
        else:
            x, residuals, rank, s = linalg.lstsq(A, b, **kwargs)
            info = {'rank': rank, 'singular_values': s}
            return x, info
    
    def _solve_svd(self, A: np.ndarray, b: np.ndarray, 
                   rcond: float = None, **kwargs) -> Tuple[np.ndarray, Dict]:
        """使用SVD分解求解，可以处理病态问题"""
        # 执行SVD分解
        U, s, Vh = linalg.svd(A, full_matrices=False)
        
        # 计算伪逆
        rcond = rcond or np.finfo(float).eps * max(A.shape)
        s_mask = s > rcond * s[0]
        s_inv = np.zeros_like(s)
        s_inv[s_mask] = 1/s[s_mask]
        
        # 计算解
        x = Vh.T @ (s_inv.reshape(-1, 1) * (U.T @ b))
        
        # 计算有效秩和条件数
        rank = np.sum(s_mask)
        condition_number = s[0] / s[s_mask][-1] if rank > 0 else np.inf
        
        info = {
            'rank': rank,
            'singular_values': s,
            'condition_number': condition_number
        }
        
        return x, info
    
    def _solve_qr(self, A: np.ndarray, b: np.ndarray, **kwargs) -> Tuple[np.ndarray, Dict]:
        """使用QR分解求解，比SVD更高效，但对病态问题处理不如SVD"""
        # 执行QR分解
        Q, R = linalg.qr(A, mode='economic')
        
        # 计算解
        x = linalg.solve_triangular(R, Q.T @ b)
        
        # 计算条件数
        condition_number = np.linalg.cond(R)
        
        info = {
            'condition_number': condition_number
        }
        
        return x, info
    
    def _solve_ridge(self, A: Union[np.ndarray, sp.spmatrix], b: np.ndarray, 
                    alpha: float = 1.0, **kwargs) -> Tuple[np.ndarray, Dict]:
        """岭回归/Tikhonov正则化，适用于病态问题"""
        m, n = A.shape
        
        if sp.issparse(A):
            # 对于稀疏矩阵，使用迭代方法求解正则化最小二乘问题
            # 构建增广矩阵 [A; sqrt(alpha)*I]
            I = sp.eye(n)
            A_aug = sp.vstack([A, np.sqrt(alpha) * I])
            b_aug = np.vstack([b, np.zeros((n, 1))])
            
            x, istop, itn, normr = splinalg.lsqr(A_aug, b_aug.flatten())[:4]
            info = {'iterations': itn, 'status': istop, 'alpha': alpha}
            return x.reshape(-1, 1), info
        else:
            # 对于密集矩阵，使用正规方程求解
            ATA = A.T @ A
            ATb = A.T @ b
            
            # 添加正则化项
            ATA_reg = ATA + alpha * np.eye(n)
            
            # 求解正则化系统
            x = linalg.solve(ATA_reg, ATb, assume_a='pos')
            
            # 计算条件数
            condition_number = np.linalg.cond(ATA_reg)
            
            info = {
                'alpha': alpha,
                'condition_number': condition_number
            }
            
            return x, info
    
    def _solve_lsqr(self, A: Union[np.ndarray, sp.spmatrix], b: np.ndarray, 
                   tol: float = 1e-8, iter_lim: int = None, **kwargs) -> Tuple[np.ndarray, Dict]:
        """使用LSQR迭代算法求解，适用于大型稀疏系统"""
        iter_lim = iter_lim or max(A.shape) * 10
        
        # 确保b是一维的
        b_flat = b.flatten()
        
        # 使用LSQR求解
        x, istop, itn, r1norm, r2norm, anorm, acond, arnorm, xnorm, var = splinalg.lsqr(
            A, b_flat, atol=tol, btol=tol, iter_lim=iter_lim, **kwargs
        )
        
        info = {
            'iterations': itn,
            'status': istop,
            'residual_norm_1': r1norm,
            'residual_norm_2': r2norm,
            'matrix_norm': anorm,
            'condition_number_estimate': acond,
            'solution_norm': xnorm
        }
        
        return x.reshape(-1, 1), info
    
    def _solve_lsmr(self, A: Union[np.ndarray, sp.spmatrix], b: np.ndarray, 
                   tol: float = 1e-8, max_iter: int = None, **kwargs) -> Tuple[np.ndarray, Dict]:
        """使用LSMR迭代算法求解，通常比LSQR收敛更快"""
        max_iter = max_iter or max(A.shape) * 10
        
        # 确保b是一维的
        b_flat = b.flatten()
        
        # 使用LSMR求解
        x, istop, itn, normr, normar, norma, conda, normx = splinalg.lsmr(
            A, b_flat, atol=tol, btol=tol, maxiter=max_iter, **kwargs
        )
        
        info = {
            'iterations': itn,
            'status': istop,
            'residual_norm': normr,
            'matrix_norm': norma,
            'condition_number_estimate': conda,
            'solution_norm': normx
        }
        
        return x.reshape(-1, 1), info
    
    def _solve_gpu_batched_lstsq(self, A: np.ndarray, b: np.ndarray, **kwargs) -> Tuple[np.ndarray, Dict]:
        """使用批处理方式在GPU上求解，防止显存溢出"""
        if not self.use_gpu:
            raise RuntimeError("请求GPU求解，但GPU不可用")
            
        m, n = A.shape
        
        # 估计一个合理的批处理大小
        if self.gpu_batch_size is None:
            try:
                free_memory = cp.cuda.runtime.memGetInfo()[0]
                element_size = 4 if self.use_mixed_precision else 8
                max_rows = int(free_memory * 0.5 / (n * element_size))  # 使用50%可用内存
                batch_size = min(max_rows, 10000)  # 不超过10000行
            except:
                batch_size = 5000  # 保守默认值
        else:
            batch_size = self.gpu_batch_size
            
        if self.verbose:
            print(f"使用GPU批处理，批大小: {batch_size}行")
        
        # 计算批次数
        num_batches = (m + batch_size - 1) // batch_size
        
        # 准备结果变量
        if self.use_mixed_precision:
            # 对A和b使用较低精度
            dtype = np.float32
        else:
            dtype = np.float64
            
        # 初始化ATA和ATb（在CPU上）
        ATA = np.zeros((n, n), dtype=dtype)
        ATb = np.zeros((n, b.shape[1]), dtype=dtype)
        
        # 分批处理
        for i in range(num_batches):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, m)
            
            # 提取当前批次
            A_batch = A[start_idx:end_idx].astype(dtype)
            b_batch = b[start_idx:end_idx].astype(dtype)
            
            # 转移到GPU
            A_gpu = cp.asarray(A_batch)
            b_gpu = cp.asarray(b_batch)
            
            # 计算当前批次的贡献
            ATA_batch = cp.asnumpy(A_gpu.T @ A_gpu)
            ATb_batch = cp.asnumpy(A_gpu.T @ b_gpu)
            
            # 累加结果
            ATA += ATA_batch
            ATb += ATb_batch
            
            # 主动释放GPU内存
            del A_gpu, b_gpu
            cp.get_default_memory_pool().free_all_blocks()
        
        # 最终求解（在CPU上完成，以保证精度）
        x = np.linalg.solve(ATA, ATb)
        
        # 计算条件数
        condition_number = np.linalg.cond(ATA)
        
        info = {
            'condition_number': condition_number,
            'batches': num_batches,
            'batch_size': batch_size,
            'mixed_precision': self.use_mixed_precision,
            'backend': 'cupy+numpy'
        }
        
        return x, info
    
    def _solve_gpu_direct(self, A: np.ndarray, b: np.ndarray, **kwargs) -> Tuple[np.ndarray, Dict]:
        """使用CuPy直接求解线性最小二乘问题，增加了内存管理"""
        if not self.use_gpu:
            raise RuntimeError("请求GPU求解，但GPU不可用")
        
        # 检查矩阵大小，决定是否使用float32以节省显存
        m, n = A.shape
        use_float32 = self.use_mixed_precision and m*n > 1e6
        
        # 数据类型转换
        dtype = np.float32 if use_float32 else np.float64
        
        try:    
            # 转移数据到GPU
            A_gpu = cp.asarray(A, dtype=dtype)
            b_gpu = cp.asarray(b, dtype=dtype)
            
            # 使用CuPy的最小二乘求解
            x_gpu = cp.linalg.lstsq(A_gpu, b_gpu, rcond=None)[0]
            
            # 计算条件数
            try:
                svd_s = cp.linalg.svd(A_gpu, compute_uv=False)
                condition_number = float(svd_s[0] / svd_s[-1])
            except:
                # 如果SVD失败，可能是内存问题，跳过条件数计算
                condition_number = None
            
            # 将结果转回CPU
            x = cp.asnumpy(x_gpu)
            
            # 主动释放GPU内存
            del A_gpu, b_gpu, x_gpu
            cp.get_default_memory_pool().free_all_blocks()
            
        except cp.cuda.memory.OutOfMemoryError:
            if self.verbose:
                print("GPU内存不足，尝试使用批处理方法")
            return self._solve_gpu_batched_lstsq(A, b, **kwargs)
        
        info = {
            'condition_number': condition_number,
            'precision': '32bit' if use_float32 else '64bit',
            'backend': 'cupy'
        }
        
        return x, info
    
    def _solve_gpu_svd(self, A: np.ndarray, b: np.ndarray, 
                       rcond: float = None, **kwargs) -> Tuple[np.ndarray, Dict]:
        """使用CuPy的SVD分解求解"""
        if not self.use_gpu:
            raise RuntimeError("请求GPU求解，但GPU不可用")
            
        # 转移数据到GPU
        A_gpu = cp.asarray(A)
        b_gpu = cp.asarray(b)
        
        # 执行SVD分解
        U, s, Vh = cp.linalg.svd(A_gpu, full_matrices=False)
        
        # 计算伪逆
        rcond = rcond or float(cp.finfo(cp.float64).eps * max(A_gpu.shape))
        s_mask = s > rcond * s[0]
        s_inv = cp.zeros_like(s)
        s_inv[s_mask] = 1/s[s_mask]
        
        # 计算解
        x_gpu = Vh.T @ (s_inv.reshape(-1, 1) * (U.T @ b_gpu))
        
        # 计算有效秩和条件数
        rank = int(cp.sum(s_mask))
        condition_number = float(s[0] / s[s_mask][-1]) if rank > 0 else float('inf')
        
        # 将结果转回CPU
        x = cp.asnumpy(x_gpu)
        s_cpu = cp.asnumpy(s)
        
        info = {
            'rank': rank,
            'singular_values': s_cpu,
            'condition_number': condition_number,
            'backend': 'cupy'
        }
        
        return x, info
    
    def _solve_gpu_qr(self, A: np.ndarray, b: np.ndarray, **kwargs) -> Tuple[np.ndarray, Dict]:
        """使用CuPy的QR分解求解"""
        if not self.use_gpu:
            raise RuntimeError("请求GPU求解，但GPU不可用")
            
        # 转移数据到GPU
        A_gpu = cp.asarray(A)
        b_gpu = cp.asarray(b)
        
        # 执行QR分解
        Q, R = cp.linalg.qr(A_gpu, mode='economic')
        
        # 计算解
        QTb = Q.T @ b_gpu
        x_gpu = cp.linalg.solve_triangular(R, QTb)
        
        # 计算条件数
        condition_number = float(cp.linalg.cond(R))
        
        # 将结果转回CPU
        x = cp.asnumpy(x_gpu)
        
        info = {
            'condition_number': condition_number,
            'backend': 'cupy'
        }
        
        return x, info
    
    def _solve_gpu_lstsq(self, A: np.ndarray, b: np.ndarray, **kwargs) -> Tuple[np.ndarray, Dict]:
        """使用CuPy的封装最小二乘函数求解，针对大型矩阵优化"""
        if not self.use_gpu:
            raise RuntimeError("请求GPU求解，但GPU不可用")
            
        # 转移数据到GPU
        A_gpu = cp.asarray(A)
        b_gpu = cp.asarray(b)
        
        # 使用gels求解最小二乘问题（可能使用cuSOLVER）
        # CuPy目前没有直接暴露gels，所以我们使用lstsq
        x_gpu = cp.linalg.lstsq(A_gpu, b_gpu)[0]
        
        # 将结果转回CPU
        x = cp.asnumpy(x_gpu)
        
        info = {
            'backend': 'cupy'
        }
        
        return x, info
    
    def _process_matrix_chunk(self, func: Callable, A_chunk: np.ndarray, *args, **kwargs) -> np.ndarray:
        """处理矩阵分块的工作函数"""
        return func(A_chunk, *args, **kwargs)
        
    def _solve_parallel_direct(self, A: np.ndarray, b: np.ndarray, **kwargs) -> Tuple[np.ndarray, Dict]:
        """使用线程池并行求解的直接法"""
        if sp.issparse(A):
            # 稀疏矩阵暂不支持并行，回退到普通求解
            return self._solve_direct(A, b, **kwargs)
            
        m, n = A.shape
        
        # 检查问题规模是否值得并行
        if m < self.n_jobs * self.chunk_size:
            # 问题太小，不值得并行
            return self._solve_direct(A, b, **kwargs)
        
        # 计算分块数量
        chunk_size = min(self.chunk_size, m // self.n_jobs)
        n_chunks = min(self.n_jobs, m // chunk_size)
        
        # 将矩阵分块
        A_chunks = np.array_split(A, n_chunks, axis=0)
        b_chunks = np.array_split(b, n_chunks, axis=0)
        
        # 为每个分块计算 A^T*b 和 A^T*A
        futures_ATb = []
        futures_ATA = []
        
        for i in range(n_chunks):
            futures_ATb.append(self.executor.submit(
                lambda A_i, b_i: A_i.T @ b_i, A_chunks[i], b_chunks[i]
            ))
            futures_ATA.append(self.executor.submit(
                lambda A_i: A_i.T @ A_i, A_chunks[i]
            ))
        
        # 合并结果
        ATb = sum(future.result() for future in futures_ATb)
        ATA = sum(future.result() for future in futures_ATA)
        
        # 求解合并后的系统
        x = linalg.solve(ATA, ATb, assume_a='pos')
        
        # 计算条件数
        condition_number = np.linalg.cond(ATA)
        
        info = {
            'condition_number': condition_number,
            'parallel_chunks': n_chunks,
            'parallel_jobs': self.n_jobs
        }
        
        return x, info
    
    def _solve_parallel_svd(self, A: np.ndarray, b: np.ndarray, 
                           rcond: float = None, **kwargs) -> Tuple[np.ndarray, Dict]:
        """使用并行SVD分解求解，通过划分数据并行计算部分结果"""
        if sp.issparse(A):
            # 稀疏矩阵暂不支持并行SVD，回退到普通求解
            return self._solve_svd(A, b, rcond=rcond, **kwargs)
        
        m, n = A.shape
        
        # 对于小矩阵，直接使用标准SVD
        if m < 5000 or n < 1000:
            return self._solve_svd(A, b, rcond=rcond, **kwargs)
        
        # 执行SVD分解
        # 注意：完整SVD很难并行化，但可以并行准备数据或使用分块SVD近似
        # 这里仍使用标准SVD，但未来可进一步优化
        U, s, Vh = linalg.svd(A, full_matrices=False)
        
        # 并行计算 U.T @ b
        chunk_size = min(self.chunk_size, U.shape[0] // self.n_jobs)
        n_chunks = min(self.n_jobs, U.shape[0] // chunk_size)
        
        if n_chunks > 1:
            U_chunks = np.array_split(U, n_chunks, axis=0)
            b_chunks = np.array_split(b, n_chunks, axis=0)
            
            futures = []
            for i in range(n_chunks):
                futures.append(self.executor.submit(
                    lambda U_i, b_i: U_i.T @ b_i, U_chunks[i], b_chunks[i]
                ))
            
            UTb_partial = [future.result() for future in futures]
            UTb = sum(UTb_partial)
        else:
            UTb = U.T @ b
        
        # 计算伪逆
        rcond = rcond or np.finfo(float).eps * max(A.shape)
        s_mask = s > rcond * s[0]
        s_inv = np.zeros_like(s)
        s_inv[s_mask] = 1/s[s_mask]
        
        # 计算解
        x = Vh.T @ (s_inv.reshape(-1, 1) * UTb)
        
        # 计算有效秩和条件数
        rank = np.sum(s_mask)
        condition_number = s[0] / s[s_mask][-1] if rank > 0 else np.inf
        
        info = {
            'rank': rank,
            'singular_values': s,
            'condition_number': condition_number,
            'parallel_chunks': n_chunks,
            'parallel_jobs': self.n_jobs
        }
        
        return x, info
    
    def _solve_parallel_ridge(self, A: np.ndarray, b: np.ndarray, 
                           alpha: float = 1.0, **kwargs) -> Tuple[np.ndarray, Dict]:
        """并行实现的岭回归/Tikhonov正则化"""
        m, n = A.shape
        
        # 使用并行计算ATb和ATA
        chunk_size = min(self.chunk_size, m // self.n_jobs)
        n_chunks = min(self.n_jobs, m // chunk_size)
        
        # 将矩阵分块
        A_chunks = np.array_split(A, n_chunks, axis=0)
        b_chunks = np.array_split(b, n_chunks, axis=0)
        
        # 并行计算ATA和ATb
        futures_ATb = []
        futures_ATA = []
        
        for i in range(n_chunks):
            futures_ATb.append(self.executor.submit(
                lambda A_i, b_i: A_i.T @ b_i, A_chunks[i], b_chunks[i]
            ))
            futures_ATA.append(self.executor.submit(
                lambda A_i: A_i.T @ A_i, A_chunks[i]
            ))
        
        # 合并结果
        ATb = sum(future.result() for future in futures_ATb)
        ATA = sum(future.result() for future in futures_ATA)
        
        # 添加正则化项
        ATA_reg = ATA + alpha * np.eye(n)
        
        # 求解正则化系统
        x = linalg.solve(ATA_reg, ATb, assume_a='pos')
        
        # 计算条件数
        condition_number = np.linalg.cond(ATA_reg)
        
        info = {
            'alpha': alpha,
            'condition_number': condition_number,
            'parallel_chunks': n_chunks
        }
        
        return x, info
    
    @staticmethod
    def recommend_method(A: Union[np.ndarray, sp.spmatrix], memory_limit: float = None) -> str:
        """根据矩阵特性推荐求解方法
        
        Args:
            A: 系统矩阵
            memory_limit: 内存限制（以GB为单位）
        
        Returns:
            str: 推荐的求解方法
        """
        m, n = A.shape
        is_sparse = sp.issparse(A)
        density = A.nnz / (m * n) if is_sparse else 1.0
        cpu_cores = os.cpu_count()
        
        # 估计内存需求（以GB为单位）
        element_size = 8  # 假设双精度浮点数
        memory_direct = m * n * element_size / (1024**3)
        
        # 基于矩阵特性的建议
        if m > 1e6 or n > 1e5:
            return "lsmr"  # 超大规模问题
        elif is_sparse and density < 0.01:
            return "lsqr"  # 稀疏矩阵
        elif memory_limit and memory_direct > memory_limit:
            return "lsmr"  # 内存受限
        elif m > 10*n and not is_sparse and m > 5000 and cpu_cores > 2:
            return "parallel_direct"  # 大型密集过定系统，使用并行
        elif m > 10*n:
            return "qr"    # 高度过定（行远多于列）
        elif np.linalg.cond(A) > 1e10 if not is_sparse and m < 1e4 and n < 1e3 else False:
            return "svd"   # 病态问题且规模适中
        elif not is_sparse and m < 1e4 and n < 1e3:
            return "direct"  # 小到中等规模密集矩阵
        else:
            return "lsmr"  # 默认选择
    
    def close(self):
        """关闭资源"""
        if hasattr(self, 'executor'):
            self.executor.shutdown()

    def _get_available_gpu_memory(self):
        """获取当前可用的GPU内存（以GB为单位）"""
        if not self.use_gpu:
            return 0
        
        try:
            free, total = cp.cuda.runtime.memGetInfo()
            return free / (1024**3)
        except:
            return None

    def get_performance_report(self) -> Dict:
        """获取性能报告统计数据"""
        if not self.performance_tracking or not self.performance_history:
            return {"error": "性能跟踪未启用或没有执行求解"}
            
        methods = {}
        total_time = 0
        total_solves = len(self.performance_history)
        
        for record in self.performance_history:
            method = record['method']
            if method not in methods:
                methods[method] = {
                    'count': 0,
                    'total_time': 0,
                    'avg_time': 0,
                    'min_time': float('inf'),
                    'max_time': 0
                }
                
            methods[method]['count'] += 1
            methods[method]['total_time'] += record['solve_time']
            methods[method]['min_time'] = min(methods[method]['min_time'], record['solve_time'])
            methods[method]['max_time'] = max(methods[method]['max_time'], record['solve_time'])
            total_time += record['solve_time']
        
        # 计算平均时间
        for method in methods:
            methods[method]['avg_time'] = methods[method]['total_time'] / methods[method]['count']
        
        return {
            'total_solves': total_solves,
            'total_time': total_time,
            'avg_time': total_time / total_solves if total_solves > 0 else 0,
            'method_stats': methods,
            'backend': 'GPU' if self.use_gpu else 'CPU'
        }
        
    def reset_performance_tracking(self):
        """重置性能跟踪数据"""
        if self.performance_tracking:
            self.performance_history = []

    def _has_cuda_device(self):
        if not CUPY_AVAILABLE:
            return False
        try:
            n_devices = cp.cuda.runtime.getDeviceCount()
            return n_devices > 0
        except Exception:
            return False
