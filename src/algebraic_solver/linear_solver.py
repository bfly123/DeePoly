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

# AttemptImportCuPy，IfUnavailable则提供Warning
try:
    import cupy as cp
    from cupyx.scipy.sparse import linalg as cplinalg
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False
    warnings.warn("CuPy未Installation，GPU加速Function将Unavailable。Using pip install cupy-cuda11x Installation（According toCUDAVersion选择）")


class LinearSolver:
    """过定LinearSystemSolve器，用于SolutionAx=bProblem（A的行数Greater than列数），SupportCuPy GPU加速和CPU并行Compute"""

    def __init__(self, verbose: bool = False, use_gpu: bool = True, 
                 n_jobs: int = None, chunk_size: int = 1000, 
                 gpu_memory_fraction: float = 0.9, 
                 gpu_batch_size: int = None,
                 use_mixed_precision: bool = False,
                 performance_tracking: bool = False,
                 config: object = None):
        """Initialize solver
        
        Args:
            verbose: YesNo打印详细information
            use_gpu: YesNoUsingGPU（IfAvailable）, True表示DefaultUsingGPU
            n_jobs: 并行CPUCore数，None表示UsingEntireAvailableCore
            chunk_size: Matrix分BlockSize，用于并行Compute
            gpu_memory_fraction: GPUMemory usageRatio，防Stop占用Entire显存
            gpu_batch_size: GPU批ProcessSize，None表示自动确定
            use_mixed_precision: YesNoUsing混合Precision（对SomeOperationUsingfloat16/32）
            performance_tracking: YesNo跟踪PerformanceIndicator
            config: Configurationobject，用于覆盖DefaultSetup
        """
        self.verbose = verbose
        self.performance_tracking = performance_tracking
        self.gpu_memory_fraction = gpu_memory_fraction
        self.gpu_batch_size = gpu_batch_size
        self.use_mixed_precision = use_mixed_precision
        
        # Unified device configuration - eliminate branching
        linear_device = getattr(config, 'linear_device', '').lower() if config else ''
        if 'cpu' in linear_device:
            use_gpu = False
        
        # Check实际CUDAEquipment
        if use_gpu and CUPY_AVAILABLE:
            try:
                n_devices = cp.cuda.runtime.getDeviceCount()
                if n_devices == 0:
                    warnings.warn("未DetectionToCUDAEquipment，将Return退ToCPUCompute")
                    use_gpu = False
            except Exception:
                warnings.warn("DetectionCUDAEquipment时Exit错，将Return退ToCPUCompute")
                use_gpu = False
        
        # SetupCPU并行Parameter
        self.n_jobs = n_jobs if n_jobs is not None else max(1, os.cpu_count() - 1)  # 保留一个Core给System
        self.chunk_size = chunk_size
        
        # DetectionYesNo可以UsingGPU
        self.gpu_available = CUPY_AVAILABLE
        
        # IfSetupUsingGPU但GPUUnavailable，displayWarning并Return退ToCPU
        if use_gpu and not self.gpu_available:
            warnings.warn("请求UsingGPU但CuPyUnavailable，将Return退ToCPUCompute")
            self.use_gpu = False
        else:
            self.use_gpu = use_gpu and self.gpu_available
            
        if self.verbose:
            print(f"ComputeBackwardEnd: {'GPU (CuPy)' if self.use_gpu else f'CPU ({self.n_jobs}Core)'}")
            if self.use_mixed_precision:
                print("Using混合PrecisionCompute")
            
        # methodRegister
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
        
        # GPU特定method
        if self.use_gpu:
            self.methods.update({
                "gpu_direct": self._solve_gpu_direct,
                "gpu_svd": self._solve_gpu_svd,
                "gpu_qr": self._solve_gpu_qr,
                "gpu_lstsq": self._solve_gpu_lstsq,
                "gpu_batched_lstsq": self._solve_gpu_batched_lstsq
            })
            
        # CreateLine程池
        self.executor = ThreadPoolExecutor(max_workers=self.n_jobs)
        
        # ConfigurationGPUMemory usage
        if self.use_gpu:
            try:
                # RestrictionMemory usageRatio
                mempool = cp.get_default_memory_pool()
                mempool.set_limit(fraction=self.gpu_memory_fraction)
                if self.verbose:
                    print(f"GPUMemory usageRatioSetup为: {self.gpu_memory_fraction:.2f}")
                    free_mem = self._get_available_gpu_memory()
                    if free_mem:
                        print(f"AvailableGPUInner存: {free_mem:.2f} GB")
            except Exception as e:
                warnings.warn(f"SetupGPUInner存RestrictionFail: {str(e)}")
        
        # Performance跟踪
        self.performance_history = [] if performance_tracking else None
        
        # Step计数器，用于跟踪每步的ResidualNorm
        self.step_counter = 0
    
    def _check_gpu_capability(self) -> bool:
        """CheckGPUYesNo适合Compute（Performance、Inner存等）"""
        if not CUPY_AVAILABLE:
            return False
            
        try:
            # CheckGPUInner存
            free_memory_gb = self._get_available_gpu_memory()
            if free_memory_gb is None or free_memory_gb < 1.0:  # At leastNeed1GBAvailableInner存
                if self.verbose:
                    print("GPUInner存不足，切换ToCPUpattern")
                return False
                
            # 简单Performance testing
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
            
            # 释放TestInner存
            del a_gpu, b_gpu
            cp.get_default_memory_pool().free_all_blocks()
            
            # IfGPU比CPU快，UsingGPU
            use_gpu = gpu_time < cpu_time
            if self.verbose:
                print(f"Performance testing: CPU={cpu_time:.4f}second, GPU={gpu_time:.4f}second")
                print(f"自动选择: {'GPU' if use_gpu else 'CPU'}")
            
            return use_gpu
            
        except Exception as e:
            if self.verbose:
                print(f"GPUDetectionFail: {str(e)}，切换ToCPUpattern")
            return False

    def solve(self, 
              A: Union[np.ndarray, sp.spmatrix], 
              b: np.ndarray,
              method: str = "gpu_auto", 
              fallback: bool = True,
              **kwargs) -> Tuple[np.ndarray, Dict]:
        """SolveLinearSystemAx=b
        
        Args:
            A: SystemMatrix，Shape为(m,n)，m>n（行数Greater than列数）
            b: Right侧Vector，Shape为(m,)或(m,1)
            method: Solvemethod，Default为"gpu_auto"优先选择GPUmethod
            fallback: If选定methodFail，YesNoAttemptReturn退To更安全的method
            **kwargs: 特定method的ExtraParameter
        
        Returns:
            Tuple[np.ndarray, Dict]: SolutionVector和SolveinformationDictionary
        """
        # CheckInput
        m, n = A.shape
        if m < n:
            raise ValueError(f"InputMatrix不Yes过定System：行数({m}) < 列数({n})")
            
        # 确保bYesCorrect的Shape
        if b.ndim == 1:
            b = b.reshape(-1, 1)
        
        # 自动选择method
        original_method = method
        if method == "auto" or method == "gpu_auto":
            method = self._auto_select_method(A)
            # If请求为gpu_auto且有GPU，但选择了CPUmethod，则Attempt找替代GPUmethod
            if method.startswith("gpu_") == False and self.use_gpu and original_method == "gpu_auto":
                # Attempt选择对应的GPUmethod
                if method == "direct":
                    method = "gpu_direct"
                elif method == "svd":
                    method = "gpu_svd"
                elif method == "qr":
                    method = "gpu_qr"
                else:
                    # DefaultUsinggpu_direct
                    method = "gpu_direct"
                    
            if self.verbose:
                print(f"自动选择method: {method}")
        
        # CheckmethodYesNo存At
        if method not in self.methods:
            raise ValueError(f"未知的Solvemethod: {method}。Availablemethod: {list(self.methods.keys())}")
        
        # 计时Start
        start_time = time.time()
        
        try:
            # 调用对应的method
            x, info = self.methods[method](A, b, **kwargs)
        except Exception as e:
            if fallback and method != "direct":
                # Exit现Abnormal，AttemptReturn退To更安全的method
                # IfGPUmethodFail，FirstAttempt其他GPUmethod，Then再Return退ToCPU
                if method.startswith("gpu_") and self.use_gpu:
                    fallback_method = "gpu_batched_lstsq"
                    warnings.warn(f"method {method} Fail: {str(e)}，AttemptReturn退ToGPU批Processmethod")
                else:
                    fallback_method = "direct"
                    warnings.warn(f"method {method} Fail: {str(e)}，AttemptReturn退To直接法")
                    
                return self.solve(A, b, method=fallback_method, fallback=False, **kwargs)
            else:
                raise RuntimeError(f"SolveFail且无法Return退: {str(e)}")
        
        # ComputeSolveTime
        solve_time = time.time() - start_time
        info['solve_time'] = solve_time
        info['method'] = method
        info['original_method'] = original_method
        
        # ComputeResidualNorm（AtCPUUp）
        if not sp.issparse(A):
            residual = A @ x - b
            residual_norm = np.linalg.norm(residual)
            residual_norm_inf = np.linalg.norm(residual, ord=np.inf)
            residual_norm_1 = np.linalg.norm(residual, ord=1)
        else:
            residual = A.dot(x) - b
            residual_norm = np.linalg.norm(residual)
            residual_norm_inf = np.linalg.norm(residual, ord=np.inf)
            residual_norm_1 = np.linalg.norm(residual, ord=1)
            
        # ComputePhase对ResidualNorm
        b_norm = np.linalg.norm(b)
        relative_residual = residual_norm / max(b_norm, 1e-16)
        
        # Store详细Residualinformation
        info['residual_norm'] = residual_norm
        info['residual_norm_inf'] = residual_norm_inf
        info['residual_norm_1'] = residual_norm_1
        info['relative_residual'] = relative_residual
        info['b_norm'] = b_norm
        
        # Step计数器Increment
        self.step_counter += 1
        
        if self.verbose:
            print(f"=== Linear Solve Step {self.step_counter} ===")
            print(f"Method: {method}")
            print(f"Matrix size: {A.shape[0]} x {A.shape[1]}")
            print(f"Condition number: {info.get('condition_number', 'N/A')}")
            print(f"Residual Analysis:")
            print(f"  ||Ax-b||_2 = {residual_norm:.6e}")
            print(f"  ||Ax-b||_∞ = {residual_norm_inf:.6e}")
            print(f"  ||Ax-b||_1 = {residual_norm_1:.6e}")
            print(f"  ||b||_2 = {b_norm:.6e}")
            print(f"  Relative residual = {relative_residual:.6e}")
            print(f"Solve time: {solve_time:.6f}s")
            if 'iterations' in info:
                print(f"Iterations: {info['iterations']}")
            if 'rank' in info:
                print(f"Matrix rank: {info['rank']}")
            print("-" * 45)
                
        # recordPerformance
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
        """自动选择最佳Solvemethod，Based onMatrixSize、Condition number和AvailableCompute资源"""
        m, n = A.shape
        is_sparse = sp.issparse(A)
        
        # For极大规模Problem或Sparse matrix，自动选择IterateSolution法
        if m > 1e6 or n > 1e5 or (is_sparse and A.nnz / (m*n) < 0.05):
            return "lsmr"
        
        # EstimationProblem规模和Inner存需求
        problem_size = m * n
        element_size = 4 if self.use_mixed_precision else 8  # float32或float64
        memory_needed_gb = problem_size * element_size * 3 / (1024**3)  # ExtraCoefficientsYes为安全裕Degree
        
        # IfGPUAvailable，优先考虑GPUmethod
        if self.use_gpu and not is_sparse:
            # CheckAvailableGPUInner存
            gpu_mem_gb = self._get_available_gpu_memory()
            
            if gpu_mem_gb:
                # According toGPUInner存选择method
                if memory_needed_gb > gpu_mem_gb * 0.8:
                    # GPUInner存Close to极限，Using批Processpattern
                    if self.verbose:
                        print(f"MatrixNeed{memory_needed_gb:.2f}GB，GPUInner存{gpu_mem_gb:.2f}GB，切换To批Process")
                    
                    if memory_needed_gb > gpu_mem_gb * 0.95:
                        # Inner存需求Almost等于或More thanGPUInner存，Return退ToCPU并行
                        if problem_size > 5e6 and self.n_jobs > 1:
                            return "parallel_direct"
                        else:
                            return "direct"
                    else:
                        # Using批Processpattern
                        return "gpu_batched_lstsq"
                else:
                    # GPUInner存Adequate，According toProblemCharacteristic选择GPUmethod
                    if problem_size > 1e7:
                        return "gpu_lstsq"
                    elif self._is_ill_conditioned(A):
                        return "gpu_svd"
                    else:
                        return "gpu_direct"
            return "gpu_direct"  # DefaultStatusDownAttempt直接GPUmethod
                
        # CheckCPUInner存（简化Estimation）
        try:
            cpu_memory_gb = psutil.virtual_memory().available / (1024**3)
        except:
            cpu_memory_gb = 8.0  # 保守Estimation
            
        # IfCPUInner存不足，AttemptIterateSolution法
        if memory_needed_gb > cpu_memory_gb * 0.8 and not is_sparse:
            if self.verbose:
                print(f"Inner存需求({memory_needed_gb:.2f}GB)More thanAvailableCPUInner存的80%，UsingIterateSolution法")
            return "lsmr"
        
        # CPUmethod选择
        # For大MatrixUsing并行method
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
        
        # ForSparse matrix
        if is_sparse:
            density = A.nnz / (m * n)
            if density < 0.01:
                return "lsqr"
            else:
                return "lsmr"
                
        # DefaultStatus
        return "direct"
    
    def _is_ill_conditioned(self, A: np.ndarray, sample_size: int = 1000, threshold: float = 1e6) -> bool:
        """EstimationMatrixYesNo病态（可能有较大的Condition number）"""
        if sp.issparse(A):
            return False  # Sparse matrix暂不EstimationCondition number
            
        m, n = A.shape
        
        # For小Matrix，直接ComputeCondition number
        if m < 5000 and n < 1000:
            try:
                return np.linalg.cond(A) > threshold
            except:
                return False
        
        # For大Matrix，SamplingEstimation
        if m > sample_size:
            # 随机Sampling行
            indices = np.random.choice(m, sample_size, replace=False)
            A_sample = A[indices]
            try:
                return np.linalg.cond(A_sample) > threshold / 10  # ReductionThreshold，BecauseSampling可能低估Condition number
            except:
                return False
                
        return False
    
    def _solve_direct(self, A: Union[np.ndarray, sp.spmatrix], b: np.ndarray, **kwargs) -> Tuple[np.ndarray, Dict]:
        """直接最小二乘法，Solve(A^T A)x = A^T b"""
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
        """UsingSVDDecomposeSolve，可以Process病态Problem"""
        # ExecutionSVDDecompose
        U, s, Vh = linalg.svd(A, full_matrices=False)
        
        # Compute伪逆
        rcond = rcond or np.finfo(float).eps * max(A.shape)
        s_mask = s > rcond * s[0]
        s_inv = np.zeros_like(s)
        s_inv[s_mask] = 1/s[s_mask]
        
        # ComputeSolution
        x = Vh.T @ (s_inv.reshape(-1, 1) * (U.T @ b))
        
        # ComputeValid秩和Condition number
        rank = np.sum(s_mask)
        condition_number = s[0] / s[s_mask][-1] if rank > 0 else np.inf
        
        info = {
            'rank': rank,
            'singular_values': s,
            'condition_number': condition_number
        }
        
        return x, info
    
    def _solve_qr(self, A: np.ndarray, b: np.ndarray, **kwargs) -> Tuple[np.ndarray, Dict]:
        """UsingQRDecomposeSolve，比SVD更高效，但对病态ProblemProcess不如SVD"""
        # ExecutionQRDecompose
        Q, R = linalg.qr(A, mode='economic')
        
        # ComputeSolution
        x = linalg.solve_triangular(R, Q.T @ b)
        
        # ComputeCondition number
        condition_number = np.linalg.cond(R)
        
        info = {
            'condition_number': condition_number
        }
        
        return x, info
    
    def _solve_ridge(self, A: Union[np.ndarray, sp.spmatrix], b: np.ndarray, 
                    alpha: float = 1.0, **kwargs) -> Tuple[np.ndarray, Dict]:
        """岭Regression/TikhonovRegularization，适用于病态Problem"""
        m, n = A.shape
        
        if sp.issparse(A):
            # ForSparse matrix，UsingIteratemethodSolveRegularization最小二乘Problem
            # Build增广Matrix [A; sqrt(alpha)*I]
            I = sp.eye(n)
            A_aug = sp.vstack([A, np.sqrt(alpha) * I])
            b_aug = np.vstack([b, np.zeros((n, 1))])
            
            x, istop, itn, normr = splinalg.lsqr(A_aug, b_aug.flatten())[:4]
            info = {'iterations': itn, 'status': istop, 'alpha': alpha}
            return x.reshape(-1, 1), info
        else:
            # For密SetMatrix，Using正规EquationSolve
            ATA = A.T @ A
            ATb = A.T @ b
            
            # 添加RegularizationItem
            ATA_reg = ATA + alpha * np.eye(n)
            
            # SolveRegularizationSystem
            x = linalg.solve(ATA_reg, ATb, assume_a='pos')
            
            # ComputeCondition number
            condition_number = np.linalg.cond(ATA_reg)
            
            info = {
                'alpha': alpha,
                'condition_number': condition_number
            }
            
            return x, info
    
    def _solve_lsqr(self, A: Union[np.ndarray, sp.spmatrix], b: np.ndarray, 
                   tol: float = 1e-8, iter_lim: int = None, **kwargs) -> Tuple[np.ndarray, Dict]:
        """UsingLSQRIterateAlgorithmSolve，适用于大型稀疏System"""
        iter_lim = iter_lim or max(A.shape) * 10
        
        # 确保bYes一维的
        b_flat = b.flatten()
        
        # UsingLSQRSolve
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
        """UsingLSMRIterateAlgorithmSolve，Usually比LSQRConvergent更快"""
        max_iter = max_iter or max(A.shape) * 10
        
        # 确保bYes一维的
        b_flat = b.flatten()
        
        # UsingLSMRSolve
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
        """Using批Process方式AtGPUUpSolve，防Stop显存溢Exit"""
        if not self.use_gpu:
            raise RuntimeError("请求GPUSolve，但GPUUnavailable")
            
        m, n = A.shape
        
        # Estimation一个合理的批ProcessSize
        if self.gpu_batch_size is None:
            try:
                free_memory = cp.cuda.runtime.memGetInfo()[0]
                element_size = 4 if self.use_mixed_precision else 8
                max_rows = int(free_memory * 0.5 / (n * element_size))  # Using50%AvailableInner存
                batch_size = min(max_rows, 10000)  # No more than10000行
            except:
                batch_size = 5000  # 保守Defaultvalue
        else:
            batch_size = self.gpu_batch_size
            
        if self.verbose:
            print(f"UsingGPU批Process，Batch size: {batch_size}行")
        
        # Compute批Number of times
        num_batches = (m + batch_size - 1) // batch_size
        
        # PrepareResultvariable
        if self.use_mixed_precision:
            # 对A和bUsing较低Precision
            dtype = np.float32
        else:
            dtype = np.float64
            
        # InitializeATA和ATb（AtCPUUp）
        ATA = np.zeros((n, n), dtype=dtype)
        ATb = np.zeros((n, b.shape[1]), dtype=dtype)
        
        # 分批Process
        for i in range(num_batches):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, m)
            
            # 提取Current批次
            A_batch = A[start_idx:end_idx].astype(dtype)
            b_batch = b[start_idx:end_idx].astype(dtype)
            
            # 转移ToGPU
            A_gpu = cp.asarray(A_batch)
            b_gpu = cp.asarray(b_batch)
            
            # ComputeCurrent批次的贡献
            ATA_batch = cp.asnumpy(A_gpu.T @ A_gpu)
            ATb_batch = cp.asnumpy(A_gpu.T @ b_gpu)
            
            # 累加Result
            ATA += ATA_batch
            ATb += ATb_batch
            
            # 主动释放GPUInner存
            del A_gpu, b_gpu
            cp.get_default_memory_pool().free_all_blocks()
        
        # FinalSolve（AtCPUUpComplete，以保证Precision）
        x = np.linalg.solve(ATA, ATb)
        
        # ComputeCondition number
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
        """UsingCuPy直接SolveLinear最小二乘Problem，Increase了Inner存管理"""
        if not self.use_gpu:
            raise RuntimeError("请求GPUSolve，但GPUUnavailable")
        
        # CheckMatrixSize，决定YesNoUsingfloat32以节省显存
        m, n = A.shape
        use_float32 = self.use_mixed_precision and m*n > 1e6
        
        # DataTypeConvert
        dtype = np.float32 if use_float32 else np.float64
        
        try:    
            # 转移DataToGPU
            A_gpu = cp.asarray(A, dtype=dtype)
            b_gpu = cp.asarray(b, dtype=dtype)
            
            # UsingCuPy的最小二乘Solve
            x_gpu = cp.linalg.lstsq(A_gpu, b_gpu, rcond=None)[0]
            
            # ComputeCondition number
            try:
                svd_s = cp.linalg.svd(A_gpu, compute_uv=False)
                condition_number = float(svd_s[0] / svd_s[-1])
            except:
                # IfSVDFail，可能YesInner存Problem，跳过Condition numberCompute
                condition_number = None
            
            # 将Result转ReturnCPU
            x = cp.asnumpy(x_gpu)
            
            # 主动释放GPUInner存
            del A_gpu, b_gpu, x_gpu
            cp.get_default_memory_pool().free_all_blocks()
            
        except cp.cuda.memory.OutOfMemoryError:
            if self.verbose:
                print("GPUInner存不足，AttemptUsing批Processmethod")
            return self._solve_gpu_batched_lstsq(A, b, **kwargs)
        
        info = {
            'condition_number': condition_number,
            'precision': '32bit' if use_float32 else '64bit',
            'backend': 'cupy'
        }
        
        return x, info
    
    def _solve_gpu_svd(self, A: np.ndarray, b: np.ndarray, 
                       rcond: float = None, **kwargs) -> Tuple[np.ndarray, Dict]:
        """UsingCuPy的SVDDecomposeSolve"""
        if not self.use_gpu:
            raise RuntimeError("请求GPUSolve，但GPUUnavailable")
            
        # 转移DataToGPU
        A_gpu = cp.asarray(A)
        b_gpu = cp.asarray(b)
        
        # ExecutionSVDDecompose
        U, s, Vh = cp.linalg.svd(A_gpu, full_matrices=False)
        
        # Compute伪逆
        rcond = rcond or float(cp.finfo(cp.float64).eps * max(A_gpu.shape))
        s_mask = s > rcond * s[0]
        s_inv = cp.zeros_like(s)
        s_inv[s_mask] = 1/s[s_mask]
        
        # ComputeSolution
        x_gpu = Vh.T @ (s_inv.reshape(-1, 1) * (U.T @ b_gpu))
        
        # ComputeValid秩和Condition number
        rank = int(cp.sum(s_mask))
        condition_number = float(s[0] / s[s_mask][-1]) if rank > 0 else float('inf')
        
        # 将Result转ReturnCPU
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
        """UsingCuPy的QRDecomposeSolve"""
        if not self.use_gpu:
            raise RuntimeError("请求GPUSolve，但GPUUnavailable")
            
        # 转移DataToGPU
        A_gpu = cp.asarray(A)
        b_gpu = cp.asarray(b)
        
        # ExecutionQRDecompose
        Q, R = cp.linalg.qr(A_gpu, mode='economic')
        
        # ComputeSolution
        QTb = Q.T @ b_gpu
        x_gpu = cp.linalg.solve_triangular(R, QTb)
        
        # ComputeCondition number
        condition_number = float(cp.linalg.cond(R))
        
        # 将Result转ReturnCPU
        x = cp.asnumpy(x_gpu)
        
        info = {
            'condition_number': condition_number,
            'backend': 'cupy'
        }
        
        return x, info
    
    def _solve_gpu_lstsq(self, A: np.ndarray, b: np.ndarray, **kwargs) -> Tuple[np.ndarray, Dict]:
        """UsingCuPy的Encapsulation最小二乘functionSolve，针对大型MatrixOptimize"""
        if not self.use_gpu:
            raise RuntimeError("请求GPUSolve，但GPUUnavailable")
            
        # 转移DataToGPU
        A_gpu = cp.asarray(A)
        b_gpu = cp.asarray(b)
        
        # UsinggelsSolve最小二乘Problem（可能UsingcuSOLVER）
        # CuPyCurrently没有直接暴露gels，Therefore我们Usinglstsq
        x_gpu = cp.linalg.lstsq(A_gpu, b_gpu)[0]
        
        # 将Result转ReturnCPU
        x = cp.asnumpy(x_gpu)
        
        info = {
            'backend': 'cupy'
        }
        
        return x, info
    
    def _process_matrix_chunk(self, func: Callable, A_chunk: np.ndarray, *args, **kwargs) -> np.ndarray:
        """ProcessMatrix分Block的工作function"""
        return func(A_chunk, *args, **kwargs)
        
    def _solve_parallel_direct(self, A: np.ndarray, b: np.ndarray, **kwargs) -> Tuple[np.ndarray, Dict]:
        """UsingLine程池并行Solve的直接法"""
        if sp.issparse(A):
            # Sparse matrix暂不Support并行，Return退ToOrdinarySolve
            return self._solve_direct(A, b, **kwargs)
            
        m, n = A.shape
        
        # CheckProblem规模YesNovalue得并行
        if m < self.n_jobs * self.chunk_size:
            # Problem太小，不value得并行
            return self._solve_direct(A, b, **kwargs)
        
        # Compute分Block数量
        chunk_size = min(self.chunk_size, m // self.n_jobs)
        n_chunks = min(self.n_jobs, m // chunk_size)
        
        # 将Matrix分Block
        A_chunks = np.array_split(A, n_chunks, axis=0)
        b_chunks = np.array_split(b, n_chunks, axis=0)
        
        # 为Each分BlockCompute A^T*b 和 A^T*A
        futures_ATb = []
        futures_ATA = []
        
        for i in range(n_chunks):
            futures_ATb.append(self.executor.submit(
                lambda A_i, b_i: A_i.T @ b_i, A_chunks[i], b_chunks[i]
            ))
            futures_ATA.append(self.executor.submit(
                lambda A_i: A_i.T @ A_i, A_chunks[i]
            ))
        
        # MergeResult
        ATb = sum(future.result() for future in futures_ATb)
        ATA = sum(future.result() for future in futures_ATA)
        
        # SolveMergeBackward的System
        x = linalg.solve(ATA, ATb, assume_a='pos')
        
        # ComputeCondition number
        condition_number = np.linalg.cond(ATA)
        
        info = {
            'condition_number': condition_number,
            'parallel_chunks': n_chunks,
            'parallel_jobs': self.n_jobs
        }
        
        return x, info
    
    def _solve_parallel_svd(self, A: np.ndarray, b: np.ndarray, 
                           rcond: float = None, **kwargs) -> Tuple[np.ndarray, Dict]:
        """Using并行SVDDecomposeSolve，PassDivideData并行ComputePartialResult"""
        if sp.issparse(A):
            # Sparse matrix暂不Support并行SVD，Return退ToOrdinarySolve
            return self._solve_svd(A, b, rcond=rcond, **kwargs)
        
        m, n = A.shape
        
        # For小Matrix，直接UsingStandardSVD
        if m < 5000 or n < 1000:
            return self._solve_svd(A, b, rcond=rcond, **kwargs)
        
        # ExecutionSVDDecompose
        # Note：IntactSVD很难并行化，但可以并行PrepareData或Using分BlockSVDApproximate
        # 这Inside仍UsingStandardSVD，但未Come可Enter一步Optimize
        U, s, Vh = linalg.svd(A, full_matrices=False)
        
        # 并行Compute U.T @ b
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
        
        # Compute伪逆
        rcond = rcond or np.finfo(float).eps * max(A.shape)
        s_mask = s > rcond * s[0]
        s_inv = np.zeros_like(s)
        s_inv[s_mask] = 1/s[s_mask]
        
        # ComputeSolution
        x = Vh.T @ (s_inv.reshape(-1, 1) * UTb)
        
        # ComputeValid秩和Condition number
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
        """并行Implementation的岭Regression/TikhonovRegularization"""
        m, n = A.shape
        
        # Using并行ComputeATb和ATA
        chunk_size = min(self.chunk_size, m // self.n_jobs)
        n_chunks = min(self.n_jobs, m // chunk_size)
        
        # 将Matrix分Block
        A_chunks = np.array_split(A, n_chunks, axis=0)
        b_chunks = np.array_split(b, n_chunks, axis=0)
        
        # 并行ComputeATA和ATb
        futures_ATb = []
        futures_ATA = []
        
        for i in range(n_chunks):
            futures_ATb.append(self.executor.submit(
                lambda A_i, b_i: A_i.T @ b_i, A_chunks[i], b_chunks[i]
            ))
            futures_ATA.append(self.executor.submit(
                lambda A_i: A_i.T @ A_i, A_chunks[i]
            ))
        
        # MergeResult
        ATb = sum(future.result() for future in futures_ATb)
        ATA = sum(future.result() for future in futures_ATA)
        
        # 添加RegularizationItem
        ATA_reg = ATA + alpha * np.eye(n)
        
        # SolveRegularizationSystem
        x = linalg.solve(ATA_reg, ATb, assume_a='pos')
        
        # ComputeCondition number
        condition_number = np.linalg.cond(ATA_reg)
        
        info = {
            'alpha': alpha,
            'condition_number': condition_number,
            'parallel_chunks': n_chunks
        }
        
        return x, info
    
    @staticmethod
    def recommend_method(A: Union[np.ndarray, sp.spmatrix], memory_limit: float = None) -> str:
        """According toMatrixCharacteristicRecommendationSolvemethod
        
        Args:
            A: SystemMatrix
            memory_limit: Inner存Restriction（以GB为单位）
        
        Returns:
            str: Recommendation的Solvemethod
        """
        m, n = A.shape
        is_sparse = sp.issparse(A)
        density = A.nnz / (m * n) if is_sparse else 1.0
        cpu_cores = os.cpu_count()
        
        # EstimationInner存需求（以GB为单位）
        element_size = 8  # Assumption双Precision浮point数
        memory_direct = m * n * element_size / (1024**3)
        
        # Based onMatrixCharacteristic的Suggestion
        if m > 1e6 or n > 1e5:
            return "lsmr"  # 超大规模Problem
        elif is_sparse and density < 0.01:
            return "lsqr"  # Sparse matrix
        elif memory_limit and memory_direct > memory_limit:
            return "lsmr"  # Inner存受限
        elif m > 10*n and not is_sparse and m > 5000 and cpu_cores > 2:
            return "parallel_direct"  # 大型密Set过定System，Using并行
        elif m > 10*n:
            return "qr"    # 高Degree过定（行远多于列）
        elif np.linalg.cond(A) > 1e10 if not is_sparse and m < 1e4 and n < 1e3 else False:
            return "svd"   # 病态Problem且规模适中
        elif not is_sparse and m < 1e4 and n < 1e3:
            return "direct"  # 小To中等规模密SetMatrix
        else:
            return "lsmr"  # Default选择
    
    def close(self):
        """Close资源"""
        if hasattr(self, 'executor'):
            self.executor.shutdown()

    def _get_available_gpu_memory(self):
        """GetCurrentAvailable的GPUInner存（以GB为单位）"""
        if not self.use_gpu:
            return 0
        
        try:
            free, total = cp.cuda.runtime.memGetInfo()
            return free / (1024**3)
        except:
            return None

    def get_performance_report(self) -> Dict:
        """GetPerformanceReportStatisticsData"""
        if not self.performance_tracking or not self.performance_history:
            return {"error": "Performance跟踪未Enable或没有ExecutionSolve"}
            
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
        
        # ComputeAverageTime
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
        """ResetPerformance跟踪Data"""
        if self.performance_tracking:
            self.performance_history = []
            
    def reset_step_counter(self):
        """ResetStep计数器，适用于NewTime step或新Problem"""
        self.step_counter = 0
        if self.verbose:
            print("=== Linear Solver Step Counter Reset ===")

    def _has_cuda_device(self):
        if not CUPY_AVAILABLE:
            return False
        try:
            n_devices = cp.cuda.runtime.getDeviceCount()
            return n_devices > 0
        except Exception:
            return False
