#!/usr/bin/env python3
"""
Allen-Cahn Reference Solution Solver
Loads and processes MATLAB reference data from Chebfun solution
"""

import numpy as np
import scipy.io
import os
from scipy.interpolate import interp1d
import warnings

class ACReferenceSolver:
    """Reference solver for Allen-Cahn equation using MATLAB/Chebfun data"""
    
    def __init__(self, project_root):
        """
        Initialize reference solver
        
        Args:
            project_root: Root directory of the project
        """
        self.project_root = project_root
        self.mat_file_path = os.path.join(
            project_root, 
            "cases", "Time_pde_cases", "AC_equation", "reference_data", "allen_cahn.mat"
        )
        self.ref_data = None
        self._load_reference_data()
    
    def _load_reference_data(self):
        """Load reference data from MATLAB file"""
        try:
            if not os.path.exists(self.mat_file_path):
                raise FileNotFoundError(f"Reference data file not found: {self.mat_file_path}")
            
            # Load MATLAB data
            mat_data = scipy.io.loadmat(self.mat_file_path)
            
            # Extract arrays (MATLAB saves as nested arrays)
            self.ref_data = {
                't': mat_data['t'].flatten(),  # Time array
                'x': mat_data['x'].flatten(),  # Spatial array
                'usol': mat_data['usol']       # Solution matrix (time, space)
            }
            
            print(f"Reference data loaded successfully:")
            print(f"  Time domain: [{self.ref_data['t'][0]:.6f}, {self.ref_data['t'][-1]:.6f}] ({len(self.ref_data['t'])} points)")
            print(f"  Space domain: [{self.ref_data['x'][0]:.6f}, {self.ref_data['x'][-1]:.6f}] ({len(self.ref_data['x'])} points)")
            print(f"  Solution shape: {self.ref_data['usol'].shape}")
            
        except Exception as e:
            warnings.warn(f"Failed to load reference data: {e}")
            self.ref_data = None
    
    def solve_reference(self, T, dt, N_points=None):
        """
        Get reference solution at specified time
        
        Args:
            T: Target time
            dt: Time step (not used, kept for API compatibility)
            N_points: Number of spatial points (optional)
            
        Returns:
            Dictionary with reference solution data
        """
        if self.ref_data is None:
            raise RuntimeError("Reference data not available")
        
        # Find closest time index
        time_idx = np.argmin(np.abs(self.ref_data['t'] - T))
        actual_time = self.ref_data['t'][time_idx]
        
        # Get solution at this time
        u_ref = self.ref_data['usol'][time_idx, :]
        x_ref = self.ref_data['x']
        
        # Interpolate to different spatial resolution if requested
        if N_points is not None and N_points != len(x_ref):
            x_new = np.linspace(x_ref[0], x_ref[-1], N_points)
            interpolator = interp1d(x_ref, u_ref, kind='cubic', bounds_error=False, fill_value='extrapolate')
            u_ref = interpolator(x_new)
            x_ref = x_new
        
        return {
            'x': x_ref,
            'u': u_ref,
            'time': actual_time,
            'time_index': time_idx,
            'requested_time': T
        }
    
    def compare_with_training_data(self, train_data, deepoly_segments, ref_data, T):
        """
        Compare DeePoly solution with reference at training points
        
        Args:
            train_data: Training data dictionary with segment information
            deepoly_segments: DeePoly solution segments
            ref_data: Reference data from solve_reference
            T: Current time
            
        Returns:
            Dictionary with detailed comparison metrics
        """
        try:
            # Get reference solution
            x_ref = ref_data['x']
            u_ref = ref_data['u']
            
            # Process training data segments
            x_segments = train_data['x_segments_norm']
            n_segments = len(x_segments)
            
            # Convert normalized coordinates back to physical coordinates
            x_domain = train_data.get('x_domain', [[-1, 1]])  # Default AC domain
            x_min, x_max = x_domain[0][0], x_domain[0][1]
            
            # Collect all training points and DeePoly values
            x_train_all = []
            deepoly_all = []
            
            for seg_idx, x_seg_norm in enumerate(x_segments):
                # Convert normalized coordinates to physical
                x_seg_phys = x_seg_norm * (x_max - x_min) / 2 + (x_max + x_min) / 2
                x_train_all.extend(x_seg_phys.flatten())
                
                # Get DeePoly solution for this segment
                if hasattr(deepoly_segments, '__len__') and seg_idx < len(deepoly_segments):
                    u_seg = deepoly_segments[seg_idx]
                    if hasattr(u_seg, 'flatten'):
                        deepoly_all.extend(u_seg.flatten())
                    else:
                        deepoly_all.extend([u_seg] * len(x_seg_phys.flatten()))
                else:
                    # Fallback: use zeros
                    deepoly_all.extend([0.0] * len(x_seg_phys.flatten()))
            
            x_train_all = np.array(x_train_all)
            deepoly_all = np.array(deepoly_all)
            
            # Interpolate reference solution at training points
            ref_interpolator = interp1d(x_ref, u_ref, kind='cubic', bounds_error=False, fill_value='extrapolate')
            ref_at_train = ref_interpolator(x_train_all)
            
            # Compute global metrics
            abs_error = np.abs(deepoly_all - ref_at_train)
            
            # Handle potential numerical issues
            ref_norm = np.linalg.norm(ref_at_train)
            deepoly_norm = np.linalg.norm(deepoly_all)
            
            if ref_norm < 1e-12:
                relative_errors = np.inf * np.ones(3)  # L1, L2, Linf
            else:
                relative_errors = [
                    np.sum(abs_error) / np.sum(np.abs(ref_at_train)),  # Relative L1
                    np.linalg.norm(abs_error) / ref_norm,              # Relative L2
                    np.max(abs_error) / np.max(np.abs(ref_at_train))   # Relative Linf
                ]
            
            # Compute correlation
            if len(ref_at_train) > 1 and np.std(ref_at_train) > 1e-12 and np.std(deepoly_all) > 1e-12:
                correlation = np.corrcoef(ref_at_train, deepoly_all)[0, 1]
                if np.isnan(correlation):
                    correlation = 0.0
            else:
                correlation = 0.0
            
            global_metrics = {
                'n_total_points': len(x_train_all),
                'n_segments': n_segments,
                'L1_error': np.sum(abs_error),
                'L2_error': np.linalg.norm(abs_error),
                'Linf_error': np.max(abs_error),
                'relative_L1': relative_errors[0],
                'relative_L2': relative_errors[1],
                'relative_Linf': relative_errors[2],
                'reference_norm': ref_norm,
                'deepoly_norm': deepoly_norm,
                'correlation': correlation
            }
            
            # Compute segment-wise metrics
            segment_metrics = []
            start_idx = 0
            
            for seg_idx, x_seg_norm in enumerate(x_segments):
                seg_len = len(x_seg_norm.flatten())
                end_idx = start_idx + seg_len
                
                if end_idx <= len(x_train_all):
                    x_seg_train = x_train_all[start_idx:end_idx]
                    deepoly_seg = deepoly_all[start_idx:end_idx]
                    ref_seg = ref_at_train[start_idx:end_idx]
                    
                    seg_abs_error = np.abs(deepoly_seg - ref_seg)
                    
                    segment_metrics.append({
                        'segment_idx': seg_idx,
                        'n_points': seg_len,
                        'L2_error': np.linalg.norm(seg_abs_error),
                        'Linf_error': np.max(seg_abs_error),
                        'deepoly_points': len(deepoly_seg),
                        'ref_points': len(ref_seg)
                    })
                
                start_idx = end_idx
            
            return {
                'global_metrics': global_metrics,
                'segment_metrics': segment_metrics,
                'deepoly_global': deepoly_all,
                'reference_at_training': {
                    'U_global_ref': ref_at_train,
                    'x_global': x_train_all,
                    'time': T
                }
            }
            
        except Exception as e:
            # Return minimal comparison data in case of error
            print(f"Warning: Reference comparison failed: {e}")
            return {
                'global_metrics': {
                    'n_total_points': 0,
                    'n_segments': 0,
                    'L1_error': np.inf,
                    'L2_error': np.inf,
                    'Linf_error': np.inf,
                    'relative_L1': np.inf,
                    'relative_L2': np.inf,
                    'relative_Linf': np.inf,
                    'reference_norm': 0.0,
                    'deepoly_norm': 0.0,
                    'correlation': 0.0
                },
                'segment_metrics': [],
                'deepoly_global': np.array([]),
                'reference_at_training': {
                    'U_global_ref': np.array([]),
                    'x_global': np.array([]),
                    'time': T
                }
            }
    
    def get_initial_condition(self, x):
        """
        Get Allen-Cahn initial condition: u₀(x) = x² cos(πx)
        
        Args:
            x: Spatial coordinates
            
        Returns:
            Initial condition values
        """
        return x**2 * np.cos(np.pi * x)
    
    def get_equation_info(self):
        """Get information about the Allen-Cahn equation"""
        return {
            'name': 'Allen-Cahn Equation',
            'equation': 'du/dt = 0.0001 * d²u/dx² + 5u - 5u³',
            'domain': '[-1, 1]',
            'initial_condition': 'u₀(x) = x² cos(πx)',
            'time_domain': '[0, 1]',
            'parameters': {
                'diffusion': 0.0001,
                'linear_coeff': 5,
                'cubic_coeff': -5
            }
        }