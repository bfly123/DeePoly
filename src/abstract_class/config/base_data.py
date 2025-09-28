import numpy as np
import torch
import os
import importlib.util
import sys
from typing import Dict, List, Tuple, Optional, Union
from abc import ABC, abstractmethod

class BaseDataGenerator(ABC):
    """Base data generator class for handling data generation and processing
    
    This class provides the foundation for generating and processing data for various types of problems.
    It handles boundary conditions, domain points, and data normalization.
    """
    
    def __init__(self, config):
        """Initialize the data generator
        
        Args:
            config: Configuration object containing problem parameters
        """
        self.config = config
        self.n_segments = config.n_segments  # Segment configuration 
        self.Ns = np.prod(config.n_segments)  # Total number of segments
        self.Nw = config.points_per_swap  # Number of boundary points per swap
        self.n_dim = config.n_dim
        self.n_eqs = config.n_eqs
        self.case_dir = getattr(config, "case_dir", None)
        self.custom_data_generator = self._load_custom_data_generator()

    def _load_custom_data_generator(self) -> Optional[object]:
        """Load custom data generator module from the case directory
        
        Returns:
            Optional[object]: Custom data generator module if found, None otherwise
        """
        if not self.case_dir:
            print("Case path not set, using default data generation method")
            return None

        data_generate_path = os.path.join(self.case_dir, "data_generate.py")
        if not os.path.exists(data_generate_path):
            print(f"Custom data generator module not found: {data_generate_path}")
            return None

        if self.case_dir not in sys.path:
            sys.path.insert(0, self.case_dir)
            
        try:
            return importlib.import_module("data_generate")
        except ImportError as e:
            print(f"Error importing data_generate module: {e}")
            return None

    def _parse_math_expression(self, expr: str, x: np.ndarray) -> np.ndarray:
        """Parse and evaluate mathematical expressions
        
        Args:
            expr: Mathematical expression string
            x: Input coordinates for evaluation
            
        Returns:
            np.ndarray: Evaluated values
        """
        # Import necessary modules
        import numpy as np
        from math import sin, cos, tan, exp, pi, sqrt, tanh, sinh, cosh

        # Create namespace with mathematical functions and operations
        namespace = {
            'sin': np.sin,
            'cos': np.cos,
            'tan': np.tan,
            'tanh': np.tanh,
            'sinh': np.sinh,
            'cosh': np.cosh,
            'exp': np.exp,
            'pi': np.pi,
            'sqrt': np.sqrt,
            'log': np.log,
            'log10': np.log10,
            'abs': np.abs,
            'np': np,
            '**': lambda x, y: x ** y,
            '/': lambda x, y: x / y,
            '*': lambda x, y: x * y,
            '+': lambda x, y: x + y,
            '-': lambda x, y: x - y
        }

        # Build function expression
        var_names = ", ".join(self.config.spatial_vars)
        func_str = f"lambda {var_names}: {expr}"

        try:
            value_func = eval(func_str, namespace)
            result = value_func(*[x[:, i] for i in range(x.shape[1])])
            # Ensure result is a numpy array with correct shape
            if isinstance(result, (int, float)):
                result = np.full((x.shape[0], 1), result)
            else:
                result = result.reshape(-1, 1)
            return result
        except Exception as e:
            print(f"Error parsing expression: {e}")
            print(f"Expression: {expr}")
            return np.zeros((x.shape[0], 1))

    def _parse_piecewise_expression(self, value_dict: dict, x: np.ndarray) -> np.ndarray:
        """Parse and evaluate piecewise mathematical expressions
        
        Args:
            value_dict: Dictionary defining the piecewise function
            x: Input coordinates for evaluation
            
        Returns:
            np.ndarray: Evaluated values
        """
        import numpy as np
        
        if value_dict.get('type') != 'piecewise':
            print("Not a piecewise function")
            return np.zeros((x.shape[0], 1))
        
        var = value_dict.get('variable')
        if var not in self.config.spatial_vars:
            print(f"Error: Variable {var} not in spatial variables {self.config.spatial_vars}")
            return np.zeros((x.shape[0], 1))
        
        var_idx = self.config.spatial_vars.index(var)
        x_var = x[:, var_idx]
        result = np.zeros((x.shape[0], 1))
        processed = np.zeros((x.shape[0],), dtype=bool)
        
        print(f"Processing piecewise function on variable {var} (index {var_idx})")
        print(f"x_var range: min={x_var.min()}, max={x_var.max()}")
        print(f"Number of points: {len(x_var)}")
        
        for piece in value_dict.get('pieces', []):
            range_min, range_max = piece.get('range', [float('-inf'), float('inf')])
            expression = piece.get('expression', '0')
            mask = (x_var >= range_min) & (x_var < range_max) & ~processed
            print(f"Range [{range_min}, {range_max}): {mask.sum()} points matched")
            if isinstance(expression, str):
                temp_result = self._parse_math_expression(expression, x)
                result[mask] = temp_result[mask]
                print(f"Applied expression '{expression}' to {mask.sum()} points")
            else:
                result[mask] = expression
                print(f"Applied constant {expression} to {mask.sum()} points")
            processed[mask] = True
        
        print(f"Result range after processing: min={result.min()}, max={result.max()}")
        return result

    def _load_source_term(self, x_global: np.ndarray) -> np.ndarray:
        """Load and evaluate source term function

        Args:
            x_global: Global point coordinates

        Returns:
            np.ndarray: Source term values (always same shape as x_global points)
        """
        # Priority 1: Check for new 'S' field in eq dictionary
        if hasattr(self.config, "eq") and isinstance(self.config.eq, dict) and "S" in self.config.eq:
            if self.config.eq["S"] and len(self.config.eq["S"]) > 0:
                source_expr = self.config.eq["S"][0]  # Take first source term
                if source_expr == "0":
                    # Explicitly return zeros with correct shape
                    return np.zeros((x_global.shape[0], 1))
                else:
                    # Parse and evaluate the expression
                    return self._parse_math_expression(source_expr, x_global)

        # Priority 2: Check legacy 'source_term' field
        if hasattr(self.config, "source_term"):
            if isinstance(self.config.source_term, str):
                return self._parse_math_expression(self.config.source_term, x_global)
            elif self.config.source_term is True:
                if hasattr(self.custom_data_generator, "generate_source_term"):
                    return self.custom_data_generator.generate_source_term(x_global)
                else:
                    raise ValueError("Custom source term generator not found")

        # Default: return zeros with correct shape
        return np.zeros((x_global.shape[0], 1))

    def read_boundary_conditions(self) -> Dict:
        """纯AbstractUBoundary conditionsRead - ExcludeAnyPhysicalvariable引用"""
        if not hasattr(self.config, 'boundary_conditions') or not self.config.boundary_conditions:
            return {}
            
        # 为EachU分量IndexInitializeBoundary conditionsStore - 纯Abstractstructure
        boundary_data = {}
        for var_idx in range(len(self.config.vars_list)):
            boundary_data[var_idx] = {
                'dirichlet': {'x': [], 'values': []},
                'neumann': {'x': [], 'values': [], 'normals': []},
                'robin': {'x': [], 'values': [], 'params': [], 'normals': []},
                'periodic': {'pairs': []}  # UniversalCycleBoundary conditions对
            }
        
        # 简化的Boundary conditionsProcess逻辑 - Based onUVectorIndex
        for bc in self.config.boundary_conditions:
            region = bc['region']
            bc_type = bc['type'].lower()
            points = bc['points']
            
            # ProcessCycleBoundary conditions
            if bc_type == 'periodic':
                # CycleBoundary conditionsNeed配对information
                if 'pair_with' not in bc:
                    continue
                pair_region = bc['pair_with']
                constraint_type = bc.get('constraint', 'dirichlet')  # Default为dirichletCycle
                
                # Generate两个Boundary的point
                x_boundary_1 = self._generate_boundary_points(region, points)
                x_boundary_2 = self._generate_boundary_points(pair_region, points)
                
                if x_boundary_1.size == 0 or x_boundary_2.size == 0:
                    continue
                
                # Convertregion为segmentIndex
                segment_1 = self._region_to_segment(region)
                segment_2 = self._region_to_segment(pair_region)
                
                # 添加CycleBoundary conditions对
                for var_idx in range(len(self.config.vars_list)):
                    periodic_pair = {
                        'segment_1': segment_1,  # 直接StoresegmentIndex
                        'segment_2': segment_2,  # 直接StoresegmentIndex
                        'x_1': x_boundary_1,
                        'x_2': x_boundary_2,
                        'constraint_type': constraint_type  # 'dirichlet' 或 'neumann'
                    }
                    
                    if constraint_type == 'neumann':
                        # ForNeumannCycleCondition，还Need法Vector
                        normals_1 = self._get_boundary_normals(region, x_boundary_1.shape[0])
                        normals_2 = self._get_boundary_normals(pair_region, x_boundary_2.shape[0])
                        periodic_pair['normals_1'] = normals_1
                        periodic_pair['normals_2'] = normals_2
                    
                    boundary_data[var_idx]['periodic']['pairs'].append(periodic_pair)
                continue
            
            # Process常规Boundary conditions
            value = bc['value']
            
            # GenerateBoundarypoint和法Vector
            x_boundary = self._generate_boundary_points(region, points)
            if x_boundary.size == 0:
                continue
                
            normals = self._get_boundary_normals(region, x_boundary.shape[0]) if bc_type != 'dirichlet' else None
            
            # ComputeBoundaryvalue
            if isinstance(value, str):
                boundary_values = self._parse_math_expression(value, x_boundary)
            else:
                boundary_values = np.full((x_boundary.shape[0], 1), float(value))
            
            # 添加ToAllU分量的Boundary conditions（DefaultAll分量UsingSameBoundary conditions）
            for var_idx in range(len(self.config.vars_list)):
                boundary_data[var_idx][bc_type]['x'].append(x_boundary)
                boundary_data[var_idx][bc_type]['values'].append(boundary_values)
                if normals is not None:
                    boundary_data[var_idx][bc_type]['normals'].append(normals)
        
        # MergeBoundaryData - 纯AbstractProcess
        for var_idx in boundary_data:
            for bc_type in boundary_data[var_idx]:
                if bc_type == 'periodic':
                    # CycleBoundary conditionsDo not needMerge，MaintainpairsListstructure
                    continue
                elif boundary_data[var_idx][bc_type]['x']:
                    boundary_data[var_idx][bc_type]['x'] = np.vstack(boundary_data[var_idx][bc_type]['x'])
                    boundary_data[var_idx][bc_type]['values'] = np.vstack(boundary_data[var_idx][bc_type]['values'])
                    if bc_type in ['neumann', 'robin'] and boundary_data[var_idx][bc_type]['normals']:
                        boundary_data[var_idx][bc_type]['normals'] = np.vstack(boundary_data[var_idx][bc_type]['normals'])
                else:
                    boundary_data[var_idx][bc_type]['x'] = np.array([])
                    boundary_data[var_idx][bc_type]['values'] = np.array([])
                    if bc_type in ['neumann', 'robin']:
                        boundary_data[var_idx][bc_type]['normals'] = np.array([])
        
        return boundary_data

    def _generate_boundary_points(self, region: str, points: int) -> np.ndarray:
        """Generate points for specified boundary region
        
        Args:
            region: Boundary region identifier
            points: Number of points to generate
            
        Returns:
            np.ndarray: Generated boundary points
        """
        x_domain = self.config.x_domain
        
        # Generic boundary generation for arbitrary dimensions
        if region.startswith('dim'):
            try:
                parts = region.split('_')
                dim_idx = int(parts[0][3:])
                is_max = parts[1] == 'max'
                
                boundary_value = x_domain[dim_idx][1] if is_max else x_domain[dim_idx][0]
                boundary_points = np.zeros((points, self.n_dim))
                boundary_points[:, dim_idx] = boundary_value
                
                for d in range(self.n_dim):
                    if d != dim_idx:
                        boundary_points[:, d] = np.linspace(
                            x_domain[d][0], x_domain[d][1], points
                        )
                
                return boundary_points
            except (ValueError, IndexError) as e:
                print(f"Error parsing boundary region {region}: {e}")
                return np.array([])
        
        # Legacy support for named boundaries
        if self.n_dim == 1:
            if region == 'left':
                return np.array([[x_domain[0][0]]])
            elif region == 'right':
                return np.array([[x_domain[0][1]]])
        
        elif self.n_dim == 2:
            x_min, x_max = x_domain[0][0], x_domain[0][1]
            y_min, y_max = x_domain[1][0], x_domain[1][1]
            
            if region == 'left':
                y_coords = np.linspace(y_min, y_max, points)
                return np.column_stack((np.ones(points) * x_min, y_coords))
            elif region == 'right':
                y_coords = np.linspace(y_min, y_max, points)
                return np.column_stack((np.ones(points) * x_max, y_coords))
            elif region == 'bottom':
                x_coords = np.linspace(x_min, x_max, points)
                return np.column_stack((x_coords, np.ones(points) * y_min))
            elif region == 'top':
                x_coords = np.linspace(x_min, x_max, points)
                return np.column_stack((x_coords, np.ones(points) * y_max))
        
        elif self.n_dim == 3:
            x_min, x_max = x_domain[0][0], x_domain[0][1]
            y_min, y_max = x_domain[1][0], x_domain[1][1]
            z_min, z_max = x_domain[2][0], x_domain[2][1]
            
            points_per_dim = int(np.sqrt(points))
            
            if region in ['left', 'right']:
                x_val = x_min if region == 'left' else x_max
                y_coords = np.linspace(y_min, y_max, points_per_dim)
                z_coords = np.linspace(z_min, z_max, points_per_dim)
                y_grid, z_grid = np.meshgrid(y_coords, z_coords)
                return np.column_stack((
                    np.ones(points_per_dim**2) * x_val,
                    y_grid.flatten(),
                    z_grid.flatten()
                ))
        
        print(f"Warning: Boundary region '{region}' not supported for {self.n_dim}D")
        return np.array([])
    
    def _region_to_segment(self, region: str) -> int:
        """Convert region name to segment index
        
        Args:
            region: Region name like 'left', 'right', 'dim0_min', 'dim0_max'
            
        Returns:
            int: Segment index
        """
        if region.startswith('dim'):
            try:
                parts = region.split('_')
                dim_idx = int(parts[0][3:])
                is_max = parts[1] == 'max'
                
                # ComputesegmentIndex：ForBoundary，Yes该Dimensions的最小或最大segment
                segment_coords = [0] * self.n_dim
                if is_max:
                    segment_coords[dim_idx] = self.n_segments[dim_idx] - 1
                else:
                    segment_coords[dim_idx] = 0
                    
                return self._coords_to_segment_index(segment_coords)
                
            except (ValueError, IndexError):
                return 0
        
        # Legacy support for named boundaries
        if self.n_dim == 1:
            if region == 'left':
                return 0  # 第一个segment
            elif region == 'right':
                return self.n_segments[0] - 1  # Finally一个segment
        
        elif self.n_dim == 2:
            if region == 'left':
                return 0  # segment (0, *)
            elif region == 'right':
                return self.n_segments[0] - 1  # segment (nx-1, *)
            elif region == 'bottom':
                return 0  # segment (*, 0)  
            elif region == 'top':
                return (self.n_segments[1] - 1) * self.n_segments[0]  # segment (*, ny-1)
        
        return 0  # DefaultReturn第一个segment
    
    def _coords_to_segment_index(self, coords: list) -> int:
        """Convert segment coordinates to linear index"""
        idx = 0
        for i, coord in enumerate(coords):
            if i == 0:
                idx += coord
            else:
                idx += coord * np.prod(self.n_segments[:i])
        return idx

    def _get_boundary_normals(self, region: str, num_points: int) -> np.ndarray:
        """Get normal vectors for boundary
        
        Args:
            region: Boundary region identifier
            num_points: Number of points
            
        Returns:
            np.ndarray: Normal vectors
        """
        normals = np.zeros((num_points, self.n_dim))
        
        if region.startswith('dim'):
            try:
                parts = region.split('_')
                dim_idx = int(parts[0][3:])
                is_max = parts[1] == 'max'
                normals[:, dim_idx] = 1.0 if is_max else -1.0
                return normals
            except (ValueError, IndexError):
                pass
        
        if self.n_dim == 1:
            if region == 'left':
                normals[:, 0] = -1.0
            elif region == 'right':
                normals[:, 0] = 1.0
            
        elif self.n_dim == 2:
            if region == 'left':
                normals[:, 0] = -1.0
            elif region == 'right':
                normals[:, 0] = 1.0
            elif region == 'bottom':
                normals[:, 1] = -1.0
            elif region == 'top':
                normals[:, 1] = 1.0
            
        elif self.n_dim == 3:
            if region == 'left':
                normals[:, 0] = -1.0
            elif region == 'right':
                normals[:, 0] = 1.0
            elif region == 'bottom':
                normals[:, 1] = -1.0
            elif region == 'top':
                normals[:, 1] = 1.0
            elif region == 'back':
                normals[:, 2] = -1.0
            elif region == 'front':
                normals[:, 2] = 1.0
        
        return normals

    def _generate_global_points(self, mode: str) -> np.ndarray:
        """Generate global sampling points
        
        Args:
            mode: Data mode ("train" or "test")
            
        Returns:
            np.ndarray: Generated points
        """
        if mode == "train":
            Np_total = self.config.points_domain
            points = []
            for i in range(self.n_dim):
                points.append(np.random.uniform(
                    self.config.x_domain[i][0],
                    self.config.x_domain[i][1],
                    Np_total
                ))
            return np.column_stack(points)
        else:
            if isinstance(self.config.points_domain_test, int):
                Np_total = self.config.points_domain_test
                points = []
                for i in range(self.n_dim):
                    points.append(np.random.uniform(
                        self.config.x_domain[i][0]+0.01,
                        self.config.x_domain[i][1]-0.01,
                        Np_total
                    ))
                return np.column_stack(points)
            else:
                grids = []
                for i in range(self.n_dim):
                    grids.append(np.linspace(
                        self.config.x_domain[i][0]+0.01,
                        self.config.x_domain[i][1]-0.01,
                        self.config.points_domain_test[i]
                    ))
                return np.array(np.meshgrid(*grids)).reshape(self.n_dim, -1).T

    def _normalize_data(self, x: np.ndarray, x_min: np.ndarray, x_max: np.ndarray) -> np.ndarray:
        """Normalize data to [0,1] range
        
        Args:
            x: Input data
            x_min: Minimum values
            x_max: Maximum values
            
        Returns:
            np.ndarray: Normalized data
        """
        x = np.asarray(x, dtype=np.float64)
        x_min = np.asarray(x_min, dtype=np.float64)
        x_max = np.asarray(x_max, dtype=np.float64)

        normalized = np.zeros_like(x)
        for i in range(x.shape[-1]):
            if x_max[..., i] - x_min[..., i] > 1e-10:
                normalized[..., i] = (x[..., i] - x_min[..., i]) / (
                    x_max[..., i] - x_min[..., i]
                )
            else:
                normalized[..., i] = x[..., i]
        return normalized

    def split_global_points(self, x_global: np.ndarray) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """Split global points into local segments
        
        Args:
            x_global: Global point coordinates
            
        Returns:
            Tuple containing lists of segmented points and masks
        """
        x_segments = []
        masks = []
        
        for n in range(self.Ns):
            mask = self._create_segment_mask(x_global, n)
            x_seg = x_global[mask]
            x_segments.append(x_seg)
            masks.append(mask)
            
        return x_segments, masks

    def split_global_field(self, masks: List[np.ndarray], u_global: np.ndarray) -> List[np.ndarray]:
        """Split global field into local segments
        
        Args:
            masks: List of segment masks
            u_global: Global field values
            
        Returns:
            List[np.ndarray]: Segmented field values
        """
        if len(u_global.shape) == 1:
            u_global = u_global.reshape(-1, 1)
            
        u_segments = []
        for n in range(self.Ns):
            mask = masks[n]
            u_seg = u_global[mask]
            u_segments.append(u_seg)
        
        return u_segments

    def _create_segment_mask(self, x: np.ndarray, i: int) -> np.ndarray:
        """Create segment mask
        
        Args:
            x: Point coordinates
            i: Segment index
            
        Returns:
            np.ndarray: Boolean mask for segment
        """
        x_min, x_max = self.config.x_min[i], self.config.x_max[i]
        masks = []
        for j in range(self.n_dim):
            masks.append(x[:, j] >= x_min[j]-0.001)
            masks.append(x[:, j] <= x_max[j]+0.001)
        
        main_mask = np.logical_and.reduce(masks)
        
       #if i == 0:
       #    boundary_mask = np.logical_or.reduce([
       #        np.isclose(x[:, j], x_min[j]) for j in range(self.n_dim)
       #    ])
       #    return np.logical_or(main_mask, boundary_mask)
        return main_mask

    def _process_segments(self, x_segments: List[np.ndarray], global_boundary_dict: Dict) -> Tuple[np.ndarray, np.ndarray, List[np.ndarray], List[Dict]]:
        """Process data for each segment
        
        Args:
            x_segments: List of segment coordinates
            global_boundary_dict: Global boundary conditions
            
        Returns:
            Tuple containing processed segment data
        """
        x_segments_norm = []
        x_swap = np.zeros((self.Ns, 2*self.n_dim, self.Nw, self.n_dim), dtype=np.float64)
        x_swap_norm = np.zeros_like(x_swap)
        boundary_segments_dict = []
        
        for n in range(self.Ns):
            x_segments_norm.append(self._normalize_data(x_segments[n], self.config.x_min[n], self.config.x_max[n]))
            
            if self.config.points_per_swap > 0:
                x_swap_n = self._generate_swap_points(n)
                x_swap[n] = x_swap_n
                x_swap_norm[n] = self._normalize_swap_data(x_swap_n, n)
            
            segment_boundary_dict = self._process_segment_boundary(global_boundary_dict, n)
            boundary_segments_dict.append(segment_boundary_dict)
        
        return x_segments_norm, x_swap, x_swap_norm, boundary_segments_dict

    def _generate_swap_points(self, n: int) -> np.ndarray:
        """Generate swap points for segment
        
        Args:
            n: Segment index
            
        Returns:
            np.ndarray: Generated swap points
        """
        x_min, x_max = self.config.x_min, self.config.x_max
        x_swap = np.zeros((2*self.n_dim, self.Nw, self.n_dim))
        
        for i in range(self.n_dim):
            x_swap[2*i] = self._generate_swap_boundary_points(x_min[n], x_max[n], i, 0)
            x_swap[2*i+1] = self._generate_swap_boundary_points(x_min[n], x_max[n], i, 1)
            
        return x_swap

    def _generate_swap_boundary_points(self, x_min: np.ndarray, x_max: np.ndarray, dim: int, is_upper: int) -> np.ndarray:
        """Generate swap boundary points
        
        Args:
            x_min: Minimum coordinates
            x_max: Maximum coordinates
            dim: Dimension index
            is_upper: Whether to generate upper boundary points
            
        Returns:
            np.ndarray: Generated boundary points
        """
        points = np.zeros((self.Nw, self.n_dim))
        points[:, dim] = x_max[dim] if is_upper else x_min[dim]
        
        for i in range(self.n_dim):
            if i != dim:
                points[:, i] = np.linspace(x_min[i], x_max[i], self.Nw)
                
        return points

    def _normalize_swap_data(self, x_swap: np.ndarray, n: int) -> np.ndarray:
        """Normalize swap point data
        
        Args:
            x_swap: Swap point coordinates
            n: Segment index
            
        Returns:
            np.ndarray: Normalized swap points
        """
        x_swap_norm = np.zeros_like(x_swap)
        for j in range(x_swap.shape[0]):
            for k in range(x_swap.shape[1]):
                x_swap_norm[j, k] = self._normalize_data(
                    x_swap[j, k], self.config.x_min[n], self.config.x_max[n]
                )
        return x_swap_norm

    def _process_segment_boundary(self, global_boundary_data: Dict, segment_idx: int) -> Dict:
        """Process boundary conditions for a segment - 纯AbstractUProcess
        
        Args:
            global_boundary_data: Global boundary conditions (indexed by var_idx)
            segment_idx: Segment index
            
        Returns:
            Dict: Processed segment boundary conditions (indexed by var_idx)
        """
        segment_boundary_data = {}
        x_min, x_max = self.config.x_min, self.config.x_max
        
        for var_idx in global_boundary_data:
            segment_boundary_data[var_idx] = {
                'dirichlet': {'x': [], 'values': []},
                'neumann': {'x': [], 'values': [], 'normals': []},
                'robin': {'x': [], 'values': [], 'params': [], 'normals': []},
                'periodic': {'pairs': []}
            }
            
            for bc_type in global_boundary_data[var_idx]:
                if bc_type == 'periodic':
                    # CycleBoundary conditionsSpecialProcess
                    segment_boundary_data[var_idx][bc_type]['pairs'] = []
                    for pair in global_boundary_data[var_idx][bc_type]['pairs']:
                        # Process第一GroupBoundarypoint
                        x_boundary_1 = pair['x_1']
                        mask_1 = self._create_segment_mask(x_boundary_1, segment_idx)
                        
                        # Process第二GroupBoundarypoint
                        x_boundary_2 = pair['x_2']
                        mask_2 = self._create_segment_mask(x_boundary_2, segment_idx)
                        
                        # If该段有CycleBoundarypoint，则Process
                        if np.any(mask_1) or np.any(mask_2):
                            segment_pair = pair.copy()
                            if np.any(mask_1):
                                x_seg_1 = x_boundary_1[mask_1]
                                x_seg_1_norm = self._normalize_data(x_seg_1, x_min[segment_idx], x_max[segment_idx])
                                segment_pair['x_1'] = x_seg_1_norm
                                if 'normals_1' in pair:
                                    segment_pair['normals_1'] = pair['normals_1'][mask_1]
                            if np.any(mask_2):
                                x_seg_2 = x_boundary_2[mask_2]
                                x_seg_2_norm = self._normalize_data(x_seg_2, x_min[segment_idx], x_max[segment_idx])
                                segment_pair['x_2'] = x_seg_2_norm
                                if 'normals_2' in pair:
                                    segment_pair['normals_2'] = pair['normals_2'][mask_2]
                            
                            segment_boundary_data[var_idx][bc_type]['pairs'].append(segment_pair)
                    continue
                elif 'x' not in global_boundary_data[var_idx][bc_type] or len(global_boundary_data[var_idx][bc_type]['x']) == 0:
                    continue
                    
                x_boundary = global_boundary_data[var_idx][bc_type]['x']
                mask = self._create_segment_mask(x_boundary, segment_idx)
                
                if not np.any(mask):
                    continue
                    
                x_seg = x_boundary[mask]
                values_seg = global_boundary_data[var_idx][bc_type]['values'][mask]
                x_seg_norm = self._normalize_data(x_seg, x_min[segment_idx], x_max[segment_idx])
                
                segment_boundary_data[var_idx][bc_type]['x'] = x_seg_norm
                segment_boundary_data[var_idx][bc_type]['values'] = values_seg
                
                if bc_type in ['neumann', 'robin'] and 'normals' in global_boundary_data[var_idx][bc_type]:
                    normals_seg = global_boundary_data[var_idx][bc_type]['normals'][mask]
                    segment_boundary_data[var_idx][bc_type]['normals'] = normals_seg
                    
                if bc_type == 'robin' and 'params' in global_boundary_data[var_idx][bc_type]:
                    segment_boundary_data[var_idx][bc_type]['params'] = global_boundary_data[var_idx][bc_type]['params']
        
        return segment_boundary_data

    @abstractmethod
    def generate_global_field(self, x_global: np.ndarray) -> np.ndarray:
        """Generate global field values
        
        Args:
            x_global: Global point coordinates
            
        Returns:
            np.ndarray: Global field values
        """
        pass

    @abstractmethod
    def generate_data(self, mode: str = "train") -> Dict:
        """Generate training/testing data
        
        Args:
            mode: Data mode ("train" or "test")
            
        Returns:
            Dict: Generated data dictionary
        """
        pass

    @abstractmethod
    def _prepare_output_dict(self, *args) -> Dict:
        """Prepare output data dictionary
        
        Args:
            *args: Variable arguments
            
        Returns:
            Dict: Prepared output dictionary
        """
        pass
