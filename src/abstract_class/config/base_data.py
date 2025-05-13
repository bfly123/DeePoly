import numpy as np
import torch
import os
import importlib.util
import sys
from typing import Dict, List, Tuple, Optional
from abc import ABC, abstractmethod

class BaseDataGenerator(ABC):
    """Base data generator class"""
    
    def __init__(self, config):
        self.config = config
        self.Ns = np.prod(config.n_segments)  # Total number of segments
        self.Nw = config.points_per_swap  # Number of boundary points per swap
        self.n_dim = config.n_dim
        self.n_eqs = config.n_eqs
        self.case_dir = getattr(config, "case_dir", None)
        # Try to load custom data generator module
        #self.custom_data_generator = self._load_custom_data_generator()

    def _load_custom_data_generator(self):
        """Load data_generate.py module from the case directory"""
        if not self.case_dir:
            print("Case path not set, using default data generation method")
            return None

        data_generate_path = os.path.join(self.case_dir, "data_generate.py")

        if not os.path.exists(data_generate_path):
            print(f"Custom data generator module not found: {data_generate_path}, using default data generation method")
            return None

        # Add case directory to Python path
        if self.case_dir not in sys.path:
            sys.path.insert(0, self.case_dir)
            
        try:
            return importlib.import_module("data_generate")
        except ImportError as e:
            print(f"Error importing data_generate module: {e}")
            return None

    def _load_source_term(self, x_global: np.ndarray) -> Optional[callable]:
        """Load source term function from data_generate.py in the case directory
        Returns:
            function: Source term function or None if not found
        """
        config = self.config
        if hasattr(config, "source_term"):
            if isinstance(config.source_term, str):
                source_term_func = self._parse_source_term(config.source_term)
                return source_term_func(*[x_global[:, i] for i in range(x_global.shape[1])])

            elif config.source_term is True:
                if hasattr(self, "custom_data_generator") and self.custom_data_generator and hasattr(
                    self.custom_data_generator, "generate_source_term"
                ):
                    return self.custom_data_generator.generate_source_term(x_global)
                else:
                    raise ValueError(
                        "Custom data generator not found, or the custom data generator does not implement generate_source_term method"
                    )
        else:
            raise ValueError("Source term function not found in config")

    def _parse_source_term(self, source_expr: str) -> callable:
        """Parse source term expression

        Args:
            source_expr: Source term expression string, e.g., "sin(pi*x)*sin(pi*y)"

        Returns:
            source_func: Source term function
        """
        # Import necessary modules
        import numpy as np
        from math import sin, cos, exp, pi, sqrt

        # Create a namespace with all required functions
        namespace = {
            'sin': np.sin,
            'cos': np.cos,
            'exp': np.exp,
            'pi': np.pi,
            'sqrt': np.sqrt,
            'np': np
        }

        # Build function expression
        var_names = ", ".join(self.config.spatial_vars)
        func_str = f"lambda {var_names}: {source_expr}"

        try:
            # Compile function with the namespace
            source_func = eval(func_str, namespace)
            return source_func
        except Exception as e:
            print(f"Error parsing source term expression: {e}")
            # Return default zero function
            return lambda *args: 0.0

    @abstractmethod
    def generate_global_field(self, x_global: np.ndarray) -> np.ndarray:
        """Generate global initial field"""
        pass

    @abstractmethod
    def generate_data(self, mode: str = "train") -> Dict:
        """Generate training/testing data"""
        pass
        
    def read_boundary_conditions(self) -> Dict:
        """Read boundary condition configuration"""
        boundary_dict = {}
        
        # Check if boundary conditions exist
        if not hasattr(self.config, 'boundary_conditions') or not self.config.boundary_conditions:
            return boundary_dict
            
        # Initialize boundary condition dictionary for each variable
        for var in self.config.vars_list:
            boundary_dict[var] = {
                'dirichlet': {
                    'x': [],
                    'u': []
                },
                'neumann': {
                    'x': [],
                    'u': [],
                    'normals': []
                },
                'robin': {
                    'x': [],
                    'u': [],
                    'params': [],
                    'normals': []
                }
            }
        
        # Process each boundary condition
        for bc in self.config.boundary_conditions:
            region = bc['region']
            bc_type = bc['type'].lower()
            value = bc['value']
            points = bc['points']
            
            # Generate points for boundary region
            x_boundary = self._generate_boundary_points(region, points)
            
            # Get normal vectors for the region
            normals = self._get_boundary_normals(region, x_boundary.shape[0])
            
            # Apply boundary conditions to all variables
            for var in self.config.vars_list:
                if bc_type == 'dirichlet':
                    # For Dirichlet conditions, add points and values directly
                    boundary_dict[var]['dirichlet']['x'].append(x_boundary)
                    # Create value array of corresponding size
                    u_values = np.ones((x_boundary.shape[0], 1)) * value
                    boundary_dict[var]['dirichlet']['u'].append(u_values)
                
                elif bc_type == 'neumann':
                    boundary_dict[var]['neumann']['x'].append(x_boundary)
                    u_values = np.ones((x_boundary.shape[0], 1)) * value
                    boundary_dict[var]['neumann']['u'].append(u_values)
                    boundary_dict[var]['neumann']['normals'].append(normals)
                
                elif bc_type == 'robin':
                    boundary_dict[var]['robin']['x'].append(x_boundary)
                    u_values = np.ones((x_boundary.shape[0], 1)) * value
                    boundary_dict[var]['robin']['u'].append(u_values)
                    boundary_dict[var]['robin']['normals'].append(normals)
                    # Add Robin boundary condition parameters
                    params = bc.get('params', [1.0, 0.0])  # Default parameters
                    boundary_dict[var]['robin']['params'].append(params)
        
        # Combine boundary points and values for each variable
        for var in boundary_dict:
            for bc_type in boundary_dict[var]:
                if boundary_dict[var][bc_type]['x']:
                    boundary_dict[var][bc_type]['x'] = np.vstack(boundary_dict[var][bc_type]['x'])
                    boundary_dict[var][bc_type]['u'] = np.vstack(boundary_dict[var][bc_type]['u'])
                    
                    # Also combine normal vectors if they exist
                    if bc_type in ['neumann', 'robin'] and boundary_dict[var][bc_type]['normals']:
                        boundary_dict[var][bc_type]['normals'] = np.vstack(boundary_dict[var][bc_type]['normals'])
                else:
                    boundary_dict[var][bc_type]['x'] = np.array([])
                    boundary_dict[var][bc_type]['u'] = np.array([])
                    if bc_type in ['neumann', 'robin']:
                        boundary_dict[var][bc_type]['normals'] = np.array([])
        
        return boundary_dict
    
    def _generate_boundary_points(self, region: str, points: int) -> np.ndarray:
        """Generate points for specified boundary region - supports arbitrary dimensions"""
        x_domain = self.config.x_domain
        
        # Generic boundary generation for arbitrary dimensions
        if region.startswith('dim'):
            # Format: 'dim{d}_min' or 'dim{d}_max' where d is dimension index
            try:
                parts = region.split('_')
                dim_idx = int(parts[0][3:])  # Extract dimension index from 'dim{d}'
                is_max = parts[1] == 'max'   # Determine if this is max boundary
                
                # Create points in all dimensions, with fixed value in the boundary dimension
                boundary_value = x_domain[dim_idx][1] if is_max else x_domain[dim_idx][0]
                
                # Initialize points array
                boundary_points = np.zeros((points, self.n_dim))
                boundary_points[:, dim_idx] = boundary_value
                
                # Fill other dimensions with random or uniform points
                for d in range(self.n_dim):
                    if d != dim_idx:
                        boundary_points[:, d] = np.linspace(
                            x_domain[d][0], x_domain[d][1], points
                        )
                
                return boundary_points
            except (ValueError, IndexError) as e:
                print(f"Error parsing boundary region {region}: {e}")
                return np.array([])
        
        # Legacy support for named boundaries in 1D, 2D, 3D
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
            
            if region == 'left' or region == 'right':
                x_val = x_min if region == 'left' else x_max
                y_coords = np.linspace(y_min, y_max, points_per_dim)
                z_coords = np.linspace(z_min, z_max, points_per_dim)
                y_grid, z_grid = np.meshgrid(y_coords, z_coords)
                return np.column_stack((
                    np.ones(points_per_dim**2) * x_val,
                    y_grid.flatten(),
                    z_grid.flatten()
                ))
            # Other 3D boundary conditions can be extended as needed
            
        print(f"Warning: Boundary region '{region}' not supported for {self.n_dim}D")
        return np.array([])

    def _get_boundary_normals(self, region: str, num_points: int) -> np.ndarray:
        """Get normal vectors for boundary - supports arbitrary dimensions"""
        # Initialize normal vectors as zero vectors
        normals = np.zeros((num_points, self.n_dim))
        
        # Generic normal vector generation for arbitrary dimensions
        if region.startswith('dim'):
            try:
                parts = region.split('_')
                dim_idx = int(parts[0][3:])  # Extract dimension index from 'dim{d}'
                is_max = parts[1] == 'max'   # Determine if this is max boundary
                
                # Set normal vector component in the boundary dimension
                normals[:, dim_idx] = 1.0 if is_max else -1.0
                return normals
            except (ValueError, IndexError):
                pass
        
        # Legacy support for named boundaries in 1D, 2D, 3D
        if self.n_dim == 1:
            if region == 'left':
                normals[:, 0] = -1.0  # Left boundary normal points outward (negative)
            elif region == 'right':
                normals[:, 0] = 1.0   # Right boundary normal points outward (positive)
            
        elif self.n_dim == 2:
            if region == 'left':
                normals[:, 0] = -1.0  # Normal vector pointing left (-1, 0)
            elif region == 'right':
                normals[:, 0] = 1.0   # Normal vector pointing right (1, 0)
            elif region == 'bottom':
                normals[:, 1] = -1.0  # Normal vector pointing down (0, -1)
            elif region == 'top':
                normals[:, 1] = 1.0   # Normal vector pointing up (0, 1)
            
        elif self.n_dim == 3:
            if region == 'left':
                normals[:, 0] = -1.0  # (-1, 0, 0)
            elif region == 'right':
                normals[:, 0] = 1.0   # (1, 0, 0)
            elif region == 'bottom':
                normals[:, 1] = -1.0  # (0, -1, 0)
            elif region == 'top':
                normals[:, 1] = 1.0   # (0, 1, 0)
            elif region == 'back':
                normals[:, 2] = -1.0  # (0, 0, -1)
            elif region == 'front':
                normals[:, 2] = 1.0   # (0, 0, 1)
        
        return normals
    def split_global_points(self, x_global: np.ndarray) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """Split global points into local segments
        
        Args:
            x_global: Global point coordinates
            
        Returns:
            Tuple containing a list of segmented points and a list of masks
        """
        x_segments = []
        masks = []
        
        for n in range(self.Ns):
            mask = self._create_segment_mask(x_global, n)
            x_seg = x_global[mask]
            x_segments.append(x_seg)
            masks.append(mask)
            
        return x_segments, masks

    def split_global_field(self, masks: List[np.ndarray], u_global: np.ndarray) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """Split global field into local segments"""
        u_segments = []
        for n in range(self.Ns):
            mask = masks[n]
            u_seg = u_global[mask]
            u_segments.append(u_seg)
        return u_segments

    def _generate_global_points(self, mode: str) -> np.ndarray:
        """Generate global sampling points - supports arbitrary dimensions"""
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

    def _create_segment_mask(self, x: np.ndarray, i: int) -> np.ndarray:
        """Create segment mask - supports arbitrary dimensions"""
        x_min, x_max = self.config.x_min[i], self.config.x_max[i]
        masks = []
        for j in range(self.n_dim):
            masks.append(x[:, j] > x_min[j])
            masks.append(x[:, j] <= x_max[j])
        
        main_mask = np.logical_and.reduce(masks)
        
        if i == 0:
            # For the first segment, include points exactly on the lower boundary
            boundary_mask = np.logical_or.reduce([
                np.isclose(x[:, j], x_min[j]) for j in range(self.n_dim)
            ])
            return np.logical_or(main_mask, boundary_mask)
        return main_mask

    def _normalize_data(self, x: np.ndarray, x_min: np.ndarray, x_max: np.ndarray) -> np.ndarray:
        """Normalize data - supports arbitrary dimensions"""
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

    def _process_segments(self, x_segments: List[np.ndarray], global_boundary_dict: Dict) -> Tuple[np.ndarray, np.ndarray, List[np.ndarray], List[Dict]]:
        """Process data for each segment"""
        x_segments_norm = []
        x_swap = np.zeros((self.Ns, 2*self.n_dim, self.Nw, self.n_dim), dtype=np.float64)
        x_swap_norm = np.zeros_like(x_swap)
        
        boundary_segments_dict = []
        
        for n in range(self.Ns):
            # Process segment data normalization
            x_segments_norm.append(self._normalize_data(x_segments[n], self.config.x_min[n], self.config.x_max[n]))
            
            # Process swap points
            if self.config.points_per_swap > 0:
                x_swap_n = self._generate_swap_points(n)
                x_swap[n] = x_swap_n
                x_swap_norm[n] = self._normalize_swap_data(x_swap_n, n)
            
            # Process boundary conditions for current segment
            segment_boundary_dict = self._process_segment_boundary(global_boundary_dict, n)
            boundary_segments_dict.append(segment_boundary_dict)
        
        return x_swap, x_swap_norm, x_segments_norm, boundary_segments_dict
    
    def _generate_swap_points(self, n: int) -> np.ndarray:
        """Generate swap points - supports arbitrary dimensions"""
        x_min, x_max = self.config.x_min, self.config.x_max
        x_swap = np.zeros((2*self.n_dim, self.Nw, self.n_dim))
        
        # Generate boundary points for each dimension
        for i in range(self.n_dim):
            # Lower boundary
            x_swap[2*i] = self._generate_swap_boundary_points(x_min[n], x_max[n], i, 0)
            # Upper boundary
            x_swap[2*i+1] = self._generate_swap_boundary_points(x_min[n], x_max[n], i, 1)
            
        return x_swap

    def _generate_swap_boundary_points(self, x_min: np.ndarray, x_max: np.ndarray, dim: int, is_upper: int) -> np.ndarray:
        """Generate swap boundary points for a single dimension"""
        points = np.zeros((self.Nw, self.n_dim))
        points[:, dim] = x_max[dim] if is_upper else x_min[dim]
        
        # Distribute points uniformly along other dimensions
        for i in range(self.n_dim):
            if i != dim:
                points[:, i] = np.linspace(x_min[i], x_max[i], self.Nw)
                
        return points

    def _normalize_swap_data(self, x_swap: np.ndarray, n: int) -> np.ndarray:
        """Normalize swap point data"""
        x_swap_norm = np.zeros_like(x_swap)
        for j in range(x_swap.shape[0]):
            for k in range(x_swap.shape[1]):
                x_swap_norm[j, k] = self._normalize_data(
                    x_swap[j, k], self.config.x_min[n], self.config.x_max[n]
                )
        return x_swap_norm
    
    def _process_segment_boundary(self, global_boundary_dict: Dict, segment_idx: int) -> Dict:
        """Process boundary conditions for a single segment"""
        # Initialize boundary condition dictionary for this segment
        segment_boundary_dict = {}
        x_min, x_max = self.config.x_min, self.config.x_max
        
        # Process each variable
        for var in global_boundary_dict:
            segment_boundary_dict[var] = {
                'dirichlet': {'x': [], 'u': []},
                'neumann': {'x': [], 'u': [], 'normals': []},
                'robin': {'x': [], 'u': [], 'params': [], 'normals': []}
            }
            
            # Process different types of boundary conditions
            for bc_type in global_boundary_dict[var]:
                if len(global_boundary_dict[var][bc_type]['x']) == 0:
                    continue
                    
                # Find boundary points within the current segment
                x_boundary = global_boundary_dict[var][bc_type]['x']
                mask = self._create_segment_mask(x_boundary, segment_idx)
                
                if not np.any(mask):
                    continue
                    
                # Extract boundary points and values for this segment
                x_seg = x_boundary[mask]
                u_seg = global_boundary_dict[var][bc_type]['u'][mask]
                
                # Normalize coordinates
                x_seg_norm = self._normalize_data(x_seg, x_min[segment_idx], x_max[segment_idx])
                
                # Save to the segment boundary condition dictionary
                segment_boundary_dict[var][bc_type]['x'] = x_seg_norm
                segment_boundary_dict[var][bc_type]['u'] = u_seg
                
                # Process normal vectors (if present)
                if bc_type in ['neumann', 'robin'] and 'normals' in global_boundary_dict[var][bc_type]:
                    normals_seg = global_boundary_dict[var][bc_type]['normals'][mask]
                    segment_boundary_dict[var][bc_type]['normals'] = normals_seg
                    
                # Process Robin parameters (if present)
                if bc_type == 'robin' and 'params' in global_boundary_dict[var][bc_type]:
                    segment_boundary_dict[var][bc_type]['params'] = global_boundary_dict[var][bc_type]['params']
        
        return segment_boundary_dict

    @abstractmethod
    def _prepare_output_dict(self, *args) -> Dict:
        """Prepare output data dictionary - to be implemented by subclasses"""
        pass
