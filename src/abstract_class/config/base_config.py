from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Tuple, List, Dict, Any
import numpy as np
import torch
import math
import json
import os
from datetime import datetime


@dataclass
class BaseConfig(ABC):
    """Base configuration class that defines the basic interface and common functionality for all configuration classes"""

    def __init__(self, case_dir: str):
        self.case_dir = case_dir

    # Runtime computed fields
    n_dim: int = field(init=False)  # Problem dimension
    x_min: np.ndarray = field(init=False)  # Minimum coordinates for each segment
    x_max: np.ndarray = field(init=False)  # Maximum coordinates for each segment
    segment_ranges: List[np.ndarray] = field(
        default_factory=list, init=False
    )  # Segment ranges for each dimension
    x_min_norm: np.ndarray = field(init=False)  # Normalized minimum coordinates
    x_max_norm: np.ndarray = field(init=False)  # Normalized maximum coordinates

    @abstractmethod
    def __post_init__(self):
        """Initialize configuration parameters, to be implemented by subclasses"""
        pass

#    @abstractmethod
#    def _auto_code(self):
#        """Auto-generate code, to be implemented by subclasses"""
#        pass

    def _init_segment_ranges(self):
        """Initialize segment ranges, supports arbitrary dimensions"""
        self.segment_ranges = [
            np.linspace(
                self.x_domain[dim][0], self.x_domain[dim][1], self.n_segments[dim] + 1
            )
            for dim in range(self.n_dim)
        ]
        self.x_domain = np.array(self.x_domain)
        self.poly_degree = np.array(self.poly_degree)

    def _init_boundaries(self):
        """Initialize boundary values, supports arbitrary dimensions"""
        # Calculate total number of segments
        Ns = math.prod(self.n_segments)

        # Initialize arrays
        self.x_min = np.zeros((Ns, self.n_dim))
        self.x_max = np.zeros((Ns, self.n_dim))
        self.x_min_norm = np.zeros((Ns, self.n_dim))
        self.x_max_norm = np.ones((Ns, self.n_dim))

        # Set boundaries for each segment
        for n in range(Ns):
            # Convert segment index n to multi-dimensional index using NumPy
            indices = np.unravel_index(n, self.n_segments, order='F')

            # Set boundaries for each dimension
            for dim in range(self.n_dim):
                self.x_min[n, dim] = self.segment_ranges[dim][indices[dim]]
                self.x_max[n, dim] = self.segment_ranges[dim][indices[dim] + 1]

    @property
    def device(self) -> str:
        """Return computation device"""
        return "cuda" if torch.cuda.is_available() else "cpu"

    def init_seed(self, seed: int = None):
        """Initialize random seed

        Args:
            seed: Random seed value, if None uses seed from config
        """
        seed = seed if seed is not None else self.seed
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True

    def export_to_json(self, filename="config.json"):
        """Export configuration to JSON file

        Args:
            filename: Output JSON filename
        """
        def serialize(obj):
            """Helper function to serialize objects that are not JSON serializable by default"""
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (np.integer, np.floating)):
                return obj.item()
            elif isinstance(obj, (tuple, list)):
                return [serialize(item) for item in obj]
            elif isinstance(obj, dict):
                return {key: serialize(value) for key, value in obj.items()}
            elif hasattr(obj, "__dict__"):
                return serialize(obj.__dict__)
            else:
                return str(obj)

        # Get all non-method and non-private attributes
        config_dict = {}
        for key, value in self.__dict__.items():
            if not key.startswith("_") and not callable(value):
                config_dict[key] = value

        # Add metadata
        config_dict["export_time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        config_dict["device_info"] = self.device

        # Serialize and save to file
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(serialize(config_dict), f, indent=2)

        print(f"Configuration exported to {filename}")

    def load_config_from_json(self, case_dir=None):
        """Load configuration from JSON file

        Args:
            case_dir: Case directory path, defaults to current directory
        """
        config_path = os.path.join(case_dir, "config.json")
        if not os.path.exists(config_path):
            return False

        try:
            with open(config_path, "r") as f:
                config_dict = json.load(f)

            # Update attributes
            for key, value in config_dict.items():
                if hasattr(self, key):
                    if isinstance(value, list):
                        # Handle integer lists
                        if key in self._int_list_fields():
                            try:
                                value = [int(v) if isinstance(v, str) else v for v in value]
                            except ValueError:
                                pass
                        # Handle special lists
                        elif key in self._list_fields():
                            value = self._process_list_field(key, value)

                    setattr(self, key, value)
            return True
            
        except Exception as e:
            print(f"Error loading configuration: {e}")
            return False

    def _int_list_fields(self):
        """List of fields that need to be converted to integers"""
        return []

    def _list_fields(self):
        """List of fields that need special handling"""
        return []

    def _process_list_field(self, key, value):
        """Process list type fields"""
        return value
