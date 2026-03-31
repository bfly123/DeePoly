"""
Visualization Configuration Module

Provides unified configuration for visualization across all solvers.
Supports train/test data sources, mesh/scatter plotting, and output control.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Literal, Optional
from enum import Enum


class DatasetMode(str, Enum):
    """Dataset mode for visualization"""
    TRAIN_SCATTER = "train_scatter"  # Use training data with scatter plot
    TEST_MESH = "test_mesh"          # Use test data with mesh/grid plot
    BOTH = "both"                     # Generate both train and test visualizations


class ReferencePolicy(str, Enum):
    """Policy for handling reference solution"""
    AUTO = "auto"      # Show reference comparison if available, otherwise just solution
    FORCE = "force"    # Always require reference solution, error if missing
    SKIP = "skip"      # Never show reference solution, only solution field


@dataclass
class VisualizationConfig:
    """Unified visualization configuration

    Attributes:
        dataset_mode: Which dataset to use for visualization
        outputs: List of output sources to visualize (e.g., ['spotter', 'sniper'])
        reference_policy: How to handle reference solution
        realtime_visualization: Enable realtime plotting during iteration
        save_animation: Save iteration animation as GIF
        animation_skip: Frame skip interval for animation (1 = every iteration)
        figure_dpi: DPI for saved figures
        figure_format: Output figure format (png, pdf, etc.)
    """

    # Dataset and plotting mode
    dataset_mode: DatasetMode = DatasetMode.TEST_MESH

    # Output sources to visualize
    outputs: List[str] = field(default_factory=lambda: ['sniper'])

    # Reference solution handling
    reference_policy: ReferencePolicy = ReferencePolicy.AUTO

    # Animation settings
    realtime_visualization: bool = True
    save_animation: bool = True
    animation_skip: int = 10

    # Figure quality settings
    figure_dpi: int = 100
    figure_format: str = "png"

    # Advanced options
    show_training_points: bool = False  # Overlay training points on test mesh
    error_log_scale: bool = True        # Use log scale for error plots
    colormap: str = "viridis"          # Default colormap for solution plots
    plot_mode_test: Literal['auto', 'grid', 'scatter'] = "auto"
    plot_mode_train: Literal['auto', 'grid', 'scatter'] = "scatter"
    scatter_kwargs: Dict = field(default_factory=dict)
    grid_kwargs: Dict = field(default_factory=dict)

    def should_plot_train(self) -> bool:
        """Check if training data visualization is requested"""
        return self.dataset_mode in [DatasetMode.TRAIN_SCATTER, DatasetMode.BOTH]

    def should_plot_test(self) -> bool:
        """Check if test data visualization is requested"""
        return self.dataset_mode in [DatasetMode.TEST_MESH, DatasetMode.BOTH]

    def should_show_reference(self, has_reference: bool) -> bool:
        """Determine whether to show reference solution

        Args:
            has_reference: Whether reference solution is available

        Returns:
            True if reference should be shown

        Raises:
            ValueError: If policy is FORCE but no reference available
        """
        if self.reference_policy == ReferencePolicy.SKIP:
            return False
        elif self.reference_policy == ReferencePolicy.FORCE:
            if not has_reference:
                raise ValueError("Reference solution required by policy but not available")
            return True
        else:  # AUTO
            return has_reference

    @classmethod
    def from_dict(cls, config_dict: dict) -> 'VisualizationConfig':
        """Create VisualizationConfig from dictionary

        Args:
            config_dict: Dictionary with visualization settings

        Returns:
            VisualizationConfig instance
        """
        # Convert string to enum if needed
        if 'dataset_mode' in config_dict and isinstance(config_dict['dataset_mode'], str):
            config_dict['dataset_mode'] = DatasetMode(config_dict['dataset_mode'])
        if 'reference_policy' in config_dict and isinstance(config_dict['reference_policy'], str):
            config_dict['reference_policy'] = ReferencePolicy(config_dict['reference_policy'])

        # Filter only valid fields
        valid_fields = {f.name for f in cls.__dataclass_fields__.values()}
        filtered_dict = {k: v for k, v in config_dict.items() if k in valid_fields}

        return cls(**filtered_dict)

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization"""
        result = {}
        for key, value in self.__dict__.items():
            if isinstance(value, Enum):
                result[key] = value.value
            else:
                result[key] = value
        return result


@dataclass
class VisualizationRequest:
    """Request structure for unified visualization rendering

    Encapsulates all data needed for visualization in a single structure,
    making it easy to pass between solver and visualizer.

    Attributes:
        x_coords: Spatial coordinates (n_points, n_dim)
        predictions: Dictionary of predictions {'spotter': U_array, 'sniper': U_array, ...}
        reference_solution: Optional reference solution array
        data_dict: Full data dictionary (for backwards compatibility)
        metadata: Additional metadata (iteration, residual, etc.)
    """

    x_coords: 'np.ndarray'  # (n_points, n_dim)
    predictions: dict       # {'source_name': np.ndarray}
    reference_solution: Optional['np.ndarray'] = None
    data_dict: Optional[dict] = None
    metadata: Optional[dict] = field(default_factory=dict)

    def get_prediction(self, source: str) -> Optional['np.ndarray']:
        """Get prediction from specific source

        Args:
            source: Source name (e.g., 'spotter', 'sniper')

        Returns:
            Prediction array or None if source not found
        """
        return self.predictions.get(source)

    def has_reference(self) -> bool:
        """Check if reference solution is available"""
        return self.reference_solution is not None

    @property
    def n_dim(self) -> int:
        """Get spatial dimensionality"""
        return self.x_coords.shape[1] if self.x_coords.ndim > 1 else 1
