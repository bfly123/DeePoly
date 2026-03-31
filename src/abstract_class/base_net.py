import numpy as np
import torch
import torch.nn as nn
from typing import Dict, Tuple

from .base_net_data_mixin import BaseNetDataMixin
from .base_net_training_mixin import BaseNetTrainingMixin
from .constants import pi


class BaseNet(BaseNetTrainingMixin, BaseNetDataMixin, nn.Module):
    """Base neural network class"""

    def __init__(self, config, use_periodic: bool = False):
        super().__init__()
        self.config = config
        self.use_periodic = use_periodic
        self.in_dim = config.n_dim
        self.hidden_dims = config.hidden_dims
        self.out_dim = config.n_eqs

        periodic_multiplier = 3 if use_periodic else 1
        actual_in = self.in_dim * periodic_multiplier
        layers = []
        dims = [actual_in] + list(self.hidden_dims)
        for i in range(len(dims) - 1):
            layers.extend([nn.Linear(dims[i], dims[i + 1], dtype=torch.float64), nn.Tanh()])
        self.net = nn.Sequential(*layers)
        self.out = nn.Linear(self.hidden_dims[-1], self.out_dim, bias=False, dtype=torch.float64)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward propagation with unified periodic handling"""
        features = [x]
        if self.use_periodic:
            features.extend([torch.sin(2 * np.pi * x), torch.cos(2 * np.pi * x)])
        x_processed = torch.cat(features, dim=-1) if len(features) > 1 else x
        h = self.net(x_processed)
        return h, self.out(h)

    def derivative_h(self, x: torch.Tensor) -> torch.Tensor:
        """Calculate derivative of hidden layer output with respect to input"""
        h, _ = self.forward(x)
        return torch.autograd.grad(h, x, grad_outputs=torch.ones_like(h), create_graph=True)[0]

    @staticmethod
    def gradients(outputs: torch.Tensor, inputs: torch.Tensor) -> Tuple[torch.Tensor]:
        """Calculate gradients"""
        return torch.autograd.grad(outputs, inputs, grad_outputs=torch.ones_like(outputs), create_graph=True)

    def compute_pde_operators(self, U: torch.Tensor, x_train: torch.Tensor) -> Dict[str, list]:
        """Compute PDE operators - unified auto code generation area"""

# auto code begin
# Config signature: {"F": [], "L1": ["-0.0001*diff(u,x,2)"], "L2": [], "N": ["5*u**3-5*u"], "S": [], "const_list": []}
        # Extract physical quantities from output
        u = U[..., 0]

        # Calculate derivatives in each direction
        du_x = self.gradients(u, x_train)[0][..., 0]

        # Calculate 2nd-order derivatives
        du_xx = self.gradients(du_x, x_train)[0][..., 0]

        # L1 operators
        L1 = [-0.0001*du_xx]

        # L2 operators
        L2 = []

        # F operators
        F = []

        # N operators
        N = [5*u**3-5*u]

        # S operators (source term)
        S = [0]

# auto code end

        return {"L1": L1, "L2": L2, "F": F, "N": N, "S": S}

    def physics_loss(self, data_GPU: Dict, **kwargs) -> torch.Tensor:
        """Calculate physics loss"""
        raise NotImplementedError("Subclasses must implement physics_loss method")

    def pad_operator_outputs(self, ops: Dict[str, list]) -> Dict[str, list]:
        """Pad operator lists to ``n_eqs`` so empty optional operators remain valid."""
        n_eqs = int(getattr(self.config, "n_eqs", 1))
        zero_template = None
        for key in ("L1", "L2", "F", "N", "S"):
            values = ops.get(key, [])
            if values:
                zero_template = torch.zeros_like(values[0])
                break
        if zero_template is None:
            raise ValueError("At least one operator output is required to infer tensor shapes")

        padded = {}
        for key in ("L1", "L2", "F", "N", "S"):
            values = list(ops.get(key, []))
            if len(values) < n_eqs:
                values.extend([torch.zeros_like(zero_template) for _ in range(n_eqs - len(values))])
            padded[key] = values[:n_eqs]
        return padded

    @classmethod
    def model_init(cls, config):
        """Initialize model for retry paths unless a subclass overrides it."""
        return cls(config=config).to(config.device)
