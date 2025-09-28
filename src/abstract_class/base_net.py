import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Tuple, Optional
import torch.optim as optim

class BaseNet(nn.Module):
    """Base neural network class"""

    def __init__(
        self, config, use_periodic: bool = False
    ):
        super().__init__()
        self.config = config
        self.use_periodic = use_periodic
        self.in_dim = config.n_dim
        self.hidden_dims = config.hidden_dims
        self.out_dim = config.n_eqs

        # Calculate input dimension - unified approach
        periodic_multiplier = 3 if use_periodic else 1
        actual_in = self.in_dim * periodic_multiplier

        # Build network layers
        layers = []
        dims = [actual_in] + list(self.hidden_dims)
        for i in range(len(dims) - 1):
            layers.extend(
                [nn.Linear(dims[i], dims[i + 1], dtype=torch.float64), nn.Tanh()]
            )

        self.net = nn.Sequential(*layers)
        self.out = nn.Linear(self.hidden_dims[-1], self.out_dim, bias=False, dtype=torch.float64)


    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward propagation with unified periodic handling

        Args:
            x: Input tensor

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: (Hidden layer output, Final output)
        """
        # Apply periodic features if enabled - no branching in computation
        features = [x]
        if self.use_periodic:
            features.extend([
                torch.sin(2 * np.pi * x),
                torch.cos(2 * np.pi * x)
            ])

        x_processed = torch.cat(features, dim=-1) if len(features) > 1 else x
        h = self.net(x_processed)
        return h, self.out(h)

    def derivative_h(self, x: torch.Tensor) -> torch.Tensor:
        """Calculate derivative of hidden layer output with respect to input

        Args:
            x: Input tensor

        Returns:
            torch.Tensor: Derivative
        """
        h, _ = self.forward(x)
        h_x = torch.autograd.grad(
            h, x, grad_outputs=torch.ones_like(h), create_graph=True
        )[0]
        return h_x

    @staticmethod
    def gradients(outputs: torch.Tensor, inputs: torch.Tensor) -> Tuple[torch.Tensor]:
        """Calculate gradients

        Args:
            outputs: Output tensor
            inputs: Input tensor

        Returns:
            Tuple[torch.Tensor]: Gradients
        """
        return torch.autograd.grad(
            outputs, inputs, grad_outputs=torch.ones_like(outputs), create_graph=True
        )

    def physics_loss(self, data_GPU: Dict, **kwargs) -> torch.Tensor:
        """Calculate physics loss

        Args:
            data_GPU: GPU data dictionary containing GPU data required for training
            data_train: Training data dictionary containing CPU data required for training
            **kwargs: Additional parameters, specific implementations can add as needed

        Returns:
            torch.Tensor: Loss value
        """
        raise NotImplementedError("Subclasses must implement physics_loss method")

    def prepare_gpu_data(data_train: Dict) -> Dict:
        """Prepare GPU data

        Args:
            data_train: Training data
            config: Configuration object

        Returns:
            Dict: GPU data dictionary
        """
        raise NotImplementedError("Subclasses must implement prepare_gpu_data method")

    def prepare_train_data(self, data: Dict) -> Dict:
        """Convert data to tensors with unified processing - eliminate None checking branches

        Args:
            data: Raw data dictionary

        Returns:
            Dict: Data dictionary with converted tensors
        """
        return {
            key: torch.tensor(value, dtype=torch.float64, device=self.config.device)
            for key, value in data.items()
            if value is not None
        }

    @staticmethod
    def model_init():
        """Initialize model

        Args:
            config: Configuration object

        Returns:
            BaseNet: Initialized model
        """
        raise NotImplementedError("Subclasses must implement model_init method")


    def train_net(self,data, model, data_GPU, **kwargs):
      max_retries = self.config.max_retries
      retry_count = 0
      best_loss = float("inf")
      best_model = None

      while retry_count < max_retries:
          if retry_count > 0:
              print(
                  f"Retry {retry_count}/{max_retries}, changing random seed to {self.config.seed + retry_count}"
              )
              torch.manual_seed(self.config.seed + retry_count)
              np.random.seed(self.config.seed + retry_count)
              #model = BaseNet.model_init(config).to(config.device)

          lr_adam = self.config.learning_rate
          optimizer_adam = optim.Adam(model.parameters(), lr=lr_adam)
          optimizer_lbfgs = optim.LBFGS(model.parameters(), lr=0.1)
          
          # Simplified learning rate adjustment - eliminate complex branching
          lr_thresholds = [1e-2, 1e-3]
          lr_values = [self.config.learning_rate, 0.005, 0.001]
          lr_history = []

          def adjust_learning_rate(optimizer, current_loss, epoch):
              lr_history.append(current_loss)
              current_lr = optimizer.param_groups[0]['lr']

              # Simple threshold-based adjustment
              target_lr = current_lr
              for threshold, lr_val in zip(lr_thresholds, lr_values[1:]):
                  if current_loss < threshold:
                      target_lr = lr_val
                      break

              # Apply change if significant difference
              if abs(current_lr - target_lr) > 1e-8:
                  for param_group in optimizer.param_groups:
                      param_group['lr'] = target_lr
                  return True, target_lr
              return False, current_lr

          #data_train = BaseNet.prepare_train_data(data)
          model = model.to(self.config.device)

          def closure():
              optimizer_adam.zero_grad()
              loss = model.physics_loss(data_GPU, **kwargs)
              loss.backward()
              return loss

          # Adam optimization
          final_loss = None
          for epoch in range(self.config.epochs_adam):
              optimizer_adam.zero_grad()
              loss = model.physics_loss(data_GPU, **kwargs)
              loss.backward()
              optimizer_adam.step()
              final_loss = loss.item()
              
              # Adjust learning rate based on loss
              lr_changed, current_lr = adjust_learning_rate(optimizer_adam, final_loss, epoch)
              
              # Use dt for time-dependent PDEs, otherwise just use DNNtol
              tolerance = self.config.DNNtol * getattr(self.config, 'dt', 1.0)
              if final_loss < tolerance:
                  break
              if epoch % 1000 == 0:
                  print(f" Adam Epoch {epoch}, Loss: {final_loss:.6e}, LR: {current_lr:.6e}")
              if lr_changed:
                  print(f"   Learning rate adjusted to {current_lr:.6e} at epoch {epoch}, loss {final_loss:.6e}")

          print(f"Final Adam Loss: {final_loss}")
  #
          # LBFGS optimization
          if self.config.epochs_lbfgs > 0:
              for epoch in range(self.config.epochs_lbfgs):
                  optimizer_lbfgs.zero_grad()
                  loss = model.physics_loss(data_GPU, **kwargs)
                  loss.backward()
                  optimizer_lbfgs.step(closure)
                  final_loss = loss.item()

                  if final_loss < self.config.DNNtol:
                      break
                  if epoch % 100 == 0:
                      print(f"LBFGS Epoch {epoch}, Loss: {final_loss}")

          # Save best model
          if final_loss < best_loss:
              best_loss = final_loss
              best_model = model.state_dict().copy()

          if final_loss < self.config.DNNtol:
              print(f"Target accuracy {self.config.DNNtol} achieved, stopping retries")
              break

          retry_count += 1

      if best_model is not None:
          current_model_id = id(model.state_dict())
          best_model_id = id(best_model)
          if current_model_id != best_model_id:
              model.load_state_dict(best_model)
              print(f"Using best model, loss value: {best_loss}")

      return model
