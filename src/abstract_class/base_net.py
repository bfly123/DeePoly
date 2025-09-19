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

        # Calculate actual input dimension
        actual_in = self.in_dim * 3 if use_periodic else self.in_dim

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
        """Forward propagation

        Args:
            x: Input tensor

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: (Hidden layer output, Final output)
        """
        if self.use_periodic:
            x = torch.cat(
                [x, torch.sin(2 * np.pi * x), torch.cos(2 * np.pi * x)], dim=-1
            )

        h = self.net(x)
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
        """Convert all non-null values in raw data to tensors

        Args:
            data: Raw data dictionary
            config: Configuration object for getting device information

        Returns:
            Dict: Data dictionary containing converted tensors
        """
        data_train = {}
        for key, value in data.items():
            if value is not None:
                data_train[key] = torch.tensor(
                    value, dtype=torch.float64, device=self.config.device
                )
        return data_train

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
          
          # Learning rate decay with stability mechanism
          lr_history = []  # Track loss history for stability check
          current_lr_level = 0  # 0: initial, 1: medium, 2: low
          stability_counter = 0  # Counter for stability check
          last_lr_change_epoch = 0
          
          def should_decrease_lr(loss_history, current_loss, threshold):
              """Check if we should decrease learning rate based on loss trend"""
              if len(loss_history) < 100:
                  return False
              
              # Check if loss has been above threshold for last 100 steps
              recent_losses = loss_history[-100:]
              return all(loss >= threshold for loss in recent_losses)
          
          def adjust_learning_rate(optimizer, current_loss, epoch):
              nonlocal current_lr_level, stability_counter, last_lr_change_epoch
              
              # Track loss history
              lr_history.append(current_loss)
              
              # Define thresholds and corresponding learning rates
              thresholds = [1e-2, 1e-3]
              learning_rates = [self.config.learning_rate, 0.005, 0.001]
              
              current_lr = optimizer.param_groups[0]['lr']
              new_lr = current_lr
              lr_changed = False
              
              # Check for learning rate decrease (more restrictive)
              for i, threshold in enumerate(thresholds):
                  target_level = i + 1
                  if (current_lr_level < target_level and 
                      current_loss < threshold):
                      # Immediate decrease when loss drops below threshold
                      new_lr = learning_rates[target_level]
                      current_lr_level = target_level
                      last_lr_change_epoch = epoch
                      lr_changed = True
                      break
              
              # Check for learning rate increase (with stability requirement)
              if not lr_changed:
                  for i, threshold in enumerate(reversed(thresholds)):
                      target_level = len(thresholds) - i - 1
                      if (current_lr_level > target_level and 
                          should_decrease_lr(lr_history, current_loss, threshold) and
                          epoch - last_lr_change_epoch >= 100):
                          # Only increase if loss has been above threshold for 100+ steps
                          new_lr = learning_rates[target_level]
                          current_lr_level = target_level
                          last_lr_change_epoch = epoch
                          lr_changed = True
                          break
              
              # Update learning rate if changed
              if lr_changed and abs(current_lr - new_lr) > 1e-8:
                  for param_group in optimizer.param_groups:
                      param_group['lr'] = new_lr
                  return True, new_lr
              
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
              
              if final_loss < self.config.DNNtol*self.config.dt:
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
