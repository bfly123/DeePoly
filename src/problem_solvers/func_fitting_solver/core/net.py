import torch
from typing import Dict
from src.abstract_class.base_net import BaseNet

class FuncFittingNet(BaseNet):
    """functionFittingProblem的Neural networkImplementation"""
    
    def physics_loss(self, data_GPU: Dict, **kwargs) -> torch.Tensor:
        """Use all configured output variables in the fitting loss."""
        x_train = data_GPU["x_train"]
        u_train = data_GPU["u_train"]

        _, output = self(x_train)

        fit_error = (output - u_train) ** 2
        fit_loss = torch.mean(fit_error)
        loss = fit_loss

        if torch.rand(1).item() < 0.01:
            print("\nLoss Components:")
            for eq_idx in range(output.shape[-1]):
                eq_loss = torch.mean(fit_error[..., eq_idx]).item()
                print(f"Variable {eq_idx} Loss: {eq_loss:.8f}")
            print(f"Fitting Loss: {fit_loss.item():.8f}")
            print(f"Total Loss: {loss.item():.8f}")
        return loss
    
    def prepare_gpu_data(self, data_train: Dict) -> Dict:
      return {
        "x_train": torch.tensor(
            data_train["x"],
            dtype=torch.float64,
            requires_grad=True,
            device=self.config.device,
        ),
        "u_train": torch.tensor(
            data_train["u"],
            dtype=torch.float64,
            requires_grad=True,
            device=self.config.device,
        ),
    }

    @staticmethod
    def model_init(config):
        """Initialize model for first-run and retry paths."""
        model = FuncFittingNet(config=config).to(config.device)
        return model
