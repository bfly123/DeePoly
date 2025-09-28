import torch
from typing import Dict
from src.abstract_class.base_net import BaseNet

class FuncFittingNet(BaseNet):
    """functionFittingProblem的Neural networkImplementation"""
    
    def physics_loss(self, data_GPU: Dict, **kwargs) -> torch.Tensor:
        """ComputePhysicalLoss
        
        Args:
            data_GPU: GPUDataDictionary，IncludeTraining所需的GPUData
            data_train: TrainingDataDictionary，IncludeTraining所需的CPUData
            **kwargs: ExtraParameter
            
        Returns:
            torch.Tensor: Lossvalue
        """
        # GetTrainingData
        x_train = data_GPU["x_train"]
        u_train = data_GPU["u_train"]
        
        # GetModelPrediction
        _, output = self(x_train)
        
        # 提取Predictionvalue
        u = output[..., 0]

# auto code begin
        # Extract physical quantities from output
        u = output[..., 0]

        # L1 operators
        L1 = [u]

# auto code end
        
        # ComputeFitting误差
        fit_error = (u - u_train[..., 0]) ** 2
        fit_loss = torch.mean(fit_error)
        loss = fit_loss
        # 随机打印LossComponent（1%的Probability）
        if torch.rand(1).item() < 0.01:
            print(f"\nLoss Components:")
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

    def model_init(self):
        """InitializeModel
        
        Returns:
            FuncFittingNet: InitializeBackward的Model
        """
        # SetupDefault的Hidden layerDimensions
        model = FuncFittingNet(self.config).to(self.config.device)
        return model 