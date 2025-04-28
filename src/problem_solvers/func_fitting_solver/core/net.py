import torch
from typing import Dict
from abstract_class.base_net import BaseNet

class FuncFittingNet(BaseNet):
    """函数拟合问题的神经网络实现"""
    
    def physics_loss(self, data_GPU: Dict, **kwargs) -> torch.Tensor:
        """计算物理损失
        
        Args:
            data_GPU: GPU数据字典，包含训练所需的GPU数据
            data_train: 训练数据字典，包含训练所需的CPU数据
            **kwargs: 额外参数
            
        Returns:
            torch.Tensor: 损失值
        """
        # 获取训练数据
        x_train = data_GPU["x_train"]
        u_train = data_GPU["u_train"]
        
        # 获取模型预测
        _, output = self(x_train)
        
        # 提取预测值
        u = output[..., 0]
        
        # 获取真实值
        
        # 计算拟合误差
        fit_error = (u - u_train[..., 0]) ** 2
        fit_loss = torch.mean(fit_error)
        loss = fit_loss
        # 随机打印损失组件（1%的概率）
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
        """初始化模型
        
        Returns:
            FuncFittingNet: 初始化后的模型
        """
        # 设置默认的隐藏层维度
        model = FuncFittingNet(self.config).to(self.config.device)
        return model 