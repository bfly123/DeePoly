import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Any

class LinearPDENet(nn.Module):
    """线性偏微分方程求解网络"""
    
    def __init__(self, config, activation=nn.Tanh()):
        super(LinearPDENet, self).__init__()
        
        # 配置参数
        self.config = config
        self.n_dim = config.n_dim
        self.hidden_dims = config.hidden_dims
        
        # 构建网络
        layers = []
        
        # 输入层 -> 第一个隐藏层
        layers.append(nn.Linear(self.n_dim, self.hidden_dims[0]))
        layers.append(activation)
        
        # 隐藏层
        for i in range(len(self.hidden_dims) - 1):
            layers.append(nn.Linear(self.hidden_dims[i], self.hidden_dims[i+1]))
            layers.append(activation)
        
        # 最后隐藏层 -> 输出层 (只有一个输出)
        layers.append(nn.Linear(self.hidden_dims[-1], 1))
        
        # 构建序列模型
        self.net = nn.Sequential(*layers)
        
    def forward(self, x):
        """前向传播

        Args:
            x: 输入坐标 Tensor (batch_size, n_dim)
            
        Returns:
            y: 输出值 Tensor (batch_size, 1)
        """
        return self.net(x)
    
    def prepare_gpu_data(self, data: Dict) -> Dict:
        """准备GPU数据

        Args:
            data: 输入数据字典
            
        Returns:
            gpu_data: 包含GPU张量的数据字典
        """
        gpu_data = {}
        
        # 复制所需数据到GPU
        for key, value in data.items():
            if isinstance(value, np.ndarray):
                # 将NumPy数组转换为张量并移动到GPU
                gpu_data[key] = torch.tensor(
                    value, dtype=torch.float32, device=self.config.device
                )
            elif isinstance(value, torch.Tensor):
                # 移动已有张量到GPU
                gpu_data[key] = value.to(self.config.device)
            else:
                # 保存其他类型数据
                gpu_data[key] = value
                
        return gpu_data
    
    def physics_loss(self, data_GPU: Dict) -> torch.Tensor:
        """计算物理损失函数

        Args:
            data_GPU: GPU上的数据字典
            
        Returns:
            loss: 损失值张量
        """
        # 获取输入和目标
        x = data_GPU.get('x', None)
        u = data_GPU.get('u', None)
        
        if x is None or u is None:
            raise ValueError("数据中缺少必要的输入 'x' 或目标 'u'")
        
        # 预测输出
        pred = self(x)
        
        # 计算与目标的MSE损失
        mse_loss = torch.mean((pred - u) ** 2)
        
        # 如果数据中有额外的PDE损失项，可以在这里添加
        # TODO: 添加PDE相关的损失项，如残差损失
        
        return mse_loss
    
    def train_net(self, data: Dict, model, data_GPU: Dict, epochs: int = 10000):
        """训练神经网络

        Args:
            data: 训练数据字典
            model: 神经网络模型
            data_GPU: GPU上的数据字典
            epochs: 训练轮数
            
        Returns:
            model: 训练后的模型
        """
        # 优化器
        optimizer = torch.optim.Adam(
            model.parameters(), lr=self.config.learning_rate
        )
        
        # 训练循环
        for epoch in range(epochs):
            # 设置为训练模式
            model.train()
            
            # 梯度归零
            optimizer.zero_grad()
            
            # 计算损失
            loss = model.physics_loss(data_GPU)
            
            # 反向传播
            loss.backward()
            
            # 参数更新
            optimizer.step()
            
            # 打印进度
            if (epoch + 1) % 1000 == 0:
                print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.8e}")
                
        return model 