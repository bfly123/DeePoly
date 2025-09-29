import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Tuple, Optional
import torch.optim as optim

class BaseNet(nn.Module):
    """基础神经网络类"""

    def __init__(
        self, config, use_periodic: bool = False
    ):
        super().__init__()
        self.config = config
        self.use_periodic = use_periodic
        self.in_dim = config.n_dim
        self.hidden_dims = config.hidden_dims
        self.out_dim = config.n_eqs

        # 计算实际输入维度
        actual_in = self.in_dim * 3 if use_periodic else self.in_dim

        # 构建网络层
        layers = []
        dims = [actual_in] + list(self.hidden_dims)
        for i in range(len(dims) - 1):
            layers.extend(
                [nn.Linear(dims[i], dims[i + 1], dtype=torch.float64), nn.Tanh()]
            )

        self.net = nn.Sequential(*layers)
        self.out = nn.Linear(self.hidden_dims[-1], self.out_dim, bias=False, dtype=torch.float64)


    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """前向传播

        Args:
            x: 输入张量

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: (隐藏层输出, 最终输出)
        """
        if self.use_periodic:
            x = torch.cat(
                [x, torch.sin(2 * np.pi * x), torch.cos(2 * np.pi * x)], dim=-1
            )

        h = self.net(x)
        return h, self.out(h)

    def derivative_h(self, x: torch.Tensor) -> torch.Tensor:
        """计算隐藏层输出对输入的导数

        Args:
            x: 输入张量

        Returns:
            torch.Tensor: 导数
        """
        h, _ = self.forward(x)
        h_x = torch.autograd.grad(
            h, x, grad_outputs=torch.ones_like(h), create_graph=True
        )[0]
        return h_x

    @staticmethod
    def gradients(outputs: torch.Tensor, inputs: torch.Tensor) -> Tuple[torch.Tensor]:
        """计算梯度

        Args:
            outputs: 输出张量
            inputs: 输入张量

        Returns:
            Tuple[torch.Tensor]: 梯度
        """
        return torch.autograd.grad(
            outputs, inputs, grad_outputs=torch.ones_like(outputs), create_graph=True
        )

    def physics_loss(self, data_GPU: Dict, **kwargs) -> torch.Tensor:
        """计算物理损失

        Args:
            data_GPU: GPU数据字典，包含训练所需的GPU数据
            data_train: 训练数据字典，包含训练所需的CPU数据
            **kwargs: 额外的参数，具体实现可以根据需要添加

        Returns:
            torch.Tensor: 损失值
        """
        raise NotImplementedError("子类必须实现physics_loss方法")

    def prepare_gpu_data(data_train: Dict) -> Dict:
        """准备GPU数据

        Args:
            data_train: 训练数据
            config: 配置对象

        Returns:
            Dict: GPU数据字典
        """
        raise NotImplementedError("子类必须实现prepare_gpu_data方法")

    def prepare_train_data(self, data: Dict) -> Dict:
        """将原始数据中的所有非空值转换为tensor

        Args:
            data: 原始数据字典
            config: 配置对象，用于获取device信息

        Returns:
            Dict: 包含转换后tensor的数据字典
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
        """初始化模型

        Args:
            config: 配置对象

        Returns:
            BaseNet: 初始化后的模型
        """
        raise NotImplementedError("子类必须实现model_init方法")


    def train_net(self,data, model, data_GPU, **kwargs):
      max_retries = self.config.max_retries
      retry_count = 0
      best_loss = float("inf")
      best_model = None

      while retry_count < max_retries:
          if retry_count > 0:
              print(
                  f"重试 {retry_count}/{max_retries}，更改随机种子为 {self.config.seed + retry_count}"
              )
              torch.manual_seed(self.config.seed + retry_count)
              np.random.seed(self.config.seed + retry_count)
              #model = BaseNet.model_init(config).to(config.device)

          lr_adam = self.config.learning_rate
          optimizer_adam = optim.Adam(model.parameters(), lr=lr_adam)
          optimizer_lbfgs = optim.LBFGS(model.parameters(), lr=0.1)

          #data_train = BaseNet.prepare_train_data(data)
          model = model.to(self.config.device)

          def closure():
              optimizer_adam.zero_grad()
              loss = model.physics_loss(data_GPU, **kwargs)
              loss.backward()
              return loss

          # Adam优化
          final_loss = None
          for epoch in range(self.config.epochs_adam):
              optimizer_adam.zero_grad()
              loss = model.physics_loss(data_GPU, **kwargs)
              loss.backward()
              optimizer_adam.step()
              final_loss = loss.item()

              if final_loss < self.config.DNNtol:
                  break
              if epoch % 1000 == 0:
                  print(f" Adam Epoch {epoch}, Loss: {final_loss}")

          print(f"Final Adam Loss: {final_loss}")
  #
          # LBFGS优化
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

          # 保存最佳模型
          if final_loss < best_loss:
              best_loss = final_loss
              best_model = model.state_dict().copy()

          if final_loss < self.config.DNNtol:
              print(f"已达到目标精度 {self.config.DNNtol}，停止重试")
              break

          retry_count += 1

      if best_model is not None:
          current_model_id = id(model.state_dict())
          best_model_id = id(best_model)
          if current_model_id != best_model_id:
              model.load_state_dict(best_model)
              print(f"使用最佳模型，损失值: {best_loss}")

      return model
