# 周期边界条件使用指南

## 概述

DeePoly现在支持周期边界条件，适用于需要在边界对应点保持函数值或导数相等的问题。

## 配置格式

### 基本结构
```json
{
  "type": "periodic",
  "region": "边界区域1", 
  "pair_with": "边界区域2",
  "constraint": "约束类型",
  "points": 边界点数量
}
```

### 约束类型
- `"dirichlet"`: 周期Dirichlet条件，要求对应边界点的函数值相等
  - u(x1) = u(x2)
- `"neumann"`: 周期Neumann条件，要求对应边界点的法向导数相等  
  - ∂u/∂n(x1) = ∂u/∂n(x2)

## 应用场景

### 1D问题
```json
"boundary_conditions": [
  {
    "type": "periodic",
    "region": "x=-1",
    "pair_with": "x=1", 
    "constraint": "dirichlet",
    "points": 20
  }
]
```

### 2D问题
```json
"boundary_conditions": [
  {
    "type": "periodic",
    "region": "x=0",
    "pair_with": "x=1", 
    "constraint": "dirichlet",
    "points": 50
  },
  {
    "type": "periodic", 
    "region": "y=0",
    "pair_with": "y=1",
    "constraint": "neumann",
    "points": 50
  }
]
```

### 3D问题
```json
"boundary_conditions": [
  {
    "type": "periodic",
    "region": "x=0", 
    "pair_with": "x=1",
    "constraint": "dirichlet",
    "points": 100
  },
  {
    "type": "periodic",
    "region": "y=0",
    "pair_with": "y=1", 
    "constraint": "dirichlet",
    "points": 100
  },
  {
    "type": "periodic",
    "region": "z=0",
    "pair_with": "z=1",
    "constraint": "neumann", 
    "points": 100
  }
]
```

## 技术实现

### 边界数据结构
周期边界条件在内部以pairs形式存储：
```python
boundary_data[var_idx]['periodic']['pairs'] = [
  {
    'region_1': 'x=-1',
    'region_2': 'x=1', 
    'x_1': numpy_array_coords_1,
    'x_2': numpy_array_coords_2,
    'constraint_type': 'dirichlet',
    'normals_1': normals_1,  # 仅Neumann需要
    'normals_2': normals_2   # 仅Neumann需要
  }
]
```

### 损失计算
- **Dirichlet周期**: loss = ||U(x1) - U(x2)||²
- **Neumann周期**: loss = ||∂U/∂n(x1) - ∂U/∂n(x2)||²

## 与其他边界条件的兼容性

周期边界条件可以与常规边界条件同时使用：
```json
"boundary_conditions": [
  {
    "type": "dirichlet",
    "region": "y=0", 
    "value": 0,
    "points": 50
  },
  {
    "type": "periodic",
    "region": "x=0",
    "pair_with": "x=1",
    "constraint": "dirichlet", 
    "points": 50
  }
]
```

## 注意事项

1. 配对区域必须具有相同的几何形状和点数
2. Neumann周期条件需要正确的法向量计算
3. 高维问题中可以在不同方向上应用不同的周期条件
4. 周期边界条件会自动应用到所有变量分量（vars_list中的所有变量）

## 物理应用示例

- 流体力学中的周期性流动
- 材料科学中的晶格结构
- 波动方程的周期边界
- 传热问题的周期性边界条件