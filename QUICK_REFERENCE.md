# General PDE Solver - 快速参考

## 🚀 快速开始

```bash
# 运行求解器
python -m src.problem_solvers.general_pde_solver.solver \
    --case_path cases/general_pde_cases/AC_equation

# 查看结果
ls results/
```

## 📋 关键概念

- **t**: 空间维度 (与x一起构成2D空间)
- **τ**: 伪时间 (用于迭代)
- **Spotter**: 初始化阶段 (神经网络训练)
- **Fitter**: 求解阶段 (伪时间步迭代)

## 🔧 核心文件

| 文件 | 作用 |
|------|------|
| `config.json` | 问题配置 |
| `solver.py` | 主求解器 |
| `core/spotter_init.py` | SpotterNet (初始化) |
| `core/fitter.py` | GeneralPDEFitter (求解) |
| `core/net.py` | GeneralPDENet (网络) |

## ✅ 验证清单

- [ ] 边界条件正确: 3006点
- [ ] 网络迁移零误差: < 1e-12
- [ ] PDE方程完整: 包含 ∂u/∂t
- [ ] 残差下降: > 100倍

## 📊 性能指标

```
Spotter: ~37s (1000 epochs)
Fitter: ~120s (~77 iterations)
总计: < 3分钟
```

## 🐛 常见问题

**Q: 边界条件未施加?**  
A: 检查config.json中region格式 (用"left"/"right"/"bottom")

**Q: 网络迁移失败?**  
A: 确保SpotterNet和GeneralPDENet都继承BaseNet

**Q: 残差停滞?**  
A: 调整dt参数或改善矩阵条件数

## 📚 文档索引

- 完整实现: `IMPLEMENTATION_SUMMARY.md`
- 测试结果: `TEST_RESULTS.md`
- 重构说明: `REFACTORING_SUMMARY.md`
- 最终总结: `FINAL_SUMMARY.md`

