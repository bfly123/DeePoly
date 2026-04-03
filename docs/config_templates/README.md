# YAML Config Templates

当前项目建议采用“双格式支持”策略：

- 人工维护案例: `config.yaml` / `config.yml`
- 自动生成案例、sweep、中间结果: `config.json`

运行时会按以下顺序查找案例配置：

1. `config.yaml`
2. `config.yml`
3. `config.json`

注意事项：

- 同一个 case 目录内只保留一种配置文件，避免歧义。
- `method: deepoly` 表示 `spotter + snipper`。
- `method: pinns` 表示仅使用 `spotter`，Snipper 分区、多项式、线性求解参数会被忽略。
- `nn_backend` 是顶层神经网络后端开关，推荐值：
  - `torch`
  - `jax`
  - `paddle`
- 运行后会在 `results/effective_config.json` 导出归一化后的实际运行配置，便于回归和论文复现实验记录。

推荐从以下模板开始：

- [func_fitting.template.yaml](/home/bfly/workspace/computeforcfd/混合网络/DeePoly_git/docs/config_templates/func_fitting.template.yaml)
- [linear_pde.template.yaml](/home/bfly/workspace/computeforcfd/混合网络/DeePoly_git/docs/config_templates/linear_pde.template.yaml)
- [time_pde.template.yaml](/home/bfly/workspace/computeforcfd/混合网络/DeePoly_git/docs/config_templates/time_pde.template.yaml)
- [general_pde.template.yaml](/home/bfly/workspace/computeforcfd/混合网络/DeePoly_git/docs/config_templates/general_pde.template.yaml)

从现有 JSON case 迁移到 YAML：

```bash
python tools/migrate_case_config_to_yaml.py cases/linear_pde_cases/poisson_2d_sinpixsinpiy
python tools/migrate_case_config_to_yaml.py cases --recursive --case-filter Allen_Cahn
```

如果确认要切换为 YAML 主配置，并删除原来的 `config.json`：

```bash
python tools/migrate_case_config_to_yaml.py cases/nonlinear_pde_cases/time_pde_cases/Allen_Cahn/AC_equation_100_0.1 --replace-json
```
