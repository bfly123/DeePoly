# DeePoly CLI 安装与使用

先进入你要运行 DeePoly 的 Python 环境。当前项目建议使用 `ML`：

```bash
source /home/bfly/anaconda3/etc/profile.d/conda.sh
conda activate ML
```

然后在仓库根目录执行：

```bash
bash install.sh
```

或直接：

```bash
python -m pip install -e .
```

直接执行 `python -m pip install -e .` 可以安装，但不会执行 `install.sh` 里的环境检查。

`install.sh` 只会安装到你当前指定的 Python 环境，不会主动切换环境。

安装前会检查：

- Python 版本是否满足 `>= 3.10`
- 当前解释器里是否可用 `pip`
- 当前环境是否已经安装 DeePoly 运行时依赖：`numpy`、`scipy`、`matplotlib`、`sympy`、`torch`
- 安装后当前环境的脚本目录里是否生成了 `deepoly`

如果缺依赖，脚本会直接报错并停止，由你决定在当前环境里补装依赖后再执行。

如果要显式指定解释器，也可以：

```bash
PYTHON_BIN=/home/bfly/anaconda3/envs/ML/bin/python bash install.sh
```

安装完成后，可以在任意包含 `config.json` 的 case 目录中执行：

```bash
deepoly .
```

也可以显式指定 case 目录：

```bash
deepoly cases/nonlinear_pde_cases/pseudotime_pde_cases/lid_driven_cavity_re100
```

常用可选参数：

```bash
deepoly . --device auto --linear-device cpu
deepoly . --results-dir ./results_cli
deepoly . --skip-autocode
```
