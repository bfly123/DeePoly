import argparse
import json
import os
import sys

# 直接指定src目录的绝对路径
src_dir = "/home/bfly/workspace/computeforcfd/混合网络/算例整合2/src"

# 添加到Python路径
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)
    
# 添加项目根目录到路径
project_root = os.path.dirname(src_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# 使用简化后的导入
from problem_solvers import TimePDESolver, FuncFittingSolver, LinearPDESolver
from problem_solvers.time_pde_solver.utils import TimePDEConfig
from problem_solvers.func_fitting_solver.utils import FuncFittingConfig
from problem_solvers.linear_pde_solver.utils import LinearPDEConfig


def main():
    # 调试模式：直接指定测试目录
    DEBUG_MODE = False
    
    if DEBUG_MODE:
        # 直接指定测试案例路径
        case_dir = "/home/bfly/workspace/computeforcfd/混合网络/算例整合2/cases/func_fitting_cases/discontinuous_case1"
    else:
        # 从命令行获取参数
        parser = argparse.ArgumentParser(description='求解器入口')
        parser.add_argument('--case_path', type=str, required=True,
                          help='案例目录路径，必须包含config.json文件')
        args = parser.parse_args()
        case_dir = args.case_path
    
    print(f"使用指定案例路径: {case_dir}")

    # 构造配置文件完整路径
    config_path = os.path.join(case_dir, "config.json")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"配置文件未找到: {config_path}")

    # 保存当前工作目录
    #original_dir = os.getcwd()

    ## 切换到案例目录
    #os.chdir(case_dir)
    #print(f"当前工作目录: {os.getcwd()}")

    # 加载配置文件
    with open(config_path, "r") as f:
        config_dict = json.load(f)

    # 从配置中获取问题类型
    problem_type = config_dict.get("problem_type", None)
    if problem_type is None:
        raise ValueError("配置文件中缺少 'problem_type' 字段")

    print(f"检测到问题类型: {problem_type}")

    # 确保结果目录存在
    os.makedirs(os.path.join(case_dir, "results"), exist_ok=True)

    # 根据问题类型选择求解器
    if problem_type == "time_pde":
        config = TimePDEConfig(config_path)
        solver = TimePDESolver(config)
        solver.solve()

    elif problem_type == "func_fitting":
        # 使用dataclass方式创建配置
        config = FuncFittingConfig(case_dir=case_dir)
        solver = FuncFittingSolver(config=config, case_dir=case_dir)
        solver.solve()
        
    elif problem_type == "linear_pde":
        # 使用dataclass方式创建配置
        config = LinearPDEConfig(case_dir=case_dir)
        solver = LinearPDESolver(config=config, case_dir=case_dir)
        solver.solve()
        
    else:
        raise ValueError(f"不支持的问题类型: {problem_type}")

    # 恢复原始工作目录
    #os.chdir(original_dir)
    print("求解完成!")


if __name__ == "__main__":
    main()
