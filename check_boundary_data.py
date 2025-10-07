import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.problem_solvers.general_pde_solver.utils.config import GeneralPDEConfig
from src.problem_solvers.general_pde_solver.utils.data import GeneralPDEDataGenerator
import numpy as np

case_dir = "cases/general_pde_cases/AC_equation"
config = GeneralPDEConfig(case_dir=case_dir)

datagen = GeneralPDEDataGenerator(config)
data_train = datagen.generate_data("train")

print("\n边界条件数据:")
if "global_boundary_dict" in data_train:
    for var_idx, var_data in data_train["global_boundary_dict"].items():
        print(f"\n变量 {var_idx}:")
        for bc_type, bc_data in var_data.items():
            print(f"  边界类型: {bc_type}")
            if "x" in bc_data and bc_data["x"] is not None:
                print(f"    点数: {len(bc_data['x'])}")
                print(f"    x range: [{np.min(bc_data['x']):.3f}, {np.max(bc_data['x']):.3f}]")
                if "values" in bc_data and bc_data["values"] is not None:
                    print(f"    values range: [{np.min(bc_data['values']):.3f}, {np.max(bc_data['values']):.3f}]")
                    print(f"    values mean: {np.mean(bc_data['values']):.3f}")
else:
    print("  没有边界条件数据！")
