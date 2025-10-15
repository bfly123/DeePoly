#!/bin/bash
source ~/anaconda3/bin/activate ML
cd /home/bfly/workspace/computeforcfd/混合网络/DeePoly_git
python src/main_solver.py --case_path cases/nonlinear_pde_cases/Time_pde_cases/Allen_Cahn/AC_equation_100_0.01 2>&1 | grep -v "CuPy未Installation"
