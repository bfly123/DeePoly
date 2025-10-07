#!/usr/bin/env python3
"""Quick test script to verify main_solver default configuration works"""

import sys
import os

# Add src to path
src_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "src")
sys.path.insert(0, src_dir)

from main_solver import parse_arguments, create_solver
import json

print("=" * 80)
print("Testing main_solver with default case path")
print("=" * 80)

# Test 1: Parse arguments with no command line args
print("\n1. Testing argument parsing (simulating no --case_path)...")
sys.argv = ['run_ac_test.py']  # Simulate running without arguments
args = parse_arguments()
print(f"   Default case_path: {args.case_path}")

# Test 2: Verify config file exists
case_dir = args.case_path
config_path = os.path.join(case_dir, "config.json")
print(f"\n2. Checking config file: {config_path}")
if os.path.exists(config_path):
    print(f"   ✅ Config file exists")
    with open(config_path, 'r') as f:
        config_dict = json.load(f)
    problem_type = config_dict.get('problem_type')
    hidden_dims = config_dict.get('hidden_dims')
    print(f"   Problem type: {problem_type}")
    print(f"   Network hidden_dims: {hidden_dims}")
else:
    print(f"   ❌ Config file not found!")
    sys.exit(1)

# Test 3: Verify configuration loading
print(f"\n3. Loading configuration...")
from problem_solvers.general_pde_solver.utils import GeneralPDEConfig
config = GeneralPDEConfig(case_dir=case_dir)
print(f"   ✅ Config loaded successfully")
print(f"   config.hidden_dims: {config.hidden_dims}")
print(f"   config.n_dim: {config.n_dim}")
print(f"   config.n_eqs: {config.n_eqs}")

# Test 4: Verify network creation
print(f"\n4. Creating networks...")
from problem_solvers.general_pde_solver.core.spotter_init import SpotterNet
from problem_solvers.general_pde_solver.core.net import GeneralPDENet

spotter = SpotterNet(config, use_periodic=False)
fitter = GeneralPDENet(config)

n_params_spotter = sum(p.numel() for p in spotter.parameters())
n_params_fitter = sum(p.numel() for p in fitter.parameters())

net_structure = [config.n_dim] + list(config.hidden_dims) + [config.n_eqs]

print(f"   ✅ Networks created successfully")
print(f"   Network structure: {net_structure}")
print(f"   SpotterNet parameters: {n_params_spotter:,}")
print(f"   GeneralPDENet parameters: {n_params_fitter:,}")
print(f"   Match: {n_params_spotter == n_params_fitter}")

print("\n" + "=" * 80)
print("✅ All configuration tests passed!")
print("=" * 80)
print("\nTo run full training:")
print("  python -m src.main_solver")
print("\nOr with explicit path:")
print("  python -m src.main_solver --case_path cases/general_pde_cases/AC_equation")
print("=" * 80)
