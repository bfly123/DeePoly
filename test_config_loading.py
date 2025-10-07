"""Test script to check if config.hidden_dims is loaded correctly"""

import sys
import os

# Add project root to path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

from src.problem_solvers.general_pde_solver.utils.config import GeneralPDEConfig
from src.problem_solvers.general_pde_solver.core.spotter_init import SpotterNet
from src.problem_solvers.general_pde_solver.core.net import GeneralPDENet
import json

# Load config
case_dir = "cases/general_pde_cases/AC_equation"
print("=" * 80)
print("Testing config loading")
print("=" * 80)

# 1. Check JSON file
config_path = os.path.join(case_dir, "config.json")
print(f"\n1. Reading JSON file: {config_path}")
with open(config_path, 'r') as f:
    config_dict = json.load(f)
print(f"   hidden_dims from JSON: {config_dict.get('hidden_dims', 'NOT FOUND')}")

# 2. Load config object
print(f"\n2. Loading GeneralPDEConfig:")
config = GeneralPDEConfig(case_dir=case_dir)
print(f"   config.hidden_dims = {config.hidden_dims}")
print(f"   config.n_dim = {config.n_dim}")
print(f"   config.n_eqs = {config.n_eqs}")

# 3. Create SpotterNet
print(f"\n3. Creating SpotterNet:")
spotter = SpotterNet(config, use_periodic=False)
print(f"   spotter.config.hidden_dims = {spotter.config.hidden_dims}")
print(f"   spotter.hidden_dims = {spotter.hidden_dims}")
n_params_spotter = sum(p.numel() for p in spotter.parameters())
print(f"   Total parameters: {n_params_spotter}")

# Print network structure
in_dim = config.n_dim
out_dim = config.n_eqs
hidden_dims = spotter.hidden_dims
net_structure = [in_dim] + list(hidden_dims) + [out_dim]
print(f"   Network structure: {net_structure}")

# 4. Create GeneralPDENet
print(f"\n4. Creating GeneralPDENet:")
fitter = GeneralPDENet(config)
print(f"   fitter.config.hidden_dims = {fitter.config.hidden_dims}")
print(f"   fitter.hidden_dims = {fitter.hidden_dims}")
n_params_fitter = sum(p.numel() for p in fitter.parameters())
print(f"   Total parameters: {n_params_fitter}")

# Print network structure
hidden_dims = fitter.hidden_dims
net_structure = [in_dim] + list(hidden_dims) + [out_dim]
print(f"   Network structure: {net_structure}")

# 5. Check if structures match
print(f"\n5. Verification:")
print(f"   SpotterNet parameters: {n_params_spotter}")
print(f"   GeneralPDENet parameters: {n_params_fitter}")
print(f"   Match: {n_params_spotter == n_params_fitter}")

if n_params_spotter == n_params_fitter:
    print("   ✅ Network structures are identical!")
else:
    print("   ❌ Network structures differ!")

print("=" * 80)
