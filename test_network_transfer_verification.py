"""Verify complete network transfer from SpotterNet to GeneralPDENet"""

import sys
import os
import torch
import numpy as np

# Add project root to path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

from src.problem_solvers.general_pde_solver.utils.config import GeneralPDEConfig
from src.problem_solvers.general_pde_solver.core.spotter_init import SpotterNet
from src.problem_solvers.general_pde_solver.core.net import GeneralPDENet

# Load config
case_dir = "cases/general_pde_cases/AC_equation"
config = GeneralPDEConfig(case_dir=case_dir)

print("=" * 80)
print("Network Transfer Verification")
print("=" * 80)

# Step 1: Create SpotterNet
print("\n1. Creating SpotterNet...")
spotter = SpotterNet(config, use_periodic=False).to(config.device)
n_params_spotter = sum(p.numel() for p in spotter.parameters())
net_structure = [config.n_dim] + list(spotter.hidden_dims) + [config.n_eqs]
print(f"   Network structure: {net_structure}")
print(f"   Total parameters: {n_params_spotter}")

# Step 2: Create GeneralPDENet
print("\n2. Creating GeneralPDENet...")
fitter = GeneralPDENet(config).to(config.device)
n_params_fitter = sum(p.numel() for p in fitter.parameters())
net_structure = [config.n_dim] + list(fitter.hidden_dims) + [config.n_eqs]
print(f"   Network structure: {net_structure}")
print(f"   Total parameters: {n_params_fitter}")

# Step 3: Transfer parameters
print("\n3. Transferring parameters from SpotterNet to GeneralPDENet...")
try:
    fitter.load_state_dict(spotter.state_dict())
    print("   ✅ load_state_dict() executed successfully")
except Exception as e:
    print(f"   ❌ Error: {e}")
    sys.exit(1)

# Step 4: Verify transfer accuracy
print("\n4. Verifying transfer accuracy...")
# Create test input
x_test = torch.randn(100, config.n_dim, dtype=torch.float64, device=config.device)

# Get outputs
with torch.no_grad():
    _, U_spotter = spotter(x_test)
    _, U_fitter = fitter(x_test)

# Compute difference
max_diff = torch.max(torch.abs(U_spotter - U_fitter)).item()
print(f"   Maximum output difference: {max_diff:.2e}")

if max_diff < 1e-14:
    print(f"   ✅ Perfect transfer (error < 1e-14)")
else:
    print(f"   ⚠️  Transfer has numerical error: {max_diff:.2e}")

# Step 5: Summary
print("\n" + "=" * 80)
print("VERIFICATION SUMMARY")
print("=" * 80)
print(f"SpotterNet:     {n_params_spotter:,} parameters")
print(f"GeneralPDENet:  {n_params_fitter:,} parameters")
print(f"Match:          {n_params_spotter == n_params_fitter}")
print(f"Transfer error: {max_diff:.2e}")

if n_params_spotter == n_params_fitter and max_diff < 1e-14:
    print("\n🎉 SUCCESS: Network architecture is fully aligned!")
    print("   ✅ Same structure")
    print("   ✅ Same parameter count")
    print("   ✅ Zero-loss parameter transfer")
else:
    print("\n❌ FAILURE: Networks are not properly aligned")

print("=" * 80)
