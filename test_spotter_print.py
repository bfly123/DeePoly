"""Quick test to verify Spotter print output shows correct network structure"""

import sys
import os
import torch
import numpy as np

# Add project root to path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

from src.problem_solvers.general_pde_solver.utils.config import GeneralPDEConfig
from src.problem_solvers.general_pde_solver.core.spotter_init import SpotterNet
from src.problem_solvers.general_pde_solver.utils.data import GeneralPDEDataGenerator

# Load config
case_dir = "cases/general_pde_cases/AC_equation"
config = GeneralPDEConfig(case_dir=case_dir)

print("\n" + "=" * 80)
print("Testing Spotter Print Output")
print("=" * 80)

# Create data generator
datagen = GeneralPDEDataGenerator(config)
data = datagen.generate_data("train")

# Create SpotterNet
spotter = SpotterNet(config, use_periodic=False).to(config.device)

print("\n" + "=" * 80)
print("Running Spotter training (10 epochs only for verification)")
print("=" * 80 + "\n")

# Train for just 10 epochs to see the print output
loss_history = spotter.train_spotter(
    data=data,  # Pass the full data dictionary
    n_epochs=10,  # Only 10 epochs for quick test
    n_pde_points=5000,
    lr_init=0.01,
    print_every=5  # Print every 5 epochs
)

print("\n" + "=" * 80)
print("Verification Complete!")
print("=" * 80)
print("\n✅ Check the output above - it should show:")
print("   - 网络结构: [2, 90, 90, 90, 50, 1] (参数数量: 21250)")
print("   - 配置的 hidden_dims: [90, 90, 90, 50]")
print("\nNOT the old hardcoded: [2, 128, 128, 128, 128, 1]")
print("=" * 80 + "\n")
