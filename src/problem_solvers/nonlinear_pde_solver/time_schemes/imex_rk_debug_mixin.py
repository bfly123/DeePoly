import os
from typing import List

import matplotlib.pyplot as plt
import numpy as np


class ImexRKDebugMixin:
    def _debug_L2F_term(self, segment_idx: int, U_seg: np.ndarray, stage: int):
        """Debug L2*F term to check if it's working correctly"""
        print(f"\n=== Debugging L2⊙F Term (Stage {stage}, Segment {segment_idx}) ===")
        L2_operators = self.fitter._linear_operators[segment_idx].get("L2", None)
        L2_seg = L2_operators[0] if L2_operators is not None else None
        if L2_seg is None:
            return

        features_list = self.fitter._features[segment_idx]
        features = features_list[0] if isinstance(features_list, list) else features_list
        if not self.fitter.has_operator("F"):
            print("F operator not found!")
            print("=" * 60)
            return

        F_vals = self.fitter.F_func(features_list, U_seg)
        ne = self.config.n_eqs
        dgN = self.fitter.dgN
        for eq_idx in range(ne):
            beta_test = np.ones(dgN)
            L2_beta = L2_seg @ beta_test
            F_eq = F_vals[:, eq_idx]
            L2F_contrib = L2_beta * F_eq
            ratio = np.linalg.norm(L2F_contrib) / (np.linalg.norm(L2_beta) + 1e-12)
            print(f"L2 @ β (test) range: [{L2_beta.min():.6e}, {L2_beta.max():.6e}]")
            print(
                "L2⊙F contribution range: "
                f"[{L2F_contrib.min():.6e}, {L2F_contrib.max():.6e}]"
            )
            print(f"L2⊙F contribution norm: {np.linalg.norm(L2F_contrib):.6e}")
            print(f"Pure L2 contribution norm: {np.linalg.norm(L2_beta):.6e}")
            print(f"L2⊙F/L2 ratio: {ratio:.6f}")
        print("=" * 60)

    def _plot_stage_solution(self, U_seg_stage: List[np.ndarray], stage: int, dt: float):
        """Plot intermediate stage solution for debugging and analysis"""
        try:
            if stage == 1 and len(U_seg_stage) > 0:
                self._debug_L2F_term(0, U_seg_stage[0], stage)

            U_stage_global = self.fitter.segments_to_global(U_seg_stage)
            x_coords = self._resolve_plot_coordinates()
            if x_coords is None:
                print(f"Warning: Cannot find coordinates for stage {stage} plotting")
                return

            if x_coords.shape[1] == 1:
                self._plot_stage_solution_1d(x_coords, U_stage_global, stage, dt)
            elif x_coords.shape[1] == 2:
                self._plot_stage_solution_2d(x_coords, U_stage_global, stage, dt)
            else:
                print(f"Warning: Stage plotting not supported for {x_coords.shape[1]}D")
                return
            self._save_stage_plot(stage, dt)
        except Exception as exc:
            print(f"Warning: Failed to plot stage {stage} solution: {exc}")

    def _resolve_plot_coordinates(self):
        if hasattr(self.fitter, "data") and "x" in self.fitter.data:
            return self.fitter.data["x"]
        if hasattr(self.fitter, "data") and "x_segments_norm" in self.fitter.data:
            return np.vstack(self.fitter.data["x_segments_norm"])
        return None

    def _plot_stage_solution_1d(self, x_coords: np.ndarray, U_stage_global: np.ndarray, stage: int, dt: float):
        plt.figure(figsize=(10, 6))
        x_flat = x_coords.flatten()
        sort_idx = np.argsort(x_flat)
        x_sorted = x_flat[sort_idx]
        for eq_idx in range(U_stage_global.shape[1]):
            U_sorted = U_stage_global[sort_idx, eq_idx]
            plt.plot(
                x_sorted,
                U_sorted,
                "o-",
                label=f"U_{eq_idx} Stage {stage}",
                linewidth=2,
                markersize=4,
                alpha=0.8,
            )
        plt.xlabel("x", fontsize=12)
        plt.ylabel("U(x)", fontsize=12)
        plt.title(f"IMEX-RK Stage {stage} Solution (dt={dt:.6f})", fontsize=14, fontweight="bold")
        plt.grid(True, alpha=0.3)
        plt.legend()

    def _plot_stage_solution_2d(self, x_coords: np.ndarray, U_stage_global: np.ndarray, stage: int, dt: float):
        plt.figure(figsize=(10, 6))
        for eq_idx in range(U_stage_global.shape[1]):
            plt.scatter(
                x_coords[:, 0],
                x_coords[:, 1],
                c=U_stage_global[:, eq_idx],
                cmap="RdYlBu_r",
                s=20,
                alpha=0.8,
            )
            plt.colorbar(label=f"U_{eq_idx}")
        plt.xlabel("x", fontsize=12)
        plt.ylabel("y", fontsize=12)
        plt.title(f"IMEX-RK Stage {stage} Solution (dt={dt:.6f})", fontsize=14, fontweight="bold")
        plt.axis("equal")

    def _save_stage_plot(self, stage: int, dt: float):
        plt.tight_layout()
        results_dir = getattr(self.config, "results_dir", ".")
        os.makedirs(results_dir, exist_ok=True)
        plot_path = os.path.join(results_dir, f"imex_stage_{stage}_dt_{dt:.6f}.png")
        plt.savefig(plot_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Stage {stage} solution plot saved to: {plot_path}")
