import os
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm
from scipy.interpolate import interp1d


class TimeSolverPostprocessMixin:
    def _postprocess_results(self, U_final: np.ndarray, U_seg_final: list, model_final, coeffs_final):
        """BackwardProcessResult - 专注于trainData与Reference solution对比"""
        self._final_model = self._get_spotter_feature_source() if hasattr(self, "_get_spotter_feature_source") else model_final
        self._final_coeffs = coeffs_final
        _ = (U_final, U_seg_final)
        print("=== Post-processing and Visualization ===")
        self._configure_postprocess_style()
        time_history, solution_history = self.visualizer.get_animation_data()
        if len(time_history) == 0 or len(solution_history) == 0:
            print("Warning: No time evolution data available for analysis")
            return

        print("1. AnalyzeFinal时刻Solution与Reference solution的对比...")
        self._plot_final_time_comparison(time_history, solution_history)
        print("2. Generate时空二维Graph...")
        self._plot_spacetime_comparison(time_history, solution_history)
        print("3. Statistics所Sometimes刻的误差...")
        self._compute_overall_error_statistics(time_history, solution_history)
        print("4. GenerateTime演化animation...")
        self._generate_time_evolution_animation(time_history, solution_history)
        print("5. ExportData供MATLABAnalyze...")
        self._export_data_for_matlab(time_history, solution_history)
        print(f"=== Analysis results saved to: {self.config.results_dir} ===")
        print("=== To generate MATLAB plots, run: matlab -batch 'plot_comparison' in the reference_data directory ===")

    def _configure_postprocess_style(self) -> None:
        plt.style.use("default")
        plt.rcParams.update(
            {
                "font.size": 11,
                "axes.labelsize": 12,
                "axes.titlesize": 14,
                "xtick.labelsize": 10,
                "ytick.labelsize": 10,
                "legend.fontsize": 11,
                "figure.titlesize": 16,
                "axes.grid": True,
                "grid.alpha": 0.3,
                "lines.linewidth": 2,
                "axes.spines.top": False,
                "axes.spines.right": False,
                "figure.facecolor": "white",
            }
        )

    def _generate_time_evolution_animation(self, time_history, solution_history) -> None:
        try:
            gif_filename = os.path.join(self.config.results_dir, "time_evolution.gif")
            animation_skip = getattr(self.config, "animation_skip", 1)
            sampled_time_history = time_history[::animation_skip] if animation_skip > 1 else time_history
            sampled_solution_history = solution_history[::animation_skip] if animation_skip > 1 else solution_history
            if animation_skip > 1:
                print(f"  UsingSamplingDataGenerateanimation (每 {animation_skip} 步SamplingOnce)")
            self.visualizer.create_time_evolution_gif(
                sampled_time_history,
                sampled_solution_history,
                self._get_plot_data(),
                filename=gif_filename,
                solver=self,
            )
            print(f"  animation已Save至: {gif_filename}")
        except Exception as exc:
            print(f"  animationGenerateFail: {exc}")

    def _plot_final_time_comparison(self, time_history, solution_history):
        """1. Final时刻trainResult与Reference solution对比和逐point误差"""
        if not self.reference_solution:
            print("  No reference solution available for comparison")
            return

        T_final = time_history[-1]
        U_final = solution_history[-1]
        u_ref_final = self.get_reference_solution_at_time(T_final)
        if u_ref_final is None:
            print("  Cannot get reference solution at final time")
            return

        x_flat = self._get_plot_x_coordinates()
        x_ref = self.reference_solution["x_ref"]
        U_flat = U_final.flatten()
        if getattr(self.config, "n_dim", None) == 1:
            x_train, u_train = self._get_final_training_plot_data()
            if x_train is not None and u_train is not None:
                x_flat = x_train
                U_flat = u_train
        u_ref_interp = interp1d(x_ref, u_ref_final, kind="cubic", bounds_error=False, fill_value="extrapolate")(x_flat)
        pointwise_error = np.abs(U_flat - u_ref_interp)
        max_error = np.max(pointwise_error)
        l2_error = np.sqrt(np.mean(pointwise_error**2))
        self._save_final_time_comparison_plot(T_final, x_ref, u_ref_final, x_flat, U_flat, pointwise_error)
        print(f"  Final time error - Max: {max_error:.2e}, L2: {l2_error:.2e}")

    def _get_final_training_plot_data(self):
        if not hasattr(self, "data_train") or self.data_train is None:
            return None, None
        if not hasattr(self, "_final_model") or self._final_model is None:
            return None, None
        if not hasattr(self, "_final_coeffs") or self._final_coeffs is None:
            return None, None
        try:
            U_train, _ = self.fitter.construct(self.data_train, self._final_model, self._final_coeffs)
            x_train = self.data_train.get("x")
            if x_train is None:
                return None, None
            x_train_flat = x_train[:, 0] if x_train.ndim > 1 else x_train
            return x_train_flat, U_train.flatten()
        except Exception as exc:
            print(f"  Warning: Failed to construct final training-point plot data: {exc}")
            return None, None

    def _save_final_time_comparison_plot(self, T_final, x_ref, u_ref_final, x_flat, U_flat, pointwise_error) -> None:
        if getattr(self.config, "n_dim", None) == 1:
            fig, ax = plt.subplots(1, 1, figsize=(12, 6))
            ref_sort_idx = np.argsort(x_ref)
            x_ref_sorted = np.asarray(x_ref).flatten()[ref_sort_idx]
            u_ref_sorted = np.asarray(u_ref_final).flatten()[ref_sort_idx]
            x_num = np.asarray(x_flat).flatten()
            u_num = np.asarray(U_flat).flatten()
            ax.plot(
                x_ref_sorted,
                u_ref_sorted,
                "b-",
                linewidth=2.5,
                alpha=0.9,
                label="Reference Solution",
                zorder=1,
            )
            ax.scatter(
                x_num,
                u_num,
                c="red",
                s=36,
                alpha=0.9,
                label="Training Points",
                zorder=2,
            )
            ax.set_xlabel("x", fontsize=12)
            ax.set_ylabel("u", fontsize=12)
            ax.set_title(f"Final Time Comparison (T = {T_final:.4f})", fontsize=14, fontweight="bold")
            ax.legend(fontsize=11, loc="best")
            ax.grid(True, alpha=0.3)
            u_min_plot = min(np.min(u_num), np.min(u_ref_sorted))
            u_max_plot = max(np.max(u_num), np.max(u_ref_sorted))
            u_margin = (u_max_plot - u_min_plot) * 0.05
            ax.set_ylim([u_min_plot - u_margin, u_max_plot + u_margin])
            x_min_plot = min(np.min(x_num), np.min(x_ref_sorted))
            x_max_plot = max(np.max(x_num), np.max(x_ref_sorted))
            x_margin = (x_max_plot - x_min_plot) * 0.03
            ax.set_xlim([x_min_plot - x_margin, x_max_plot + x_margin])
            ax.text(
                0.02,
                0.98,
                f"Max Error: {np.max(pointwise_error):.2e}\nL2 Error: {np.sqrt(np.mean(pointwise_error**2)):.2e}",
                transform=ax.transAxes,
                verticalalignment="top",
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
            )
            plt.tight_layout()
        else:
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
            ax1.plot(x_ref, u_ref_final, "b-", linewidth=2.5, alpha=0.8, label="Reference Solution", zorder=1)
            ax1.plot(x_flat, U_flat, "r-", linewidth=2.0, alpha=0.95, label="Test Prediction", zorder=2)
            ax1.set_xlabel("x", fontsize=12)
            ax1.set_ylabel("u", fontsize=12)
            ax1.set_title(f"Final Time Comparison (T = {T_final:.4f})", fontsize=14, fontweight="bold")
            ax1.legend(fontsize=11, loc="best")
            ax1.grid(True, alpha=0.3)
            u_min_plot = min(np.min(U_flat), np.min(u_ref_final))
            u_max_plot = max(np.max(U_flat), np.max(u_ref_final))
            u_margin = (u_max_plot - u_min_plot) * 0.05
            ax1.set_ylim([u_min_plot - u_margin, u_max_plot + u_margin])

            ax2.plot(x_flat, pointwise_error, "r-", linewidth=1.8)
            ax2.set_xlabel("x", fontsize=12)
            ax2.set_ylabel("|Error|", fontsize=12)
            ax2.set_title("Pointwise Error at Final Time (Log Scale)", fontsize=14, fontweight="bold")
            ax2.set_yscale("log")
            ax2.grid(True, alpha=0.3)
            ax2.text(
                0.02,
                0.98,
                f"Max Error: {np.max(pointwise_error):.2e}\nL2 Error: {np.sqrt(np.mean(pointwise_error**2)):.2e}",
                transform=ax2.transAxes,
                verticalalignment="top",
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
            )
            plt.tight_layout()
        final_comparison_path = os.path.join(self.config.results_dir, "final_time_comparison.png")
        plt.savefig(final_comparison_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"  Final time comparison saved to: {final_comparison_path}")

    def _plot_spacetime_comparison(self, time_history, solution_history):
        """2. t-x二维Graph：trainSolution、Reference solution和逐point误差"""
        if not self.reference_solution:
            print("  No reference solution available for spacetime plot")
            return

        x_flat = self._get_plot_x_coordinates()
        nt, nx = len(time_history), len(x_flat)
        U_spacetime = np.zeros((nt, nx))
        U_ref_spacetime = np.zeros((nt, nx))
        x_ref = self.reference_solution["x_ref"]

        for index, (time_value, U_t) in enumerate(zip(time_history, solution_history)):
            U_spacetime[index, :] = U_t.flatten()
            u_ref_t = self.get_reference_solution_at_time(time_value)
            if u_ref_t is not None:
                U_ref_spacetime[index, :] = interp1d(
                    x_ref,
                    u_ref_t,
                    kind="cubic",
                    bounds_error=False,
                    fill_value="extrapolate",
                )(x_flat)

        error_spacetime = np.abs(U_spacetime - U_ref_spacetime)
        self._save_spacetime_comparison_plot(time_history, x_flat, U_spacetime, U_ref_spacetime, error_spacetime)

    def _save_spacetime_comparison_plot(self, time_history, x_flat, U_spacetime, U_ref_spacetime, error_spacetime) -> None:
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        u_min = min(np.min(U_spacetime), np.min(U_ref_spacetime))
        u_max = max(np.max(U_spacetime), np.max(U_ref_spacetime))

        t_scatter, x_scatter = np.meshgrid(time_history, x_flat, indexing="ij")
        mesh1 = axes[0].pcolormesh(
            time_history,
            x_flat,
            U_spacetime.T,
            cmap="RdBu_r",
            shading="gouraud",
            vmin=u_min,
            vmax=u_max,
        )
        axes[0].set_xlabel("t", fontsize=12)
        axes[0].set_ylabel("x", fontsize=12)
        axes[0].set_title("Test Prediction", fontsize=12, fontweight="bold")
        axes[0].grid(True, alpha=0.3)
        plt.colorbar(mesh1, ax=axes[0]).set_label("u", fontsize=11)

        t_ref_grid = self.reference_solution["t_ref"]
        x_ref_grid = self.reference_solution["x_ref"]
        u_ref_grid = self.reference_solution["u_ref"]
        T_ref_mesh, X_ref_mesh = np.meshgrid(t_ref_grid, x_ref_grid)
        im2 = axes[1].pcolormesh(T_ref_mesh, X_ref_mesh, u_ref_grid.T, cmap="RdBu_r", shading="gouraud", vmin=u_min, vmax=u_max)
        axes[1].set_xlabel("t", fontsize=12)
        axes[1].set_ylabel("x", fontsize=12)
        axes[1].set_title("Reference Solution (High-Res)", fontsize=12, fontweight="bold")
        plt.colorbar(im2, ax=axes[1]).set_label("u", fontsize=11)

        error_nonzero = np.maximum(error_spacetime.ravel(), 1e-12)
        vmin_error = max(np.percentile(error_nonzero, 5), 1e-10)
        vmax_error = min(np.percentile(error_nonzero, 95) * 5, np.max(error_nonzero))
        print(f"  Error colormap range: [{vmin_error:.2e}, {vmax_error:.2e}], median: {np.median(error_nonzero):.2e}")
        mesh3 = axes[2].pcolormesh(
            time_history,
            x_flat,
            error_spacetime.T,
            cmap="plasma",
            shading="gouraud",
            norm=LogNorm(vmin=vmin_error, vmax=vmax_error),
        )
        axes[2].set_xlabel("t", fontsize=12)
        axes[2].set_ylabel("x", fontsize=12)
        axes[2].set_title("Pointwise Error (Log Scale)", fontsize=12, fontweight="bold")
        axes[2].grid(True, alpha=0.3)
        plt.colorbar(mesh3, ax=axes[2]).set_label("|Error|", fontsize=11)

        plt.tight_layout()
        spacetime_path = os.path.join(self.config.results_dir, "spacetime_comparison.png")
        plt.savefig(spacetime_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"  Spacetime comparison saved to: {spacetime_path}")

    def _compute_overall_error_statistics(self, time_history, solution_history):
        """3. Statistics sampled spacetime error on the training grid."""
        if not self.reference_solution:
            print("  No reference solution available for error statistics")
            return

        x_flat = self._get_plot_x_coordinates()
        x_ref = self.reference_solution["x_ref"]
        l2_errors = []
        linf_errors = []
        error_snapshots = []
        for time_value, U_t in zip(time_history, solution_history):
            u_ref_t = self.get_reference_solution_at_time(time_value)
            if u_ref_t is None:
                continue
            u_ref_interp = interp1d(x_ref, u_ref_t, kind="cubic", bounds_error=False, fill_value="extrapolate")(x_flat)
            error = np.abs(U_t.flatten() - u_ref_interp)
            l2_errors.append(np.sqrt(np.mean(error**2)))
            linf_errors.append(np.max(error))
            error_snapshots.append(error)

        if len(l2_errors) == 0:
            print("  No valid time points for error calculation")
            return

        error_matrix = np.vstack(error_snapshots)
        global_l2_error = np.sqrt(np.mean(error_matrix**2))
        global_linf_error = np.max(error_matrix)

        self._write_error_statistics_report(
            time_history,
            x_flat,
            l2_errors,
            linf_errors,
            global_l2_error,
            global_linf_error,
        )
        self._save_error_evolution_plot(time_history, l2_errors, linf_errors)
        print("  Overall Error Statistics:")
        print(f"    Global Spacetime L2 Error:  {global_l2_error:.6e}")
        print(f"    Global Spacetime L∞ Error:  {global_linf_error:.6e}")
        print(f"    Time-averaged L2 Error:  {np.mean(l2_errors):.6e}")
        print(f"    Time-averaged L∞ Error:  {np.mean(linf_errors):.6e}")
        print(f"    Maximum L2 Error:        {np.max(l2_errors):.6e}")
        print(f"    Maximum L∞ Error:        {np.max(linf_errors):.6e}")

    def _write_error_statistics_report(
        self,
        time_history,
        x_flat,
        l2_errors,
        linf_errors,
        global_l2_error,
        global_linf_error,
    ) -> None:
        mean_l2_error = np.mean(l2_errors)
        mean_linf_error = np.mean(linf_errors)
        max_l2_error = np.max(l2_errors)
        max_linf_error = np.max(linf_errors)
        report_path = os.path.join(self.config.results_dir, "error_statistics_report.txt")
        with open(report_path, "w", encoding="utf-8") as file_obj:
            file_obj.write("=" * 70 + "\n")
            file_obj.write("                   ERROR STATISTICS REPORT                    \n")
            file_obj.write("=" * 70 + "\n\n")
            file_obj.write("【Timeinformation】\n")
            file_obj.write("-" * 40 + "\n")
            file_obj.write(f"  TimeRange:     [{time_history[0]:.6f}, {time_history[-1]:.6f}]\n")
            file_obj.write(f"  Time step数:     {len(l2_errors)}\n")
            file_obj.write(f"  Time step size:     {self.config.dt}\n")
            file_obj.write(f"  Spacepoint数:     {len(x_flat)}\n\n")
            file_obj.write("【ComputeTimeStatistics】\n")
            file_obj.write("-" * 40 + "\n")
            if hasattr(self, "physics_compute_time"):
                file_obj.write(f"  PhysicalComputeTime: {self.physics_compute_time:.2f} second\n")
                file_obj.write(f"  Average每步耗时: {self.physics_compute_time/len(l2_errors):.4f} second\n")
                file_obj.write(f"  ComputeEfficiency:     {len(l2_errors)/self.physics_compute_time:.2f} 步/second\n")
                physical_time_range = time_history[-1] - time_history[0]
                file_obj.write(f"  Physical/ComputeTime比: {physical_time_range / self.physics_compute_time:.4f}\n")
            else:
                file_obj.write("  PhysicalComputeTime: N/A\n")
            file_obj.write("\n")
            file_obj.write("【误差Statistics】\n")
            file_obj.write("-" * 40 + "\n")
            file_obj.write("  全局时空误差:\n")
            file_obj.write(f"    - Global L2  误差:   {global_l2_error:.6e}\n")
            file_obj.write(f"    - Global L∞  误差:   {global_linf_error:.6e}\n\n")
            file_obj.write("  TimeAverage误差:\n")
            file_obj.write(f"    - L2  误差:   {mean_l2_error:.6e}\n")
            file_obj.write(f"    - L∞  误差:   {mean_linf_error:.6e}\n\n")
            file_obj.write("  最大误差 (所Sometimes刻):\n")
            file_obj.write(f"    - L2  误差:   {max_l2_error:.6e}\n")
            file_obj.write(f"    - L∞  误差:   {max_linf_error:.6e}\n\n")
            if global_l2_error > 0:
                file_obj.write("  Error analysis:\n")
                file_obj.write(f"    - L∞/L2 Rate: {mean_linf_error/mean_l2_error:.2f}\n")
                file_obj.write(f"    - GlobalPrecision:   {-np.log10(global_l2_error):.1f} 位小数\n")
                file_obj.write(f"    - TimeAveragePrecision:   {-np.log10(mean_l2_error):.1f} 位小数\n\n")
            file_obj.write("【Solve器Configuration】\n")
            file_obj.write("-" * 40 + "\n")
            file_obj.write(f"  Timeformat:     {getattr(self.config, 'time_scheme', 'N/A')}\n")
            file_obj.write(f"  Neural network:     {self.config.hidden_dims}\n")
            file_obj.write(f"  分Number of segments量:     {self.config.n_segments}\n")
            file_obj.write(f"  Polynomial阶:     {self.config.poly_degree}\n")
            file_obj.write(f"  EquipmentType:     {getattr(self.config, 'device', 'cpu')}\n\n")
            file_obj.write("=" * 70 + "\n")
            file_obj.write(f"GenerateTime: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            file_obj.write("=" * 70 + "\n")
        print(f"  误差StatisticsReport已Save至: {report_path}")
        if hasattr(self, "physics_compute_time"):
            print(f"  PhysicalComputeTime: {self.physics_compute_time:.2f} second (不含Initialize和画Graph)")
            print(f"  ComputeEfficiency: {len(l2_errors)/self.physics_compute_time:.2f} Time step/second")

    def _save_error_evolution_plot(self, time_history, l2_errors, linf_errors) -> None:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        times = time_history[: len(l2_errors)]
        ax1.scatter(times, l2_errors, c="blue", s=40, alpha=0.8, edgecolors="navy", linewidths=0.5, label="L2 Error")
        ax1.set_xlabel("Time", fontsize=12)
        ax1.set_ylabel("L2 Error (log scale)", fontsize=12)
        ax1.set_title("L2 Error Evolution (Training Grid)", fontsize=14, fontweight="bold")
        ax1.set_yscale("log")
        ax1.grid(True, alpha=0.3)
        if len(l2_errors) > 2:
            trend = np.poly1d(np.polyfit(times, np.log10(l2_errors), 1))
            ax1.plot(times, 10 ** trend(times), "b--", alpha=0.5, linewidth=1.5, label="Trend")

        ax2.scatter(times, linf_errors, c="red", s=40, alpha=0.8, edgecolors="darkred", linewidths=0.5, label="L∞ Error")
        ax2.set_xlabel("Time", fontsize=12)
        ax2.set_ylabel("L∞ Error (log scale)", fontsize=12)
        ax2.set_title("L∞ Error Evolution (Training Grid)", fontsize=14, fontweight="bold")
        ax2.set_yscale("log")
        ax2.grid(True, alpha=0.3)
        if len(linf_errors) > 2:
            trend = np.poly1d(np.polyfit(times, np.log10(linf_errors), 1))
            ax2.plot(times, 10 ** trend(times), "r--", alpha=0.5, linewidth=1.5, label="Trend")

        ax1.legend(fontsize=10)
        ax2.legend(fontsize=10)
        plt.tight_layout()
        error_evolution_path = os.path.join(self.config.results_dir, "error_evolution.png")
        plt.savefig(error_evolution_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"  Error evolution plot saved to: {error_evolution_path}")

    def _export_data_for_matlab(self, time_history, solution_history):
        """ExporttrainData供MATLABAnalyzeUsing"""
        try:
            import scipy.io

            x_flat = self._get_plot_x_coordinates()
            nt, nx = len(time_history), len(x_flat)
            U_matrix = np.zeros((nx, nt))
            for index, U_t in enumerate(solution_history):
                U_matrix[:, index] = U_t.flatten()

            export_data = {
                "time_history": np.array(time_history),
                "x_coords": x_flat,
                "solution_history": U_matrix,
                "config_info": {
                    "time_scheme": getattr(self.config, "time_scheme", "unknown"),
                    "dt": self.config.dt,
                    "T": self.config.T,
                    "n_segments": self.config.n_segments,
                    "poly_degree": self.config.poly_degree,
                    "hidden_dims": self.config.hidden_dims,
                },
            }
            case_dir = getattr(self.config, "case_dir", os.getcwd())
            ref_data_dir = os.path.join(case_dir, "reference_data")
            os.makedirs(ref_data_dir, exist_ok=True)
            matlab_data_path = os.path.join(ref_data_dir, "train_data.mat")
            scipy.io.savemat(matlab_data_path, export_data)
            print(f"  Training-grid data exported to: {matlab_data_path}")
            print(f"  Data dimensions: {nt} time steps, {nx} spatial points")
            print(f"  Time range: [{time_history[0]:.6f}, {time_history[-1]:.6f}]")
            print(f"  Spatial range: [{x_flat.min():.6f}, {x_flat.max():.6f}]")
        except ImportError:
            print("  Warning: scipy.io not available, cannot export MATLAB data")
        except Exception as exc:
            print(f"  Warning: Failed to export MATLAB data: {exc}")

    def _get_plot_data(self):
        if getattr(self, "data_train", None) is not None:
            return self.data_train
        return self.data_test

    def _get_plot_x_coordinates(self) -> np.ndarray:
        x_coords = self._get_plot_data().get("x", self._get_plot_data().get("x_segments", []))
        if isinstance(x_coords, list):
            x_plot = np.vstack(x_coords) if x_coords else np.array([[0]])
        else:
            x_plot = x_coords
        return x_plot[:, 0] if x_plot.ndim > 1 else x_plot
