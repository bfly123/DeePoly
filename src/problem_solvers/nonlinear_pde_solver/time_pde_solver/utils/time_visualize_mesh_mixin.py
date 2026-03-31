import os
from typing import Dict, Optional

import matplotlib.pyplot as plt
import numpy as np
import torch


class TimePDEMeshMixin:
    def plot_mesh_results(
        self,
        data: Dict,
        model: torch.nn.Module,
        preds: np.ndarray,
        title: Optional[str] = None,
        save_gif: bool = False,
        save_png: bool = False,
        gif_filename: str = "animation.gif",
        png_filename: Optional[str] = None,
        fast_mode: bool = False,
        contour_levels: int = 50,
    ) -> None:
        """Visualizationu分量的流场分布"""
        _ = (data, model, gif_filename, fast_mode)
        self._save_gif = save_gif
        ns1, ns2 = self.config.n_segments
        np1, np2 = self.config.points_domain_test
        x1 = np.linspace(self.config.x_domain[0, 0], self.config.x_domain[0, 1], np1 * ns1)
        x2 = np.linspace(self.config.x_domain[1, 0], self.config.x_domain[1, 1], np2 * ns2)
        x1_grid, x2_grid = np.meshgrid(x1, x2)
        u_pred = preds[:, 0].reshape(np2 * ns2, np1 * ns1)

        if not hasattr(self, "_mesh_fig"):
            self._mesh_fig = plt.figure(figsize=(10, 8))
            self._mesh_axes = self._mesh_fig.add_subplot(111)

        self._mesh_axes.clear()
        contf = self._mesh_axes.contourf(x1_grid, x2_grid, u_pred, levels=contour_levels, cmap="RdBu")
        if title:
            self._mesh_axes.set_title(title)
        self._mesh_axes.set_xlabel("x")
        self._mesh_axes.set_ylabel("y")
        self._mesh_axes.set_aspect("equal")

        if hasattr(self._mesh_axes, "colorbar"):
            self._mesh_axes.colorbar.remove()
        self._mesh_axes.colorbar = plt.colorbar(contf, ax=self._mesh_axes)
        self._mesh_axes.colorbar.set_label("u")

        plt.tight_layout()
        self._mesh_fig.canvas.draw()
        if save_png and png_filename:
            plt.savefig(png_filename, dpi=300, bbox_inches="tight")
        if self._save_gif:
            self._capture_current_mesh_frame()
        self._mesh_fig.canvas.flush_events()
        plt.pause(0.01)

    def _capture_current_mesh_frame(self) -> None:
        import io
        from PIL import Image

        buf = io.BytesIO()
        self._mesh_fig.savefig(buf, format="png", dpi=100)
        buf.seek(0)
        self._frames.append(Image.open(buf))

    def plot_mesh_results_final(
        self,
        data: Dict,
        model: torch.nn.Module,
        preds: np.ndarray,
    ) -> None:
        """绘制FinalResult：u、v、p分量Graph和黑白流LineGraph"""
        _ = model
        ne = self.config.n_eqs
        x1_grid, x2_grid, u_pred, x1_unique, x2_unique = self._build_mesh_result_arrays(data, preds)
        os.makedirs("results", exist_ok=True)
        self._save_mesh_component_plot(
            x1_grid,
            x2_grid,
            u_pred[:, :, 0],
            "U Component",
            "U Velocity",
            "results/u_component.png",
        )
        self._save_mesh_component_plot(
            x1_grid,
            x2_grid,
            u_pred[:, :, 1],
            "V Component",
            "V Velocity",
            "results/v_component.png",
        )
        self._save_pressure_plot(x1_grid, x2_grid, u_pred, ne)
        self._save_streamline_plot(x1_unique, x2_unique, u_pred[:, :, 0].T, u_pred[:, :, 1].T)
        self._write_flow_field_data(x1_grid, x2_grid, u_pred, ne)
        print("4 separate images and Tecplot format data file have been saved to the results folder")

    def _build_mesh_result_arrays(self, data: Dict, preds: np.ndarray):
        ne = self.config.n_eqs
        ns1, ns2 = self.config.n_segments
        np1, np2 = self.config.points_per_segment_test
        points_per_section = np1 * np2
        total_sections = ns1 * ns2
        x1_grid = np.zeros((np1 * ns1, np2 * ns2))
        x2_grid = np.zeros((np1 * ns1, np2 * ns2))
        u_pred = np.zeros((np1 * ns1, np2 * ns2, ne))

        for section in range(total_sections):
            section_x = section // ns2
            section_y = section % ns2
            for point in range(points_per_section):
                point_x = point // np2
                point_y = point % np2
                grid_points = data["x_segments"][section][point]
                row = point_y + section_y * np1
                col = point_x + section_x * np2
                x1_grid[row, col] = grid_points[0]
                x2_grid[row, col] = grid_points[1]
                for eq in range(ne):
                    u_pred[row, col, eq] = preds[points_per_section * section + point, eq]

        return x1_grid, x2_grid, u_pred, np.unique(x1_grid), np.unique(x2_grid)

    def _save_mesh_component_plot(self, x1_grid, x2_grid, values, title, color_label, path) -> None:
        fig = self._create_figure(figsize=(8, 6))
        contf = plt.contourf(x1_grid, x2_grid, values, levels=20, cmap="viridis")
        plt.colorbar(contf, label=color_label)
        plt.title(title)
        plt.xlabel("x")
        plt.ylabel("y")
        plt.axis("equal")
        plt.tight_layout()
        plt.savefig(path, dpi=300, bbox_inches="tight")
        plt.close(fig)

    def _save_pressure_plot(self, x1_grid, x2_grid, u_pred, ne: int) -> None:
        fig = self._create_figure(figsize=(8, 6))
        if ne >= 3:
            contf_p = plt.contourf(x1_grid, x2_grid, u_pred[:, :, 2], levels=20, cmap="viridis")
            plt.colorbar(contf_p, label="Pressure")
            plt.title("P Component")
        else:
            plt.title("Missing P Component Data")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.axis("equal")
        plt.tight_layout()
        plt.savefig("results/p_component.png", dpi=300, bbox_inches="tight")
        plt.close(fig)

    def _save_streamline_plot(self, x1_unique, x2_unique, u_component, v_component) -> None:
        fig = self._create_figure(figsize=(8, 6))
        plt.streamplot(
            x1_unique,
            x2_unique,
            u_component,
            v_component,
            color="k",
            linewidth=1.0,
            density=3.0,
            arrowsize=0,
        )
        plt.title("Streamlines")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.axis("equal")
        plt.tight_layout()
        plt.savefig("results/streamlines.png", dpi=300, bbox_inches="tight")
        plt.close(fig)

    def _write_flow_field_data(self, x1_grid, x2_grid, u_pred, ne: int) -> None:
        np1, np2 = x1_grid.shape
        with open("results/flow_field.dat", "w") as file_obj:
            file_obj.write('TITLE = "Flow Field Data"\n')
            file_obj.write('VARIABLES = "X", "Y", "U", "V"')
            if ne >= 3:
                file_obj.write(', "P"')
            file_obj.write("\n")
            file_obj.write(f"ZONE I={np1}, J={np2}, F=POINT\n")
            for j in range(np2):
                for i in range(np1):
                    line = f"{x1_grid[i,j]:.6f} {x2_grid[i,j]:.6f} {u_pred[i,j,0]:.6f} {u_pred[i,j,1]:.6f}"
                    if ne >= 3:
                        line += f" {u_pred[i,j,2]:.6f}"
                    file_obj.write(line + "\n")

    def examin_data(self, data_train: Dict) -> None:
        """CheckData分布"""
        fig = self._create_figure()
        plt.scatter(data_train["x_boundary"][:, 0], data_train["x_boundary"][:, 1], label="BoundarySolution")
        plt.scatter(data_train["x"][:, 0], data_train["x"][:, 1], label="ExactSolution")
        plt.legend()
        plt.show()
        self._close_figure(fig)

    def examin_net(self, model: torch.nn.Module, data_GPU: torch.Tensor) -> None:
        """CheckNetworkOutput并绘制流场"""
        _ = data_GPU
        model.eval()
        with torch.no_grad():
            N = 100
            x1_grid = np.linspace(self.config.x_domain[0][0], self.config.x_domain[0][1], N)
            x2_grid = np.linspace(self.config.x_domain[1][0], self.config.x_domain[1][1], N)
            X1, X2 = np.meshgrid(x1_grid, x2_grid)
            grid_points = np.column_stack((X1.ravel(), X2.ravel()))
            grid_points = torch.tensor(
                grid_points,
                dtype=torch.float64,
                device=self.config.device,
                requires_grad=True,
            )
            _, u_pred = model(grid_points)
            U = u_pred[..., 0].detach().cpu().numpy().reshape(N, N)
            fig = self._create_figure(figsize=(15, 5))
            ax1 = fig.add_subplot(111)
            cs1 = ax1.contourf(X1, X2, U, levels=50, cmap="RdBu")
            plt.colorbar(cs1, ax=ax1, label="USpeed")
            ax1.set_title("USpeed")
            ax1.set_xlabel("x")
            ax1.set_ylabel("y")
            plt.tight_layout()
            plt.show()
            self._close_figure(fig)
