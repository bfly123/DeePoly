import os

import numpy as np


class DatasetVisualizationSupportMixin:
    def _render_dataset_solution(
        self,
        output_name: str,
        dataset: str,
        enabled: bool,
        predictions: dict,
        data: dict,
        results_dir: str,
        plotter,
    ) -> dict:
        if not enabled or output_name not in predictions:
            return {}

        prepared = self._prepare_plot_data(data)
        if prepared is None:
            print(
                f"  Warning: Missing coordinates for {dataset} dataset "
                f"({output_name}), skip plotting."
            )
            return {}

        values = self._to_numpy_cpu(predictions[output_name])
        if values is None:
            return {}

        save_path = os.path.join(
            results_dir,
            self._build_output_filename(output_name, dataset),
        )
        plotter.plot_solution(
            values,
            prepared,
            save_path,
            **self._build_plot_kwargs(dataset=dataset),
        )
        return {
            dataset: {
                "save_path": save_path,
                "data": prepared,
            }
        }

    def _prepare_plot_data(self, data: dict):
        """Ensure the plot payload contains global coordinates."""
        if not data:
            return None

        if "x" in data and data["x"] is not None:
            return data

        x_segments = data.get("x_segments")
        if x_segments:
            try:
                x_array = np.vstack(x_segments)
            except ValueError:
                x_array = np.concatenate(x_segments, axis=0)
            prepared = dict(data)
            prepared["x"] = x_array
            return prepared

        return None

    def _build_output_filename(self, output_name: str, dataset: str) -> str:
        vis_cfg = getattr(self.config, "visualization")
        if dataset == "test":
            return f"{output_name}_solution.png"
        if vis_cfg.should_plot_test():
            return f"{output_name}_train_solution.png"
        return f"{output_name}_solution.png"

    def _build_plot_kwargs(self, dataset: str) -> dict:
        """Build plotting kwargs for DimensionalVisualizer."""
        vis_cfg = getattr(self.config, "visualization")
        kwargs = {
            "figsize": (10, 8),
            "colormap": vis_cfg.colormap,
            "dpi": vis_cfg.figure_dpi,
        }

        if dataset == "train":
            kwargs["plot_mode"] = getattr(vis_cfg, "plot_mode_train", "scatter")
        else:
            kwargs["plot_mode"] = getattr(vis_cfg, "plot_mode_test", "auto")

        scatter_kwargs = self._merge_dataset_kwargs(vis_cfg, dataset, "scatter_kwargs")
        if scatter_kwargs:
            kwargs["scatter_kwargs"] = scatter_kwargs

        grid_kwargs = self._merge_dataset_kwargs(vis_cfg, dataset, "grid_kwargs")
        if grid_kwargs:
            kwargs["grid_kwargs"] = grid_kwargs

        return kwargs

    @staticmethod
    def _merge_dataset_kwargs(vis_cfg, dataset: str, attr_name: str) -> dict:
        merged = {}
        base_kwargs = getattr(vis_cfg, attr_name, None)
        if base_kwargs:
            merged.update(base_kwargs)

        dataset_kwargs = getattr(vis_cfg, f"{dataset}_{attr_name}", None)
        if dataset_kwargs:
            merged.update(dataset_kwargs)
        return merged
