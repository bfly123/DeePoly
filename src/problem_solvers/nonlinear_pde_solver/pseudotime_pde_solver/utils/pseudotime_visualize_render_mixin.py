from typing import Dict, Optional


class PseudotimeVisualizeRenderMixin:
    def render(
        self,
        results: Dict,
        data_train: Dict,
        data_test: Dict,
        reference_solution: Optional[Dict] = None,
    ):
        results_dir = self.config.results_dir
        vis_cfg = self.config.visualization
        outputs = vis_cfg.outputs
        predictions_test = results["predictions"]
        predictions_train = results.get("predictions_train", {})

        self._save_convergence_plots(results["convergence_info"], results_dir)
        render_info = self.render_solution_outputs(
            outputs,
            predictions_test,
            predictions_train,
            data_test,
            data_train,
            results_dir,
        )
        self._render_reference_outputs(
            outputs,
            predictions_test,
            render_info,
            reference_solution,
            vis_cfg,
            results_dir,
        )
        self._save_flow_diagnostics_for_test_outputs(
            outputs,
            predictions_test,
            render_info,
            results_dir,
            results.get("flow_diagnostics"),
        )
        self._finalize_animation(results_dir)
        self._save_data(results, data_train, data_test, results_dir)
        print(f"\n✓ All visualizations saved to: {results_dir}")

    def _render_reference_outputs(
        self,
        outputs,
        predictions_test,
        render_info,
        reference_solution,
        vis_cfg,
        results_dir: str,
    ) -> None:
        if not (reference_solution and self._should_plot_reference()):
            return
        if not vis_cfg.should_plot_test():
            print("  Warning: Reference comparison requires test dataset visualization; skipped.")
            return
        for output_name in outputs:
            if output_name not in predictions_test:
                continue
            test_info = render_info.get(output_name, {}).get("test")
            if not test_info:
                continue
            coords = test_info["data"]["x"]
            self._save_reference_comparison(
                output_name,
                coords,
                predictions_test[output_name],
                reference_solution,
                results_dir,
            )
