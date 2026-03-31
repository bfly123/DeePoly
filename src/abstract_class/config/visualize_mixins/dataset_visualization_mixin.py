from .dataset_visualization_support_mixin import DatasetVisualizationSupportMixin


class DatasetVisualizationMixin(DatasetVisualizationSupportMixin):
    """通用数据集可视化混入，负责根据配置绘制 train/test 解图。"""

    def render_solution_outputs(
        self,
        outputs,
        predictions_test,
        predictions_train,
        data_test,
        data_train,
        results_dir: str,
    ):
        """根据配置绘制输出并返回绘图信息。"""
        vis_cfg = getattr(self.config, "visualization", None)
        if vis_cfg is None:
            raise AttributeError("Visualization configuration is missing.")

        plotter = getattr(self, "dimensional_visualizer", None)
        if plotter is None:
            raise AttributeError("dimensional_visualizer is not initialized.")

        render_info = {}

        for output_name in outputs:
            output_info = {}
            output_info.update(
                self._render_dataset_solution(
                    output_name=output_name,
                    dataset="test",
                    enabled=vis_cfg.should_plot_test(),
                    predictions=predictions_test,
                    data=data_test,
                    results_dir=results_dir,
                    plotter=plotter,
                )
            )
            output_info.update(
                self._render_dataset_solution(
                    output_name=output_name,
                    dataset="train",
                    enabled=vis_cfg.should_plot_train(),
                    predictions=predictions_train,
                    data=data_train,
                    results_dir=results_dir,
                    plotter=plotter,
                )
            )

            if output_info:
                render_info[output_name] = output_info

        return render_info
