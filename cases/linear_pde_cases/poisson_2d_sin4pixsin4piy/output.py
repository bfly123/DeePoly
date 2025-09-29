from data_generate import generate_reference_solution

def generate_output(config, data_train, data_test, fitter, model, coeffs, result_dir, visualizer, total_time, scoper_time, sniper_time):
    """Generate output results and visualizations
    
    Args:
        config: Configuration object
        data_train: Training data
        data_test: Test data
        fitter: DeePoly fitter
        model: Trained model
        coeffs: DeePoly coefficients
        result_dir: Results save directory
        visualizer: Visualizer object
        total_time: Total solution time
        scoper_time: Neural network training time
        sniper_time: Equation fitting time
    """
    
    # Use visualizer for complete 2D comparison plotting
    # Default plots all physical quantities, can specify variables like ["u", "v", "p"]
    visualizer.plot_2d_comparison(
        data_train=data_train,
        data_test=data_test,
        model=model,
        fitter=fitter,
        coeffs=coeffs,
        exact_solution_func=generate_reference_solution,
        result_dir=result_dir,
        variables=None,  # Default plots all variables, can specify like ["u"] or ["u", "v", "p"]
        timing_info={
            'total_time': total_time,
            'scoper_time': scoper_time,
            'sniper_time': sniper_time
        }
    )
    
    print(f"Visualization results saved to: {result_dir}/visualizations/")