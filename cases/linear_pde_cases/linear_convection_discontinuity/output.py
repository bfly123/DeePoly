from data_generate import generate_global_field

def generate_output(config, data_train, data_test, fitter, model, coeffs, result_dir, visualizer, total_time, scoper_time, sniper_time, **kwargs):
    """
    Generate comprehensive output results and visualizations for function fitting problems
    
    Args:
        config: Configuration object containing problem parameters
        data_train: Training data dictionary
        data_test: Test data dictionary
        fitter: DeePoly fitter object
        model: Trained neural network model
        coeffs: DeePoly coefficients
        result_dir: Directory to save results
        visualizer: Visualization object
        total_time: Total solution time
        scoper_time: Neural network training time (Scoper phase)
        sniper_time: Equation fitting time (Sniper phase)
        **kwargs: Additional arguments
    """
    
    # Determine problem dimensionality and dispatch to appropriate analysis method
    if config.n_dim == 1:
        # Generate 1D function fitting analysis
        visualizer.generate_1d_analysis(
            data_train=data_train,
            data_test=data_test,
            model=model,
            fitter=fitter,
            coeffs=coeffs,
            exact_solution_func=generate_global_field,
            result_dir=result_dir,
            variables=None,  # Plot all variables by default
            timing_info={
                'total_time': total_time,
                'scoper_time': scoper_time,
                'sniper_time': sniper_time
            }
        )
        print(f"1D function fitting analysis completed.")
        
    elif config.n_dim == 2:
        # Generate 2D function fitting analysis
        visualizer.generate_2d_analysis(
            data_train=data_train,
            data_test=data_test,
            model=model,
            fitter=fitter,
            coeffs=coeffs,
            exact_solution_func=generate_global_field,
            result_dir=result_dir,
            variables=None,  # Plot all variables by default
            timing_info={
                'total_time': total_time,
                'scoper_time': scoper_time,
                'sniper_time': sniper_time
            }
        )
        print(f"2D function fitting analysis completed.")
        
    else:
        print(f"Warning: {config.n_dim}D visualization not implemented yet.")
        print("Generating basic error analysis without visualization...")
        
        # Basic error analysis without full visualization
        _generate_basic_error_analysis(
            data_train, data_test, model, fitter, coeffs, 
            generate_global_field, result_dir, visualizer,
            timing_info={
                'total_time': total_time,
                'scoper_time': scoper_time,
                'sniper_time': sniper_time
            }
        )
    
    print(f"Results saved to: {result_dir}")
    if config.n_dim <= 2:
        print(f"Visualizations saved to: {result_dir}/visualizations/")


def _generate_basic_error_analysis(data_train, data_test, model, fitter, coeffs, 
                                  exact_solution_func, result_dir, visualizer, timing_info):
    """Generate basic error analysis for higher dimensional problems"""
    import os
    import numpy as np
    
    # Get predictions
    net_train = visualizer.get_model_predictions(model, data_train)
    net_test = visualizer.get_model_predictions(model, data_test)
    exact_train = exact_solution_func(data_train["x"])
    exact_test = exact_solution_func(data_test["x"])
    deepoly_train = visualizer.get_deepoly_predictions(fitter, data_train, model, coeffs)
    deepoly_test = visualizer.get_deepoly_predictions(fitter, data_test, model, coeffs)
    
    # Calculate errors
    errors = {
        'net_train_errors': visualizer.calculate_errors(net_train, exact_train),
        'net_test_errors': visualizer.calculate_errors(net_test, exact_test),
        'deepoly_train_errors': visualizer.calculate_errors(deepoly_train, exact_train),
        'deepoly_test_errors': visualizer.calculate_errors(deepoly_test, exact_test)
    }
    
    # Print error statistics
    visualizer._print_error_statistics(**errors)
    
    # Save error analysis report
    visualizer._save_error_analysis_report(result_dir, errors, timing_info)
    
    # Save basic results
    os.makedirs(result_dir, exist_ok=True)
    np.save(os.path.join(result_dir, "coefficients.npy"), coeffs)
    import torch
    torch.save(model.state_dict(), os.path.join(result_dir, "model.pt"))