import matplotlib.pyplot as plt
import numpy as np
import os
import time
import torch
from typing import Dict, Optional, List, Any
from data_generate import generate_global_field
def calculate_errors(true_values, predictions):
    """Calculate various error metrics"""
    mse = np.mean((predictions - true_values)**2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(predictions - true_values))
    max_error = np.max(np.abs(predictions - true_values))
    
    return {
        'MSE': mse,
        'RMSE': rmse,
        'MAE': mae,
        'MAX_ERROR': max_error
    }

def plot_solution(data: Dict, prediction: np.ndarray, save_path: Optional[str] = None):
    """Plot function fitting results
    
    Args:
        data: Dictionary containing input data
        prediction: Prediction results
        save_path: Save path
    """
    print("Using custom plotting function...")
    x_test = data['x']  # Assume only one test segment
    y_true = data['u']  # True values (may have noise)
    
    # Try to get or calculate true noise-free function values (generated by data_generate.py)
        # If data_generate.py doesn't provide y_exact_segments, use theoretical formula
    y_exact = np.sin(2 * np.pi * x_test) + 0.5 * np.cos(4 * np.pi * x_test)

    y_exact  = generate_global_field(x_test)

    # Calculate absolute error
    abs_error = np.abs(y_exact - prediction)

    # Create plot with 2 subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10), gridspec_kw={'height_ratios': [3, 1]})
    
    # Plot data points, true function and prediction results
    ax1.scatter(x_test, y_true, label='Noisy Data', s=10, alpha=0.6)
    ax1.plot(x_test, y_exact, 'g-', label='True Function', linewidth=2)
    ax1.plot(x_test, prediction, 'r--', label='Prediction', linewidth=2)
    
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.set_title('Function Fitting Results')
    ax1.legend()
    ax1.grid(True)
    
    # Plot absolute error in log scale
    ax2.semilogy(x_test, abs_error, 'b-', label='Absolute Error', linewidth=1.5)
    ax2.set_xlabel('x')
    ax2.set_ylabel('|Error| (log scale)')
    ax2.set_title('Pointwise Absolute Error (Log Scale)')
    ax2.grid(True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        print(f"Image saved to {save_path}")
        
        # Save error data to dat file
        error_data_path = os.path.splitext(save_path)[0] + '_error.dat'
        error_data = np.column_stack((x_test, abs_error))
        np.savetxt(error_data_path, error_data, header='x error', comments='')
        print(f"Error data saved to {error_data_path}")
    else:
        plt.show()
    plt.close()

def generate_output(output_data: Dict[str, Any]):
    """Generate all output results and visualizations
    
    Args:
        output_data: Dictionary containing all required data
    """
    # Unpack data
    train_data = output_data['train_data']
    test_data = output_data['test_data']
    train_predictions = output_data['train_predictions']
    test_predictions = output_data['test_predictions']
    train_segments = output_data['train_segments']
    test_segments = output_data['test_segments']
    coeffs = output_data['coeffs']
    model = output_data['model']
    config = output_data['config']
    result_dir = output_data['result_dir']
    
    # Get final loss value and solution time if available
    final_loss = output_data.get('final_loss', None)
    solution_time = output_data.get('solution_time', None)
    
    # Ensure result directory exists
    os.makedirs(result_dir, exist_ok=True)
    
    # 1. Calculate error metrics
    # Check if u exists in the data, if not try to build it from u_segments
   
    train_errors = calculate_errors(train_data['u'], train_predictions)
    test_errors = calculate_errors(test_data['u'], test_predictions)
    
    # 2. Save error metrics to text file
    with open(os.path.join(result_dir, "error_metrics.txt"), "w") as f:
        f.write("=== Training Set Errors ===\n")
        for name, value in train_errors.items():
            f.write(f"{name}: {value:.8e}\n")
        
        f.write("\n=== Test Set Errors ===\n")
        for name, value in test_errors.items():
            f.write(f"{name}: {value:.8e}\n")
        
        # Add final loss and solution time information
        f.write("\n=== Solution Information ===\n")
        if final_loss is not None:
            f.write(f"Final Training Loss: {final_loss:.8e}\n")
        if solution_time is not None:
            f.write(f"Total Solution Time: {solution_time:.4f} seconds\n")
    
    # 3. Generate visualization results
    # Only plot test set results
    test_vis_path = os.path.join(result_dir, "test_results.png")
    plot_solution(test_data, test_predictions, test_vis_path)
    
    # 4. Save coefficients and model
    np.save(os.path.join(result_dir, "coefficients.npy"), coeffs)
    torch.save(model.state_dict(), os.path.join(result_dir, "model.pt"))
    
    # 5. Save source data
    try:
        np.save(os.path.join(result_dir, "train_data.npy"), train_data)
        np.save(os.path.join(result_dir, "test_data.npy"), test_data)
    except Exception as e:
        print(f"Warning: Error saving data: {e}")
    
    # 6. Output result summary
    print("\n=== Fitting Results Summary ===")
    print(f"Training Set MSE: {train_errors['MSE']:.6e}")
    print(f"Test Set MSE: {test_errors['MSE']:.6e}")
    print(f"Test Set RMSE: {test_errors['RMSE']:.6e}")
    
    # Print final loss and solution time information
    if final_loss is not None:
        print(f"Final Training Loss: {final_loss:.6e}")
    if solution_time is not None:
        print(f"Total Solution Time: {solution_time:.4f} seconds")
        
    print(f"Results saved to: {result_dir}")