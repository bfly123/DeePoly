import os
import torch
import numpy as np
from typing import List, Dict, Optional
from src.problem_solvers.linear_pde_solver.auto_replace_loss import update_physics_loss_code
from src.meta_coding.auto_eq import parse_equation_to_list

def auto_code_scopper(config):
    """
    Generate PDE loss code based on equation configuration
    
    Args:
        config: Configuration object with equation information
    """
    print("Auto-generating PDE loss code from equation...")
    
    # Get equation information from config
    linear_equations = config.eq
    nonlinear_equations = config.eq_nonlinear if hasattr(config, "eq_nonlinear") else []
    vars_list = config.vars_list
    spatial_vars = config.spatial_vars
    
    # Prepare constants for the code generation
    const_list = []
    if hasattr(config, "const_list") and config.const_list:
        const_list = config.const_list
    
    # Handle source term if present
    if hasattr(config, "source_term") and config.source_term:
        if isinstance(config.source_term, str):
            # If source term is a string (expression), it will be handled 
            # by the parser directly in the generated code
            pass
        elif config.source_term is True:
            # If source term is True, we'll use the pre-loaded source term
            pass
    
    # Define the path to the network model file
    model_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "linear_pde_solver/core/net.py"
    )
    
    # Check if the file exists
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Network model file not found: {model_path}")
    
    # Update the loss function code
    try:
        # First backup the file
        with open(model_path, 'r') as f:
            original_content = f.read()
            
        # Call update_physics_loss_code to generate the initial code
        update_physics_loss_code(
            linear_equations=linear_equations,
            nonlinear_equations=nonlinear_equations,
            vars_list=vars_list,
            spatial_vars=spatial_vars,
            const_list=const_list,
            model_path=model_path
        )
        
        print(f"Generated initial PDE loss code in {model_path}")
        
        # Now modify the file to fix the formatting and handle source terms
        _modify_auto_generated_code(model_path, original_content)
        
        return True
    except Exception as e:
        print(f"Error generating PDE loss code: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def _modify_auto_generated_code(model_path, original_content=None):
    """
    Process the auto-generated code to fix formatting and handle source terms
    
    Args:
        model_path: Path to the network model file
        original_content: Optional original content for backup
    """
    try:
        # Read the current content
        with open(model_path, 'r') as f:
            content = f.read()
        
        # Find the auto code block
        start_marker = "# auto code begin"
        end_marker = "# auto code end"
        
        if start_marker not in content or end_marker not in content:
            print("Warning: Could not find auto code block in net.py")
            if original_content:
                # Restore original content
                with open(model_path, 'w') as f:
                    f.write(original_content)
            return
        
        # Split content at key markers
        parts = content.split(start_marker)
        before_auto = parts[0]
        
        auto_and_after = parts[1].split(end_marker)
        auto_code = auto_and_after[0]
        after_auto = end_marker + auto_and_after[1]
        
        # Don't modify the before_auto part - x_train will be provided by the user
        mod_before_auto = before_auto
        
        # Fix auto-generated code: convert eq0 to eq[0] format, don't add any additional calculations
        auto_code_lines = []
        for line in auto_code.strip().split('\n'):
            if not line.strip():
                auto_code_lines.append("        ")
                continue
                
            # Replace eq0 with eq[0] format
            fixed_line = line
            for i in range(10):  # Handle eq0 through eq9
                if f"eq{i} =" in line:
                    if i == 0:
                        fixed_line = line.replace(f"eq{i} =", "eq = [")
                        fixed_line += "]"
                    else:
                        if "eq =" not in auto_code:
                            fixed_line = line.replace(f"eq{i} =", f"eq = [None] * 10\neq[{i}] =")
                        else:
                            fixed_line = line.replace(f"eq{i} =", f"eq[{i}] =")
            
            # Add proper indentation
            indented_line = "        " + fixed_line.lstrip() if fixed_line.strip() else "        "
            auto_code_lines.append(indented_line)
        
        # Process content after auto code block
        after_lines = after_auto.split('\n')
        mod_after_lines = [after_lines[0]]  # Keep the end marker
        
        # Process remaining lines
        for i, line in enumerate(after_lines[1:]):
            # Fix any references to mse_loss or source loss
            if "total_loss = mse_loss" in line:
                mod_after_lines.append(line.replace("mse_loss", "pde_loss"))
            elif "print(f\"Source Loss: {mse_loss.item()" in line:
                continue  # Skip source loss print
            else:
                mod_after_lines.append(line)
        
        # Combine all parts
        new_content = mod_before_auto + start_marker + '\n' + '\n'.join(auto_code_lines) + '\n' + '\n'.join(mod_after_lines)
        
        # Write back to the file
        with open(model_path, 'w') as f:
            f.write(new_content)
        
        print("Successfully modified the generated PDE loss code")
    
    except Exception as e:
        print(f"Error modifying auto-generated code: {str(e)}")
        if original_content:
            print("Restoring original content...")
            with open(model_path, 'w') as f:
                f.write(original_content)
        
        import traceback
        traceback.print_exc()
