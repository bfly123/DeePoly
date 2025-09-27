#!/usr/bin/env python3
"""
Automatic Code Generation Manager

This module provides the AutoCodeManager class for handling automatic code generation
and consistency checking in the DeePoly framework. It manages the workflow of:
- Detecting when code regeneration is needed
- Triggering code generation via AutoCodeGenerator
- Handling process restart after code generation
"""

import json
import os
import sys
import subprocess
import time
from typing import Dict, Tuple, Optional

from .auto_spotter import AutoCodeGenerator


class AutoCodeManager:
    """
    Manages automatic code generation and consistency checking

    This class provides a high-level interface for managing the automatic code
    generation workflow, including consistency checking between configuration
    files and generated code, triggering regeneration when needed, and handling
    process restart.
    """

    def __init__(self, case_dir: str):
        """
        Initialize the AutoCodeManager

        Args:
            case_dir: Path to the case directory containing config.json
        """
        self.case_dir = case_dir
        self.config_path = os.path.join(case_dir, "config.json")
        self.config_dict = self._load_config()
        self.problem_type = self.config_dict.get("problem_type", None)
        self._src_dir = self._get_src_dir()

    def _get_src_dir(self) -> str:
        """Get the source directory path"""
        current_file = os.path.abspath(__file__)
        meta_coding_dir = os.path.dirname(current_file)
        return os.path.dirname(meta_coding_dir)

    def _load_config(self) -> Dict:
        """
        Load configuration from JSON file

        Returns:
            Dictionary containing the configuration data

        Raises:
            FileNotFoundError: If config.json doesn't exist
            json.JSONDecodeError: If config.json is malformed
        """
        try:
            with open(self.config_path, "r") as f:
                return json.load(f)
        except FileNotFoundError:
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")
        except json.JSONDecodeError as e:
            raise json.JSONDecodeError(f"Malformed configuration file: {self.config_path}", e.doc, e.pos)

    def _get_net_path(self) -> Optional[str]:
        """
        Get the path to net.py based on problem type

        Returns:
            Path to the net.py file, or None if not a PDE problem
        """
        if self.problem_type not in ["time_pde", "linear_pde"]:
            return None

        solver_dir = "time_pde_solver" if self.problem_type == "time_pde" else "linear_pde_solver"
        return os.path.join(self._src_dir, "problem_solvers", solver_dir, "core", "net.py")

    def _save_config(self) -> None:
        """Save the current configuration back to file"""
        with open(self.config_path, "w") as f:
            json.dump(self.config_dict, f, indent=4, ensure_ascii=False)

    def check_consistency(self) -> Tuple[bool, str]:
        """
        Check if config and generated code are consistent

        Uses the AutoCodeGenerator to perform detailed consistency checking
        between the configuration file and the generated code in net.py.

        Returns:
            Tuple of (needs_regeneration, reason_message)
        """
        net_path = self._get_net_path()
        if not net_path:
            return False, "Not a PDE problem type"

        try:
            generator = AutoCodeGenerator(self.config_path)
            need_regenerate, reason = generator.check_config_net_consistency(net_path)

            if not need_regenerate:
                print(f"✓ {reason}")

            return need_regenerate, reason

        except Exception as e:
            print(f"Error during configuration check: {e}")
            return True, "Error during detection process"

    def should_force_regeneration(self) -> bool:
        """
        Check if user explicitly requested code regeneration via config

        Returns:
            True if auto_code is explicitly set to true in config
        """
        return self.config_dict.get("auto_code", False)

    def generate_code(self) -> bool:
        """
        Generate code for the current configuration

        This method temporarily enables auto_code if needed, initializes the
        appropriate solver to trigger code generation, and handles the expected
        SystemExit that occurs after successful generation.

        Returns:
            True if code was generated and restart is needed

        Raises:
            Exception: If code generation fails
        """
        original_auto_code = self.config_dict.get("auto_code", False)

        # Temporarily enable auto_code if needed
        if not original_auto_code:
            self.config_dict["auto_code"] = True
            self._save_config()
            print("Temporarily enabling auto_code for code generation...")

        try:
            # Initialize solver to trigger auto code generation
            # The solver initialization will trigger code generation and exit
            success = self._trigger_code_generation()

            if not success:
                print("Code generation should be complete but didn't exit normally")
                return False

        except SystemExit as e:
            if e.code == 0:
                print("✓ Auto code generation completed")
                # Restore original auto_code setting after successful generation
                if not original_auto_code:
                    self.config_dict["auto_code"] = False
                    self._save_config()
                return True
            raise

        except Exception as e:
            print(f"Error during auto code generation: {e}")
            # Restore original auto_code setting on error
            if not original_auto_code:
                self.config_dict["auto_code"] = False
                self._save_config()
            raise

        return False

    def _trigger_code_generation(self) -> bool:
        """
        Trigger code generation by importing and initializing the appropriate solver

        Returns:
            True if generation was triggered successfully
        """
        if self.problem_type == "time_pde":
            from problem_solvers.time_pde_solver.utils import TimePDEConfig
            from problem_solvers import TimePDESolver

            config = TimePDEConfig(case_dir=self.case_dir)
            _ = TimePDESolver(config)

        elif self.problem_type == "linear_pde":
            from problem_solvers.linear_pde_solver.utils import LinearPDEConfig
            from problem_solvers import LinearPDESolver

            config = LinearPDEConfig(case_dir=self.case_dir)
            _ = LinearPDESolver(config=config, case_dir=self.case_dir)

        else:
            return False

        return True

    def restart_process(self) -> None:
        """
        Restart the Python process to load newly generated code

        This method builds the restart command with appropriate arguments
        and starts a new process while terminating the current one.
        """
        time.sleep(0.5)  # Wait for filesystem sync

        print("\n=== Restarting process to load newly generated code ===\n")

        # Build restart command
        python_exe = sys.executable
        new_cmd = [python_exe] + sys.argv

        # Start new process and exit current one
        subprocess.call(new_cmd)
        sys.exit(0)

    def handle_autocode_workflow(self) -> bool:
        """
        Handle the complete automatic code generation workflow

        This is the main entry point that orchestrates the entire workflow:
        1. Check if problem type supports auto code
        2. Check consistency between config and generated code
        3. Generate code if needed
        4. Restart process if code was generated

        Returns:
            True if process was restarted, False otherwise
        """
        # Check if problem type supports auto code
        if self.problem_type not in ["time_pde", "linear_pde"]:
            return False

        # Check consistency between config and generated code
        need_regenerate, reason = self.check_consistency()

        if need_regenerate:
            print(f"Code regeneration needed: {reason}")
            print("=== Running automatic code generation ===")

            if self.generate_code():
                self.restart_process()
                return True  # This line won't be reached due to sys.exit()

        # Check if user explicitly requested code generation via config
        elif self.should_force_regeneration():
            print("Config file has auto_code=true, forcing code regeneration...")
            print("=== Running code generation ===")

            # Reset auto_code flag before generation to prevent infinite loops
            self.config_dict["auto_code"] = False
            self._save_config()

            if self.generate_code():
                self.restart_process()
                return True  # This line won't be reached due to sys.exit()

        return False