from typing import Dict, Any, List, Optional
import logging
from pathlib import Path
from scripts.state_manager import StateManager
from scripts.logger import get_logger
from scripts.config_loader import ConfigLoader
from scripts.error_handler import ErrorHandler
from scripts.mcp_config import MCPConfig
from scripts.mcp_config import MCPConfigManager


class PerformanceAnalyzer:
    """
    A general-purpose tool for analyzing performance data.
    Handles error handling, logging, and state persistence.
    """

    def __init__(self, config_path: str, state_path: str):
        """
        Initializes the PerformanceAnalyzer.

        Args:
            config_path (str): Path to the configuration file.
            state_path (str): Path to the state file.
        """
        self.config_path = config_path
        self.state_path = state_path
        self.logger = get_logger(__name__)
        self.config_loader = ConfigLoader(self.config_path)
        self.config = self.config_loader.load_config()
        self.state_manager = StateManager(self.state_path)
        self.error_handler = ErrorHandler()
        self.state: Dict[str, Any] = self.state_manager.load_state() or {}

    def analyze_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyzes the provided data and performs some performance analysis.

        Args:
            data (Dict[str, Any]): The data to analyze.

        Returns:
            Dict[str, Any]: A dictionary containing the analysis results or error information.
        """
        try:
            print("Starting data analysis.")
            # Example analysis (replace with actual analysis logic)
            if not isinstance(data, dict):
                raise ValueError("Input data must be a dictionary.")

            if "performance_metrics" not in data:
                raise ValueError("Input data must contain 'performance_metrics'.")

            performance_metrics = data["performance_metrics"]

            if not isinstance(performance_metrics, dict):
                raise ValueError("'performance_metrics' must be a dictionary.")
            
            analysis_results = {}
            for metric, value in performance_metrics.items():
                if isinstance(value, (int, float)):
                    analysis_results[metric] = value * 1.1  # Simple example calculation
                else:
                    analysis_results[metric] = "N/A"
            
            self.state["last_analysis"] = analysis_results  # Update state
            self.state_manager.state = self.state
            self.state_manager.save_state(self.state)
            print("Data analysis completed successfully.")

            return {"success": True, "results": analysis_results}

        except ValueError as ve:
            self.logger.error(f"ValueError during data analysis: {ve}")
            self.error_handler.handle_error(f"ValueError during data analysis: {ve}", traceback=None)  # No traceback available in this simplified example
            return {"error": f"ValueError: {ve}"}
        except Exception as e:
            self.logger.exception(f"An unexpected error occurred during data analysis: {e}")
            self.error_handler.handle_error(f"An unexpected error occurred: {e}", traceback=None) # No traceback available in this simplified example
            return {"error": f"An unexpected error occurred: {e}"}

    def get_state(self) -> Dict[str, Any]:
        """
        Returns the current state of the analyzer.

        Returns:
            Dict[str, Any]: The current state.
        """
        return self.state

    def set_state(self, new_state: Dict[str, Any]) -> None:
        """
        Sets the current state of the analyzer.

        Args:
            new_state (Dict[str, Any]): The new state to set.
        """
        try:
            if not isinstance(new_state, dict):
                raise ValueError("New state must be a dictionary.")
            self.state = new_state
            self.state_manager.state = self.state
            self.state_manager.save_state(self.state)
        except ValueError as ve:
            self.logger.error(f"ValueError during setting state: {ve}")
            self.error_handler.handle_error(f"ValueError during setting state: {ve}", traceback=None) # No traceback available in this simplified example
        except Exception as e:
            self.logger.exception(f"An unexpected error occurred during setting state: {e}")
            self.error_handler.handle_error(f"An unexpected error occurred during setting state: {e}", traceback=None) # No traceback available in this simplified example

    def run(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Runs the performance analysis workflow.

        Args:
            data (Dict[str, Any]): The input data for analysis.

        Returns:
            Dict[str, Any]: A dictionary containing the analysis results or error information.
        """
        return self.analyze_data(data)

if __name__ == '__main__':
    # Example usage
    config_path = "config.yaml"  # Replace with your actual config path
    state_path = "state.json"  # Replace with your actual state path

    # Create dummy config.yaml
    with open(config_path, "w") as f:
        f.write("dummy_config: true")

    analyzer = PerformanceAnalyzer(config_path, state_path)

    data = {"performance_metrics": {"cpu_usage": 75.0, "memory_usage": 60.0}}
    results = analyzer.run(data)

    print("Analysis Results:", results)
    print("Current State:", analyzer.get_state())

    # Clean up dummy config
    Path(config_path).unlink()
    #Clean up dummy state file
    Path(state_path).unlink(missing_ok=True)
