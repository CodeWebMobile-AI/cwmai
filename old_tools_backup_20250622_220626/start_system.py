from typing import Dict, Any, List, Optional
import logging
from pathlib import Path
from scripts.state_manager import StateManager
from scripts.logger import get_logger
from scripts.error_handler import ErrorHandler
from scripts.mcp_config import MCPConfigManager

class StartSystem:
    """
    A general tool for starting the system, handling errors, logging, and state persistence.
    """

    def __init__(
        self,
        state_manager: StateManager,
        config_manager: MCPConfigManager,
        error_handler: ErrorHandler,
    ) -> None:
        """
        Initializes the StartSystem tool.

        Args:
            state_manager: The StateManager instance for managing state persistence.
            config_manager: The MCPConfigManager instance for managing configurations.
            error_handler: The ErrorHandler instance for handling errors.
        """
        self.state_manager = state_manager
        self.config_manager = config_manager
        self.error_handler = error_handler
        self.logger = get_logger(__name__)  # Obtain logger from scripts.logger

    def run(self, initial_data: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Starts the system, performs initial setup, and manages state.

        Args:
            initial_data: Optional initial data to load into the system state.

        Returns:
            A dictionary indicating success or error.
        """
        try:
            print("Starting the system...")

            # Load initial state if provided
            if initial_data:
                self.state_manager.state = initial_data

            # Load configurations
            config = self.config_manager.get_config()  # Assuming get_config returns a dict

            # Perform initial setup tasks (example)
            print("Performing initial setup...")
            # Example: Set up database connection (replace with actual logic)
            # db_connection = config.get("database_url")

            # Update the state with some initial information
            self.state_manager.state["system_status"] = "running"
            self.state_manager.state["config_version"] = config.get("version", "1.0")

            # Save the initial state
            self.state_manager.save_state()
            print("Initial state saved.")

            return {"success": True, "message": "System started successfully."}

        except Exception as e:
            error_message = f"Error starting the system: {str(e)}"
            self.logger.exception(error_message)  # Log the full exception with traceback
            self.error_handler.handle_error(error_message, "start_system")

            return {"error": error_message}
