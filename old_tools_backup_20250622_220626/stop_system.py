from typing import Dict, Any, List, Optional
import logging
from pathlib import Path
from scripts.logger import get_logger
from scripts.error_handler import ErrorHandler
from scripts.config_loader import ConfigLoader

class StopSystem:
    """
    A general tool to stop the system, including error handling and logging.
    """

    def __init__(self, config: ConfigLoader):
        """
        Initializes the StopSystem tool.

        Args:
            config (ConfigLoader): The configuration object.
        """
        self.config = config
        self.logger = get_logger(__name__)
        self.error_handler = ErrorHandler()  # Initialize ErrorHandler

    def execute(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Executes the system stop operation.

        Args:
            params (Dict[str, Any]): A dictionary of parameters for the operation.
                                     Expected keys: 'reason' (str, required).

        Returns:
            Dict[str, Any]: A dictionary containing the result of the operation.
                            {'success': True} if the system stop was initiated successfully,
                            {'error': 'error message'} otherwise.
        """
        try:
            reason = params.get("reason")
            if not reason:
                raise ValueError("Reason for stopping the system must be provided.")

            print(f"Initiating system stop. Reason: {reason}")

            # Placeholder for actual system stop logic.
            # Replace this with the actual code to stop the system.
            # This could involve shutting down processes, deallocating resources, etc.

            # Log a message to indicate the system stop was initiated.
            print("System stop initiated successfully.")

            return {"success": True}

        except Exception as e:
            error_message = f"Error stopping the system: {str(e)}"
            self.logger.error(error_message)
            self.error_handler.handle_error(error_message)
            return {"error": error_message}
