from typing import Dict, Any, List, Optional
import logging
from pathlib import Path
from scripts.logger import get_logger
from scripts.mcp_integration import MCPIntegrationHub
from scripts.http_ai_client import HTTPAIClient
from scripts.research_knowledge_store import ResearchKnowledgeStore
from scripts.config_loader import ConfigLoader
from scripts.error_handler import ErrorHandler
from scripts.mcp_config import MCPConfigManager

class SmartResearch:
    """
    A tool for conducting smart research using various components like HTTP AI clients,
    knowledge stores, and MCP integrations.  Handles errors and logs activity.
    """

    def __init__(
        self,
        config_manager: MCPConfigManager,
        http_ai_client: HTTPAIClient,
        knowledge_store: ResearchKnowledgeStore,
        mcp_integration: MCPIntegrationHub,
        error_handler: ErrorHandler,
        logger: logging.Logger = None,
    ):
        """
        Initializes the SmartResearch tool.

        Args:
            config_manager: The MCPConfigManager for accessing configurations.
            http_ai_client: The HTTPAIClient for interacting with AI models.
            knowledge_store: The ResearchKnowledgeStore for storing and retrieving research data.
            mcp_integration: The MCPIntegrationHub for interacting with MCP services.
            error_handler: The ErrorHandler for handling errors.
            logger: Optional logger instance.  If None, one will be created.
        """
        self.config_manager = config_manager
        self.http_ai_client = http_ai_client
        self.knowledge_store = knowledge_store
        self.mcp_integration = mcp_integration
        self.error_handler = error_handler
        self.logger = logger or get_logger(__name__)


    def run_research(self, query: str) -> Dict[str, Any]:
        """
        Executes a research query using the configured components.

        Args:
            query: The research query to execute.

        Returns:
            A dictionary containing the research results or an error message.  'success' indicates
            successful research, while 'error' indicates a failure.
        """
        try:
            print(f"Starting research for query: {query}")

            # Example: Load API Key and KnowledgeStore Name from the config
            api_key = self.config_manager.get_config("api_key")
            knowledge_store_name = self.config_manager.get_config("knowledge_store_name")

            # Example: Interact with the HTTP AI Client
            try:
                ai_response = self.http_ai_client.generate_response(query=query, api_key=api_key)
            except Exception as e:
                error_message = f"Error during AI Client interaction: {e}"
                self.logger.error(error_message)
                self.error_handler.handle_error(error_message)
                return {"error": error_message}

            # Example: Store information in KnowledgeStore.
            try:
                self.knowledge_store.store_data(knowledge_store_name, query, ai_response)
            except Exception as e:
                error_message = f"Error storing data in Knowledge Store: {e}"
                self.logger.error(error_message)
                self.error_handler.handle_error(error_message)
                return {"error": error_message}

            # Example: Interact with MCP Integration Hub
            try:
                mcp_result = self.mcp_integration.send_message(f"Research query: {query}")
                print(f"MCP Interaction Result: {mcp_result}")

            except Exception as e:
                error_message = f"Error during MCP Integration: {e}"
                self.logger.warning(error_message) # MCP is optional so we use a warning
                self.error_handler.handle_error(error_message, is_critical=False) #MCP is optional.

            print(f"Successfully completed research for query: {query}")
            return {"success": ai_response}

        except Exception as e:
            error_message = f"An unexpected error occurred during research: {e}"
            self.logger.exception(error_message)
            self.error_handler.handle_error(error_message) #Critical error during main process
            return {"error": error_message}
