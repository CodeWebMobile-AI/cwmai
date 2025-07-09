"""
AI-Generated Tool: orchestrate_workflows
Description: Tool for orchestrating workflows
Generated: 2025-06-16
"""

from typing import Dict, Any, Optional, List, Union
import json
import asyncio
import traceback
from scripts.workflow_orchestrator import WorkflowOrchestrator
from scripts.state_manager import StateManager
from functools import lru_cache
import logging
import os

__description__ = "Orchestrate and execute workflows. Supports listing, executing, creating, deleting, getting, and updating workflows.  Includes versioning and caching. Provides improved error handling, logging, and utility functions."
__parameters__ = {
    "action": {
        "type": "string",
        "description": "Action to perform: list_workflows, execute_workflow, create_workflow, delete_workflow, get_workflow, update_workflow, validate_workflow, get_workflow_versions, clear_cache, get_workflow_history",
        "required": True
    },
    "workflow_name": {
        "type": "string",
        "description": "Name of the workflow",
        "required": False
    },
    "workflow_definition": {
        "type": "string",
        "description": "JSON workflow definition",
        "required": False
    },
    "workflow_input": {
        "type": "string",
        "description": "JSON input data for the workflow execution",
        "required": False
    },
    "version": {
        "type": "integer",
        "description": "Workflow version. If provided, only this version will be returned/updated/deleted. If not provided, defaults to the latest version.",
        "required": False
    },
    "step_name":{
        "type": "string",
        "description": "Name of the step in a workflow (used for specific error handling)",
        "required": False
    },
    "force": {
        "type": "boolean",
        "description": "Force delete workflow, even if execution is in progress.  Use with caution!",
        "required": False
    },
    "workflow_execution_id": {
        "type": "string",
        "description": "The ID of a specific workflow execution, used for retrieving its history.",
        "required": False
    }
}


# Configure logging
LOG_LEVEL = os.environ.get('LOG_LEVEL', 'INFO').upper()
logging.basicConfig(level=LOG_LEVEL, format='%(asctime)s - %(levelname)s - %(message)s')


# In-memory cache for workflows.  Consider using Redis or similar for persistence in a real-world scenario.
@lru_cache(maxsize=32)
def get_workflow_from_cache(workflow_name: str, version: Optional[int], state: Dict) -> Optional[Dict]:
    """Retrieves a workflow definition from the cached state."""
    logging.debug(f"Attempting to retrieve workflow '{workflow_name}' version '{version}' from cache.")
    workflows = state.get('workflows', {})
    if workflow_name in workflows:
        if version is not None:
            if version in workflows[workflow_name]:
                logging.debug(f"Workflow '{workflow_name}' version '{version}' found in cache.")
                return workflows[workflow_name][version]
            else:
                logging.debug(f"Workflow '{workflow_name}' version '{version}' not found in cache.")
                return None
        else: # Get latest version
            versions = sorted(workflows[workflow_name].keys(), reverse=True)
            if versions:
                logging.debug(f"Workflow '{workflow_name}' latest version '{versions[0]}' found in cache.")
                return workflows[workflow_name][versions[0]]
            else:
                logging.debug(f"No versions found for workflow '{workflow_name}' in cache.")
                return None
    else:
        logging.debug(f"Workflow '{workflow_name}' not found in cache.")
        return None


def clear_workflow_cache(workflow_name: str, version: Optional[int] = None):
    """Clears a specific workflow (or all versions of it) from the cache."""
    logging.info(f"Clearing workflow '{workflow_name}' version '{version}' from cache.")
    if version is None:
        # Invalidate all versions
        for i in range(get_workflow_from_cache.cache_info().maxsize):  # Invalidate cache for all versions
            try:
                get_workflow_from_cache.cache_invalidate((workflow_name, i, {})) # Include state as a dummy argument
                logging.debug(f"Invalidated cache for workflow '{workflow_name}' version '{i}'.")
            except KeyError:
                # Ignore if the key doesn't exist in the cache
                logging.debug(f"Cache key for workflow '{workflow_name}' version '{i}' not found, skipping.")
                pass
    else:
        try:
            get_workflow_from_cache.cache_invalidate((workflow_name, version, {})) # Include state as a dummy argument
            logging.debug(f"Invalidated cache for workflow '{workflow_name}' version '{version}'.")
        except KeyError:
            # Ignore if the key doesn't exist in the cache
            logging.debug(f"Cache key for workflow '{workflow_name}' version '{version}' not found, skipping.")
            pass

def validate_workflow_definition(workflow_definition: str) -> Dict:
    """Validates the workflow definition and returns it as a dictionary."""
    try:
        workflow_def = json.loads(workflow_definition)
    except json.JSONDecodeError as e:
        logging.error(f"Invalid JSON for workflow definition: {e}")
        raise ValueError(f"Invalid JSON for workflow definition: {e}")

    if not isinstance(workflow_def, dict):
        logging.error("Workflow definition must be a JSON object (dictionary).")
        raise ValueError("Workflow definition must be a JSON object (dictionary).")

    if 'steps' not in workflow_def or not isinstance(workflow_def['steps'], list):
        logging.error("Workflow definition must contain a 'steps' key with a list of steps.")
        raise ValueError("Workflow definition must contain a 'steps' key with a list of steps.")

    # Example validation: check each step has a 'type'
    for i, step in enumerate(workflow_def['steps']):
        if not isinstance(step, dict):
           logging.error(f"Step {i+1} in the workflow must be a JSON object (dictionary).")
           raise ValueError(f"Step {i+1} in the workflow must be a JSON object (dictionary).")
        if 'type' not in step:
            logging.error(f"Step {i+1} must have a 'type' attribute.")
            raise ValueError(f"Step {i+1} must have a 'type' attribute.")
        if 'name' not in step:
            logging.error(f"Step {i+1} must have a 'name' attribute.") #Enforce name attribute for better error handling.
            raise ValueError(f"Step {i+1} must have a 'name' attribute.") #Enforce name attribute for better error handling.
        if 'inputs' in step and not isinstance(step['inputs'], list):
            logging.error(f"The inputs key in step {i+1} must be a list")
            raise ValueError(f"The inputs key in step {i+1} must be a list")
        if 'retry' in step:
            if not isinstance(step['retry'], int):
                logging.error(f"The retry key in step {i+1} must be an integer")
                raise ValueError(f"The retry key in step {i+1} must be an integer")
            if step['retry'] < 0:
                logging.error(f"The retry key in step {i+1} must be a non-negative integer")
                raise ValueError(f"The retry key in step {i+1} must be a non-negative integer")

    logging.debug("Workflow definition validated successfully.")
    return workflow_def

def parse_workflow_input(workflow_input: Optional[str]) -> Dict:
    """Parses the workflow input and returns it as a dictionary.  Returns an empty dictionary if input is None."""
    if not workflow_input:
        return {}
    try:
        input_data = json.loads(workflow_input)
    except json.JSONDecodeError as e:
        logging.error(f"Invalid JSON for workflow_input: {e}")
        raise ValueError(f"Invalid JSON for workflow_input: {e}")

    if not isinstance(input_data, dict):
        logging.error("Workflow input must be a JSON object (dictionary).")
        raise ValueError("Workflow input must be a JSON object (dictionary).")

    logging.debug("Workflow input parsed successfully.")
    return input_data

def get_workflow_versions(state: Dict, workflow_name: str) -> List[int]:
    """Retrieves a list of versions for a given workflow, sorted in descending order."""
    workflows = state.get('workflows', {})
    if workflow_name in workflows:
        return sorted(workflows[workflow_name].keys(), reverse=True)
    return []

def record_workflow_execution(state: Dict, workflow_name: str, version: int, workflow_input: Dict, result: Union[Dict, str], status: str, error: Optional[str] = None) -> str:
    """Records the execution of a workflow, including input, output, status, and errors."""
    import uuid
    execution_id = str(uuid.uuid4())
    execution_record = {
        "workflow_name": workflow_name,
        "version": version,
        "input": workflow_input,
        "result": result,
        "status": status,
        "error": error,
        "timestamp": str(datetime.datetime.now())
    }

    if 'workflow_history' not in state:
        state['workflow_history'] = {}
    state['workflow_history'][execution_id] = execution_record
    return execution_id


def get_workflow_history(state: Dict, workflow_execution_id: str) -> Optional[Dict]:
    """Retrieves the execution history for a given workflow execution ID."""
    if 'workflow_history' not in state:
        return None
    return state['workflow_history'].get(workflow_execution_id)

import datetime

async def orchestrate_workflows(**kwargs: Dict[str, Any]) -> Dict[str, Any]:
    """Execute workflow orchestration actions"""
    try:
        action = kwargs.get('action')

        if not action:
            logging.error("Action parameter is required.")
            return {"error": "action parameter is required", "status": "failed"}

        action = action.lower() # Normalize the action to lowercase for case-insensitive matching

        # Input validation
        workflow_name = kwargs.get('workflow_name')
        version = kwargs.get('version')
        step_name = kwargs.get('step_name') # For step-specific error handling.
        force_delete = kwargs.get('force', False)
        workflow_execution_id = kwargs.get('workflow_execution_id')


        if action in ['execute_workflow', 'delete_workflow', 'get_workflow', 'update_workflow'] and not workflow_name:
            logging.error("workflow_name required for execute_workflow, delete_workflow, get_workflow and update_workflow.")
            return {"error": "workflow_name required for execute_workflow, delete_workflow, get_workflow and update_workflow", "status": "failed"}

        if action in ['create_workflow', 'update_workflow']:
            if not workflow_name:
                logging.error("workflow_name is required for create_workflow and update_workflow.")
                return {"error": "workflow_name is required for create_workflow and update_workflow", "status": "failed"}
            workflow_definition = kwargs.get('workflow_definition')
            if not workflow_definition:
                logging.error("workflow_definition is required for create_workflow and update_workflow.")
                return {"error": "workflow_definition is required for create_workflow and update_workflow", "status": "failed"}

        if action == 'get_workflow_history' and not workflow_execution_id:
            logging.error("workflow_execution_id is required for get_workflow_history.")
            return {"error": "workflow_execution_id is required for get_workflow_history", "status": "failed"}

        # Initialize orchestrator and state manager
        orchestrator = WorkflowOrchestrator()
        state_manager = StateManager()

        # Load state once and reuse for efficiency
        state = state_manager.load_state()

        if action == 'list_workflows':
            workflows = state.get('workflows', {})
            workflow_list = []
            for name, versions in workflows.items():
                workflow_list.append({"name": name, "versions": list(versions.keys())})

            return {
                "workflows": workflow_list,
                "count": len(workflows),
                "status": "success"
            }

        elif action == 'execute_workflow':
            workflow_input_str = kwargs.get('workflow_input')
            try:
                workflow_input = parse_workflow_input(workflow_input_str)
            except ValueError as e:
                logging.error(f"Error parsing workflow input: {e}")
                return {"error": str(e), "status": "failed"}


            # Load workflow definition, first attempt from cache.
            workflow_def = get_workflow_from_cache(workflow_name, version, state)

            if not workflow_def:
                workflows = state.get('workflows', {})
                if workflow_name not in workflows:
                    logging.error(f"Workflow '{workflow_name}' not found.")
                    return {"error": f"Workflow '{workflow_name}' not found", "status": "failed"}

                if version is not None:
                    if version not in workflows[workflow_name]:
                        logging.error(f"Workflow '{workflow_name}' version '{version}' not found.")
                        return {"error": f"Workflow '{workflow_name}' version '{version}' not found", "status": "failed"}
                    workflow_def = workflows[workflow_name][version]
                else: # Get latest version
                    versions = get_workflow_versions(state, workflow_name)
                    if not versions:
                        logging.error(f"No versions found for workflow '{workflow_name}'.")
                        return {"error": f"No versions found for workflow '{workflow_name}'", "status": "failed"}
                    workflow_def = workflows[workflow_name][versions[0]]

                # Add to cache if not already cached. Using a dummy state here.
                get_workflow_from_cache.cache_info()
                get_workflow_from_cache(workflow_name, version, {})

            execution_id = None # Initialize execution_id
            try:
                # Implement retry logic here if necessary for robust execution
                result = await orchestrator.execute_workflow(workflow_def, workflow_input)
                execution_id = record_workflow_execution(state, workflow_name, version or get_workflow_versions(state, workflow_name)[0], workflow_input, result, "success")

                state_manager.save_state(state) # Persist execution history
                return {"result": result, "workflow_name": workflow_name, "version": version, "status": "success", "workflow_execution_id": execution_id}
            except Exception as e:
                # Log the error for debugging and potential recovery
                logging.exception(f"Error executing workflow '{workflow_name}' version '{version}'.")

                error_message = f"Workflow execution failed: {str(e)}"
                if step_name: #Provide more targeted error reporting.
                    error_message = f"Workflow execution failed at step '{step_name}': {str(e)}"

                # Implement specific error handling or retry logic if possible based on the error type
                execution_id = record_workflow_execution(state, workflow_name, version or get_workflow_versions(state, workflow_name)[0], workflow_input, error_message, "failed", error=str(e))
                state_manager.save_state(state) #Persist execution history
                return {"error": error_message, "workflow_name": workflow_name, "version": version, "status": "failed",  "workflow_execution_id": execution_id}

        elif action in ['create_workflow', 'update_workflow']:
            workflow_definition = kwargs.get('workflow_definition')

            try:
                workflow_def = validate_workflow_definition(workflow_definition)
            except ValueError as e:
                logging.error(f"Invalid workflow definition: {e}")
                return {"error": str(e), "status": "failed"}

            # Save workflow to state
            if 'workflows' not in state:
                state['workflows'] = {}
            if workflow_name not in state['workflows']:
                state['workflows'][workflow_name] = {}

            if action == 'create_workflow':
                # Find the next available version number
                versions = get_workflow_versions(state, workflow_name)
                next_version = (versions[0] + 1) if versions else 1
                if version and version != next_version:
                    logging.error(f"Version must be {next_version} or unspecified during creation.")
                    return {"error": f"Version must be {next_version} or unspecified during creation", "status": "failed"}

                state['workflows'][workflow_name][next_version] = workflow_def
                used_version = next_version

            elif action == 'update_workflow':

                if version is None:
                    logging.error("Version is required for update_workflow.")
                    return {"error": "Version is required for update_workflow", "status": "failed"}

                if version not in state['workflows'][workflow_name]:
                    logging.error(f"Workflow '{workflow_name}' version '{version}' not found. Cannot update.")
                    return {"error": f"Workflow '{workflow_name}' version '{version}' not found. Cannot update.", "status": "failed"}

                state['workflows'][workflow_name][version] = workflow_def
                used_version = version


            state_manager.save_state(state)

            # Clear cache to keep it up to date. Using a dummy state here.
            clear_workflow_cache(workflow_name, used_version)

            return {
                "success": True,
                "message": f"Workflow '{workflow_name}' version '{used_version}' {action.split('_')[0]}d successfully",
                "workflow_name": workflow_name,
                "version": used_version,
                "status": "success"
            }

        elif action == 'delete_workflow':
            workflows = state.get('workflows', {})
            if workflow_name not in workflows:
                logging.error(f"Workflow '{workflow_name}' not found.")
                return {"error": f"Workflow '{workflow_name}' not found", "status": "failed"}

            if version is not None:
                if version not in workflows[workflow_name]:
                    logging.error(f"Workflow '{workflow_name}' version '{version}' not found.")
                    return {"error": f"Workflow '{workflow_name}' version '{version}' not found", "status": "failed"}

                del workflows[workflow_name][version]
                if not workflows[workflow_name]:
                    del workflows[workflow_name]  # Remove the whole workflow if no versions left
            else:
                if not force_delete:
                     logging.warning("Deleting all versions of workflow. Use force=True to bypass confirmation.")
                     return {"error": "Deleting all versions of workflow requires force=True", "status": "failed"}

                del workflows[workflow_name]  # Delete ALL versions

            state['workflows'] = workflows  # update workflows with deleted entry
            state_manager.save_state(state)

            # Clear cache to keep it up to date. Using a dummy state here.
            clear_workflow_cache(workflow_name, version)

            message = f"Workflow '{workflow_name}'"
            if version is not None:
                message += f" version '{version}'"
            message += " deleted successfully"

            return {
                "success": True,
                "message": message,
                "workflow_name": workflow_name,
                "version": version,
                "status": "success"
            }

        elif action == 'get_workflow':
            # Load workflow definition, first attempt from cache
            workflow_def = get_workflow_from_cache(workflow_name, version, state)

            if not workflow_def:
                workflows = state.get('workflows', {})
                if workflow_name not in workflows:
                    logging.error(f"Workflow '{workflow_name}' not found.")
                    return {"error": f"Workflow '{workflow_name}' not found", "status": "failed"}

                if version is not None:
                    if version not in workflows[workflow_name]:
                        logging.error(f"Workflow '{workflow_name}' version '{version}' not found.")
                        return {"error": f"Workflow '{workflow_name}' version '{version}' not found", "status": "failed"}
                    workflow_def = workflows[workflow_name][version]
                else: # Get latest version
                    versions = get_workflow_versions(state, workflow_name)
                    if not versions:
                        logging.error(f"No versions found for workflow '{workflow_name}'.")
                        return {"error": f"No versions found for workflow '{workflow_name}'", "status": "failed"}
                    workflow_def = workflows[workflow_name][versions[0]]

                # Add to cache if not already cached. Using a dummy state here.
                get_workflow_from_cache.cache_info()
                get_workflow_from_cache(workflow_name, version, {})

            return {
                "workflow": workflow_def,
                "workflow_name": workflow_name,
                "version": version,
                "status": "success"
            }

        elif action == 'validate_workflow':
            workflow_definition = kwargs.get('workflow_definition')
            if not workflow_definition:
                logging.error("workflow_definition is required for validate_workflow.")
                return {"error": "workflow_definition is required for validate_workflow", "status": "failed"}

            try:
                workflow_def = validate_workflow_definition(workflow_definition)
                return {"message": "Workflow definition is valid", "status": "success"}
            except ValueError as e:
                logging.error(f"Invalid workflow definition: {e}")
                return {"error": str(e), "status": "failed"}

        elif action == 'get_workflow_versions':
            if not workflow_name:
                logging.error("workflow_name is required for get_workflow_versions.")
                return {"error": "workflow_name is required for get_workflow_versions", "status": "failed"}
            versions = get_workflow_versions(state, workflow_name)
            return {"workflow_name": workflow_name, "versions": versions, "status": "success"}

        elif action == 'clear_cache':
            if not workflow_name:
              logging.error("workflow_name is required for clear_cache.")
              return {"error": "workflow_name is required for clear_cache", "status": "failed"}
            clear_workflow_cache(workflow_name, version)
            return {"message": f"Cache cleared for workflow '{workflow_name}' version '{version if version is not None else 'all'}'", "status": "success"}

        elif action == 'get_workflow_history':
            history = get_workflow_history(state, workflow_execution_id)
            if history:
                return {"workflow_execution_id": workflow_execution_id, "history": history, "status": "success"}
            else:
                return {"error": f"No history found for workflow execution ID '{workflow_execution_id}'", "status": "failed"}



        else:
            logging.error(f"Unknown action: {action}")
            return {"error": f"Unknown action: {action}", "status": "failed"}

    except Exception as e:
        logging.exception("Unexpected error during workflow orchestration.") # Use logging.exception to capture the stack trace
        return {"error": f"Execution failed: {str(e)}", "status": "failed"}