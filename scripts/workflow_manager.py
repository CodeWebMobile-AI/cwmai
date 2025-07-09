#!/usr/bin/env python3
"""
Workflow Manager CLI
====================

Command-line interface for managing and executing system workflows.
"""

import asyncio
import argparse
import json
from typing import Dict, Any, Optional
from pathlib import Path

from scripts.workflow_orchestrator import WorkflowOrchestrator
from scripts.system_workflows import get_workflow_definition, list_all_workflows
from scripts.conversational_ai_assistant import ConversationalAIAssistant
from scripts.logger import get_logger

logger = get_logger("WorkflowManager")


class WorkflowManager:
    """Manages system workflows."""
    
    def __init__(self):
        self.orchestrator = WorkflowOrchestrator()
        self.ai_assistant = ConversationalAIAssistant()
        
    async def initialize(self):
        """Initialize the workflow manager."""
        await self.orchestrator.initialize()
        
    async def create_workflow(self, workflow_name: str, custom_params: Optional[Dict[str, Any]] = None) -> str:
        """Create a workflow from predefined definitions."""
        workflow_def = get_workflow_definition(workflow_name)
        
        if not workflow_def:
            raise ValueError(f"Workflow '{workflow_name}' not found")
            
        # Apply custom parameters if provided
        if custom_params:
            workflow_def['context'] = custom_params
            
        # Create the workflow
        workflow = await self.orchestrator.create_workflow(workflow_def)
        logger.info(f"Created workflow: {workflow.name} (ID: {workflow.id})")
        
        return workflow.id
        
    async def execute_workflow(self, workflow_id: str, input_data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Execute a workflow by ID."""
        logger.info(f"Executing workflow: {workflow_id}")
        result = await self.orchestrator.execute_workflow(workflow_id, input_data)
        
        if result['success']:
            logger.info(f"Workflow {workflow_id} completed successfully")
        else:
            logger.error(f"Workflow {workflow_id} failed: {result.get('error')}")
            
        return result
        
    async def create_and_execute(self, workflow_name: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Create and immediately execute a workflow."""
        workflow_id = await self.create_workflow(workflow_name, params)
        return await self.execute_workflow(workflow_id, params)
        
    def list_workflows(self) -> list:
        """List all available workflows."""
        return list_all_workflows()
        
    def get_workflow_status(self, workflow_id: str) -> Optional[Dict[str, Any]]:
        """Get the status of a specific workflow."""
        return self.orchestrator.get_workflow_status(workflow_id)
        
    async def ai_create_workflow(self, description: str) -> str:
        """Use AI to create a custom workflow from description."""
        prompt = f"""
        Create a workflow definition for the following task:
        {description}
        
        Return a valid workflow JSON structure with steps.
        Use available tools and workflow step types.
        """
        
        response = await self.ai_assistant.generate_response(prompt)
        
        try:
            workflow_def = json.loads(response)
            workflow = await self.orchestrator.create_workflow(workflow_def)
            return workflow.id
        except json.JSONDecodeError:
            logger.error("AI generated invalid workflow JSON")
            raise ValueError("Failed to generate valid workflow")


async def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(description="Workflow Manager CLI")
    subparsers = parser.add_subparsers(dest='command', help='Commands')
    
    # List command
    list_parser = subparsers.add_parser('list', help='List available workflows')
    
    # Create command
    create_parser = subparsers.add_parser('create', help='Create a workflow')
    create_parser.add_argument('workflow', help='Workflow name')
    create_parser.add_argument('--params', type=json.loads, help='JSON parameters')
    
    # Execute command
    exec_parser = subparsers.add_parser('execute', help='Execute a workflow')
    exec_parser.add_argument('workflow_id', help='Workflow ID')
    exec_parser.add_argument('--input', type=json.loads, help='JSON input data')
    
    # Run command (create and execute)
    run_parser = subparsers.add_parser('run', help='Create and execute a workflow')
    run_parser.add_argument('workflow', help='Workflow name')
    run_parser.add_argument('--params', type=json.loads, help='JSON parameters')
    
    # Status command
    status_parser = subparsers.add_parser('status', help='Get workflow status')
    status_parser.add_argument('workflow_id', help='Workflow ID')
    
    # AI create command
    ai_parser = subparsers.add_parser('ai-create', help='Create workflow using AI')
    ai_parser.add_argument('description', help='Workflow description')
    
    args = parser.parse_args()
    
    # Initialize manager
    manager = WorkflowManager()
    await manager.initialize()
    
    if args.command == 'list':
        workflows = manager.list_workflows()
        print("\nAvailable Workflows:")
        print("-" * 80)
        for wf in workflows:
            print(f"\n{wf['name']}:")
            print(f"  Name: {wf['display_name']}")
            print(f"  Description: {wf['description']}")
            
    elif args.command == 'create':
        workflow_id = await manager.create_workflow(args.workflow, args.params)
        print(f"Created workflow: {workflow_id}")
        
    elif args.command == 'execute':
        result = await manager.execute_workflow(args.workflow_id, args.input)
        print(f"\nWorkflow Result:")
        print(json.dumps(result, indent=2))
        
    elif args.command == 'run':
        result = await manager.create_and_execute(args.workflow, args.params)
        print(f"\nWorkflow Result:")
        print(json.dumps(result, indent=2))
        
    elif args.command == 'status':
        status = manager.get_workflow_status(args.workflow_id)
        if status:
            print(f"\nWorkflow Status:")
            print(json.dumps(status, indent=2))
        else:
            print(f"Workflow {args.workflow_id} not found")
            
    elif args.command == 'ai-create':
        workflow_id = await manager.ai_create_workflow(args.description)
        print(f"AI created workflow: {workflow_id}")
        
    else:
        parser.print_help()


if __name__ == "__main__":
    asyncio.run(main())