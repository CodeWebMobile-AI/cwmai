"""
Workflow Orchestrator

Enables defining and executing complex workflows involving multiple tools.
Supports sequential, parallel, and conditional execution of tool chains.
"""

import asyncio
import logging
import json
from typing import Dict, List, Any, Optional, Union, Callable
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path

from scripts.tool_calling_system import ToolCallingSystem
from scripts.state_manager import StateManager
from scripts.logger import get_logger


class WorkflowStepType(Enum):
    """Types of workflow steps."""
    TOOL = "tool"  # Execute a tool
    PARALLEL = "parallel"  # Execute multiple steps in parallel
    CONDITIONAL = "conditional"  # Execute based on condition
    LOOP = "loop"  # Loop over items
    TRANSFORM = "transform"  # Transform data
    WAIT = "wait"  # Wait/delay
    LOG = "log"  # Log a message


class WorkflowStatus(Enum):
    """Workflow execution status."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class WorkflowStep:
    """Represents a single step in a workflow."""
    
    def __init__(self, step_config: Dict[str, Any]):
        self.id = step_config.get('id', f"step_{datetime.now().timestamp()}")
        self.name = step_config.get('name', 'Unnamed Step')
        self.type = WorkflowStepType(step_config.get('type', 'tool'))
        self.config = step_config
        self.result = None
        self.error = None
        self.status = WorkflowStatus.PENDING
        self.started_at = None
        self.completed_at = None
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'id': self.id,
            'name': self.name,
            'type': self.type.value,
            'config': self.config,
            'result': self.result,
            'error': self.error,
            'status': self.status.value,
            'started_at': self.started_at.isoformat() if self.started_at else None,
            'completed_at': self.completed_at.isoformat() if self.completed_at else None,
            'duration_ms': (self.completed_at - self.started_at).total_seconds() * 1000 if self.completed_at and self.started_at else None
        }


class Workflow:
    """Represents a complete workflow."""
    
    def __init__(self, workflow_config: Dict[str, Any]):
        self.id = workflow_config.get('id', f"workflow_{datetime.now().timestamp()}")
        self.name = workflow_config.get('name', 'Unnamed Workflow')
        self.description = workflow_config.get('description', '')
        self.steps = [WorkflowStep(step) for step in workflow_config.get('steps', [])]
        self.context = workflow_config.get('context', {})
        self.status = WorkflowStatus.PENDING
        self.created_at = datetime.now(timezone.utc)
        self.started_at = None
        self.completed_at = None
        self.error = None
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'id': self.id,
            'name': self.name,
            'description': self.description,
            'steps': [step.to_dict() for step in self.steps],
            'context': self.context,
            'status': self.status.value,
            'created_at': self.created_at.isoformat(),
            'started_at': self.started_at.isoformat() if self.started_at else None,
            'completed_at': self.completed_at.isoformat() if self.completed_at else None,
            'duration_ms': (self.completed_at - self.started_at).total_seconds() * 1000 if self.completed_at and self.started_at else None,
            'error': self.error
        }


class WorkflowOrchestrator:
    """Orchestrates complex workflows involving multiple tools."""
    
    def __init__(self):
        self.logger = get_logger('WorkflowOrchestrator')
        self.tool_system = ToolCallingSystem()
        self.state_manager = StateManager()
        self.workflows = {}
        self.running_workflows = {}
        self._initialized = False
        
    async def initialize(self):
        """Initialize the orchestrator."""
        if self._initialized:
            return
            
        # Tool system doesn't have initialize method - it's ready to use
        self._load_saved_workflows()
        self._initialized = True
        
    def _load_saved_workflows(self):
        """Load saved workflows from state."""
        state = self.state_manager.load_state()
        saved_workflows = state.get('workflows', {})
        
        for workflow_id, workflow_data in saved_workflows.items():
            # Convert dict back to Workflow object
            if isinstance(workflow_data, dict):
                try:
                    workflow = Workflow(workflow_data)
                    workflow.id = workflow_id
                    workflow.status = WorkflowStatus(workflow_data.get('status', 'pending'))
                    self.workflows[workflow_id] = workflow
                except Exception as e:
                    self.logger.warning(f"Failed to load workflow {workflow_id}: {e}")
            else:
                self.workflows[workflow_id] = workflow_data
            
    def _save_workflow(self, workflow: Workflow):
        """Save workflow to state."""
        state = self.state_manager.load_state()
        if 'workflows' not in state:
            state['workflows'] = {}
            
        state['workflows'][workflow.id] = workflow.to_dict()
        self.state_manager.state = state
        self.state_manager.save_state()
        
    async def create_workflow(self, workflow_config: Dict[str, Any]) -> Workflow:
        """Create a new workflow from configuration."""
        workflow = Workflow(workflow_config)
        self.workflows[workflow.id] = workflow
        self._save_workflow(workflow)
        self.logger.info(f"Created workflow: {workflow.name} ({workflow.id})")
        return workflow
        
    async def execute_workflow(self, workflow_id: str, input_data: Dict[str, Any] = None) -> Dict[str, Any]:
        """Execute a workflow."""
        if not self._initialized:
            await self.initialize()
            
        workflow = self.workflows.get(workflow_id)
        if not workflow:
            return {
                'success': False,
                'error': f"Workflow {workflow_id} not found"
            }
            
        if workflow_id in self.running_workflows:
            return {
                'success': False,
                'error': f"Workflow {workflow_id} is already running"
            }
            
        self.running_workflows[workflow_id] = workflow
        
        try:
            # Start workflow
            workflow.status = WorkflowStatus.RUNNING
            workflow.started_at = datetime.now(timezone.utc)
            workflow.context.update(input_data or {})
            
            self.logger.info(f"Starting workflow: {workflow.name}")
            
            # Execute steps
            for step in workflow.steps:
                if workflow.status == WorkflowStatus.CANCELLED:
                    break
                    
                result = await self._execute_step(step, workflow.context)
                
                if not result['success']:
                    workflow.status = WorkflowStatus.FAILED
                    workflow.error = result.get('error', 'Unknown error')
                    break
                    
                # Update context with step results
                workflow.context[f"step_{step.id}_result"] = result.get('data')
                
            # Complete workflow
            if workflow.status == WorkflowStatus.RUNNING:
                workflow.status = WorkflowStatus.COMPLETED
                
            workflow.completed_at = datetime.now(timezone.utc)
            self._save_workflow(workflow)
            
            return {
                'success': workflow.status == WorkflowStatus.COMPLETED,
                'workflow': workflow.to_dict(),
                'final_context': workflow.context
            }
            
        except Exception as e:
            workflow.status = WorkflowStatus.FAILED
            workflow.error = str(e)
            workflow.completed_at = datetime.now(timezone.utc)
            self._save_workflow(workflow)
            
            self.logger.error(f"Workflow {workflow_id} failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'workflow': workflow.to_dict()
            }
            
        finally:
            del self.running_workflows[workflow_id]
            
    async def _execute_step(self, step: WorkflowStep, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a single workflow step."""
        step.status = WorkflowStatus.RUNNING
        step.started_at = datetime.now(timezone.utc)
        
        try:
            self.logger.info(f"Executing step: {step.name} (type: {step.type.value})")
            
            if step.type == WorkflowStepType.TOOL:
                result = await self._execute_tool_step(step, context)
            elif step.type == WorkflowStepType.PARALLEL:
                result = await self._execute_parallel_step(step, context)
            elif step.type == WorkflowStepType.CONDITIONAL:
                result = await self._execute_conditional_step(step, context)
            elif step.type == WorkflowStepType.LOOP:
                result = await self._execute_loop_step(step, context)
            elif step.type == WorkflowStepType.TRANSFORM:
                result = await self._execute_transform_step(step, context)
            elif step.type == WorkflowStepType.WAIT:
                result = await self._execute_wait_step(step, context)
            elif step.type == WorkflowStepType.LOG:
                result = await self._execute_log_step(step, context)
            else:
                result = {'success': False, 'error': f"Unknown step type: {step.type}"}
                
            step.result = result
            step.status = WorkflowStatus.COMPLETED if result['success'] else WorkflowStatus.FAILED
            step.error = result.get('error')
            
            return result
            
        except Exception as e:
            step.status = WorkflowStatus.FAILED
            step.error = str(e)
            return {'success': False, 'error': str(e)}
            
        finally:
            step.completed_at = datetime.now(timezone.utc)
            
    async def _execute_tool_step(self, step: WorkflowStep, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a tool step."""
        tool_name = step.config.get('tool')
        params = step.config.get('params', {})
        
        # Resolve parameters from context
        resolved_params = self._resolve_params(params, context)
        
        # Call the tool
        result = await self.tool_system.call_tool(tool_name, **resolved_params)
        
        return {
            'success': 'error' not in result,
            'data': result,
            'error': result.get('error')
        }
        
    async def _execute_parallel_step(self, step: WorkflowStep, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute multiple steps in parallel."""
        substeps = step.config.get('steps', [])
        
        # Create tasks for parallel execution
        tasks = []
        for substep_config in substeps:
            substep = WorkflowStep(substep_config)
            task = self._execute_step(substep, context)
            tasks.append(task)
            
        # Execute all tasks in parallel
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Check results
        all_success = True
        combined_data = {}
        errors = []
        
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                all_success = False
                errors.append(str(result))
            elif not result.get('success'):
                all_success = False
                errors.append(result.get('error', 'Unknown error'))
            else:
                combined_data[f"parallel_{i}"] = result.get('data')
                
        return {
            'success': all_success,
            'data': combined_data,
            'error': '; '.join(errors) if errors else None
        }
        
    async def _execute_conditional_step(self, step: WorkflowStep, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a conditional step."""
        condition = step.config.get('condition')
        then_step = step.config.get('then')
        else_step = step.config.get('else')
        
        # Evaluate condition
        if self._evaluate_condition(condition, context):
            if then_step:
                substep = WorkflowStep(then_step)
                return await self._execute_step(substep, context)
        else:
            if else_step:
                substep = WorkflowStep(else_step)
                return await self._execute_step(substep, context)
                
        return {'success': True, 'data': {'condition_met': self._evaluate_condition(condition, context)}}
        
    async def _execute_loop_step(self, step: WorkflowStep, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a loop step."""
        items_key = step.config.get('items')
        item_var = step.config.get('item_var', 'item')
        body_step = step.config.get('body')
        
        # Get items from context
        items = self._resolve_value(items_key, context)
        if not isinstance(items, list):
            return {'success': False, 'error': f"Loop items must be a list, got {type(items)}"}
            
        results = []
        for i, item in enumerate(items):
            # Add item to context
            loop_context = context.copy()
            loop_context[item_var] = item
            loop_context['loop_index'] = i
            
            # Execute body
            substep = WorkflowStep(body_step)
            result = await self._execute_step(substep, loop_context)
            results.append(result)
            
            if not result['success']:
                return {
                    'success': False,
                    'error': f"Loop failed at index {i}: {result.get('error')}",
                    'data': {'completed_iterations': i, 'results': results}
                }
                
        return {
            'success': True,
            'data': {'iterations': len(items), 'results': results}
        }
        
    async def _execute_transform_step(self, step: WorkflowStep, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a data transformation step."""
        transform_type = step.config.get('transform')
        input_key = step.config.get('input')
        output_key = step.config.get('output', 'transformed_data')
        
        # Get input data
        input_data = self._resolve_value(input_key, context)
        
        # Apply transformation
        try:
            if transform_type == 'extract':
                # Extract specific fields
                fields = step.config.get('fields', [])
                if isinstance(input_data, dict):
                    output_data = {field: input_data.get(field) for field in fields}
                elif isinstance(input_data, list) and all(isinstance(item, dict) for item in input_data):
                    output_data = [{field: item.get(field) for field in fields} for item in input_data]
                else:
                    output_data = input_data
                    
            elif transform_type == 'filter':
                # Filter items based on condition
                condition = step.config.get('filter_condition')
                if isinstance(input_data, list):
                    output_data = [item for item in input_data if self._evaluate_condition(condition, {'item': item})]
                else:
                    output_data = input_data
                    
            elif transform_type == 'map':
                # Map/transform each item
                mapping = step.config.get('mapping', {})
                if isinstance(input_data, list):
                    output_data = []
                    for item in input_data:
                        mapped_item = {}
                        for new_key, old_key in mapping.items():
                            mapped_item[new_key] = self._resolve_value(old_key, {'item': item})
                        output_data.append(mapped_item)
                elif isinstance(input_data, dict):
                    output_data = {}
                    for new_key, old_key in mapping.items():
                        output_data[new_key] = self._resolve_value(old_key, {'item': input_data})
                else:
                    output_data = input_data
                    
            elif transform_type == 'aggregate':
                # Aggregate data
                aggregation = step.config.get('aggregation', 'count')
                if isinstance(input_data, list):
                    if aggregation == 'count':
                        output_data = len(input_data)
                    elif aggregation == 'sum' and all(isinstance(x, (int, float)) for x in input_data):
                        output_data = sum(input_data)
                    elif aggregation == 'avg' and all(isinstance(x, (int, float)) for x in input_data):
                        output_data = sum(input_data) / len(input_data) if input_data else 0
                    else:
                        output_data = input_data
                else:
                    output_data = input_data
                    
            else:
                output_data = input_data
                
            # Store transformed data
            context[output_key] = output_data
            
            return {
                'success': True,
                'data': {output_key: output_data}
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': f"Transform failed: {e}"
            }
            
    async def _execute_wait_step(self, step: WorkflowStep, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a wait/delay step."""
        duration = step.config.get('duration', 1)
        await asyncio.sleep(duration)
        return {'success': True, 'data': {'waited': duration}}
        
    async def _execute_log_step(self, step: WorkflowStep, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a logging step."""
        message = step.config.get('message', '')
        level = step.config.get('level', 'info')
        
        # Resolve message from context
        resolved_message = self._resolve_value(message, context)
        
        # Log the message
        if level == 'debug':
            self.logger.debug(resolved_message)
        elif level == 'warning':
            self.logger.warning(resolved_message)
        elif level == 'error':
            self.logger.error(resolved_message)
        else:
            self.logger.info(resolved_message)
            
        return {'success': True, 'data': {'logged': resolved_message}}
        
    def _resolve_params(self, params: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Resolve parameters from context."""
        resolved = {}
        for key, value in params.items():
            resolved[key] = self._resolve_value(value, context)
        return resolved
        
    def _resolve_value(self, value: Any, context: Dict[str, Any]) -> Any:
        """Resolve a value from context."""
        if isinstance(value, str) and value.startswith('{{') and value.endswith('}}'):
            # Extract variable path
            path = value[2:-2].strip()
            
            # Navigate through context
            current = context
            for part in path.split('.'):
                if isinstance(current, dict) and part in current:
                    current = current[part]
                else:
                    return None
                    
            return current
        else:
            return value
            
    def _evaluate_condition(self, condition: Union[str, Dict[str, Any]], context: Dict[str, Any]) -> bool:
        """Evaluate a condition."""
        if isinstance(condition, str):
            # Simple string condition - resolve from context
            return bool(self._resolve_value(condition, context))
            
        elif isinstance(condition, dict):
            # Complex condition with operator
            operator = condition.get('operator', 'eq')
            left = self._resolve_value(condition.get('left'), context)
            right = self._resolve_value(condition.get('right'), context)
            
            if operator == 'eq':
                return left == right
            elif operator == 'ne':
                return left != right
            elif operator == 'gt':
                return left > right
            elif operator == 'gte':
                return left >= right
            elif operator == 'lt':
                return left < right
            elif operator == 'lte':
                return left <= right
            elif operator == 'in':
                return left in right
            elif operator == 'not_in':
                return left not in right
            elif operator == 'and':
                # Recursive evaluation for AND
                conditions = condition.get('conditions', [])
                return all(self._evaluate_condition(c, context) for c in conditions)
            elif operator == 'or':
                # Recursive evaluation for OR
                conditions = condition.get('conditions', [])
                return any(self._evaluate_condition(c, context) for c in conditions)
            else:
                return False
                
        else:
            return bool(condition)
            
    def get_workflow_status(self, workflow_id: str) -> Optional[Dict[str, Any]]:
        """Get the status of a workflow."""
        workflow = self.workflows.get(workflow_id)
        if workflow:
            return workflow.to_dict()
        return None
        
    def list_workflows(self) -> List[Dict[str, Any]]:
        """List all workflows."""
        workflows = []
        for w in self.workflows.values():
            if hasattr(w, 'to_dict'):
                workflows.append(w.to_dict())
            elif isinstance(w, dict):
                workflows.append(w)
        return workflows
        
    def cancel_workflow(self, workflow_id: str) -> bool:
        """Cancel a running workflow."""
        if workflow_id in self.running_workflows:
            workflow = self.running_workflows[workflow_id]
            workflow.status = WorkflowStatus.CANCELLED
            self.logger.info(f"Cancelled workflow: {workflow_id}")
            return True
        return False


# Example workflow configurations
EXAMPLE_WORKFLOWS = {
    "analyze_and_fix": {
        "name": "Analyze and Fix Repository",
        "description": "Analyze a repository for issues and create tasks to fix them",
        "steps": [
            {
                "id": "analyze",
                "name": "Analyze Repository",
                "type": "tool",
                "tool": "analyze_repository",
                "params": {
                    "repo_name": "{{repo_name}}",
                    "include_metrics": True
                }
            },
            {
                "id": "check_issues",
                "name": "Check for Issues",
                "type": "conditional",
                "condition": {
                    "operator": "gt",
                    "left": "{{step_analyze_result.issues_count}}",
                    "right": 0
                },
                "then": {
                    "id": "create_tasks",
                    "name": "Create Fix Tasks",
                    "type": "loop",
                    "items": "{{step_analyze_result.issues}}",
                    "item_var": "issue",
                    "body": {
                        "type": "tool",
                        "tool": "create_issue",
                        "params": {
                            "repo": "{{repo_name}}",
                            "title": "Fix: {{issue.title}}",
                            "body": "{{issue.description}}",
                            "labels": ["auto-generated", "fix-needed"]
                        }
                    }
                }
            },
            {
                "id": "monitor",
                "name": "Start Monitoring",
                "type": "tool",
                "tool": "monitor_worker_activity",
                "params": {}
            }
        ]
    },
    
    "parallel_analysis": {
        "name": "Parallel Multi-Repository Analysis",
        "description": "Analyze multiple repositories in parallel",
        "steps": [
            {
                "id": "parallel_analyze",
                "name": "Analyze All Repositories",
                "type": "parallel",
                "steps": [
                    {
                        "type": "tool",
                        "tool": "analyze_repository",
                        "params": {"repo_name": "{{repo1}}"}
                    },
                    {
                        "type": "tool",
                        "tool": "analyze_repository",
                        "params": {"repo_name": "{{repo2}}"}
                    },
                    {
                        "type": "tool",
                        "tool": "analyze_repository",
                        "params": {"repo_name": "{{repo3}}"}
                    }
                ]
            },
            {
                "id": "aggregate",
                "name": "Aggregate Results",
                "type": "transform",
                "transform": "aggregate",
                "input": "{{step_parallel_analyze_result}}",
                "output": "total_issues",
                "aggregation": "count"
            }
        ]
    }
}