#!/usr/bin/env python3
"""
Test Workflow Orchestrator

Demonstrates the workflow orchestration capabilities.
"""

import asyncio
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from scripts.tool_calling_system import ToolCallingSystem


async def test_workflow_orchestrator():
    """Test the workflow orchestrator."""
    print("Testing Workflow Orchestrator")
    print("=" * 50)
    
    # Initialize tool system
    tool_system = ToolCallingSystem()
    
    # Test 1: List available workflows
    print("\n1. Listing workflows...")
    result = await tool_system.call_tool("orchestrate_workflows", action="list")
    print(f"Available examples: {result.get('available_examples', [])}")
    print(f"Total workflows: {result.get('total_workflows', 0)}")
    
    # Test 2: Create a simple workflow
    print("\n2. Creating a simple workflow...")
    simple_workflow = {
        "name": "System Check Workflow",
        "description": "Check system status and take action if needed",
        "steps": [
            {
                "id": "check_status",
                "name": "Check System Status",
                "type": "tool",
                "tool": "get_system_status",
                "params": {}
            },
            {
                "id": "log_status",
                "name": "Log Status",
                "type": "log",
                "message": "System status checked. Continuous AI running: {{step_check_status_result.continuous_ai_running}}",
                "level": "info"
            },
            {
                "id": "conditional_action",
                "name": "Conditional Action",
                "type": "conditional",
                "condition": {
                    "operator": "eq",
                    "left": "{{step_check_status_result.continuous_ai_running}}",
                    "right": False
                },
                "then": {
                    "type": "log",
                    "message": "System is not running. Consider starting it.",
                    "level": "warning"
                },
                "else": {
                    "type": "log",
                    "message": "System is running normally.",
                    "level": "info"
                }
            }
        ]
    }
    
    create_result = await tool_system.call_tool(
        "orchestrate_workflows",
        action="create",
        workflow_config=simple_workflow
    )
    
    print(f"Create result: {create_result}")
    
    if create_result.get('success'):
        workflow_id = create_result['workflow_id']
        print(f"✓ Created workflow: {workflow_id}")
        
        # Test 3: Execute the workflow
        print("\n3. Executing the workflow...")
        exec_result = await tool_system.call_tool(
            "orchestrate_workflows",
            action="execute",
            workflow_id=workflow_id
        )
        
        if exec_result['success']:
            print("✓ Workflow executed successfully")
            print(f"Final status: {exec_result['workflow_result']['workflow']['status']}")
        else:
            print(f"✗ Workflow execution failed: {exec_result.get('error')}")
            
        # Test 4: Check workflow status
        print("\n4. Checking workflow status...")
        status_result = await tool_system.call_tool(
            "orchestrate_workflows",
            action="status",
            workflow_id=workflow_id
        )
        
        if status_result['success']:
            workflow = status_result['workflow']
            print(f"Workflow: {workflow['name']}")
            print(f"Status: {workflow['status']}")
            print(f"Duration: {workflow.get('duration_ms', 0):.2f}ms")
            
            # Show step results
            print("\nStep Results:")
            for step in workflow['steps']:
                print(f"  - {step['name']}: {step['status']}")
                if step.get('error'):
                    print(f"    Error: {step['error']}")
    
    # Test 5: Create and execute a more complex workflow
    print("\n5. Creating a complex workflow with parallel execution...")
    complex_workflow = {
        "name": "Parallel Repository Analysis",
        "description": "Analyze multiple aspects of the system in parallel",
        "steps": [
            {
                "id": "parallel_checks",
                "name": "Parallel System Checks",
                "type": "parallel",
                "steps": [
                    {
                        "type": "tool",
                        "tool": "count_repositories",
                        "params": {}
                    },
                    {
                        "type": "tool",
                        "tool": "count_tasks",
                        "params": {}
                    },
                    {
                        "type": "tool",
                        "tool": "get_system_status",
                        "params": {}
                    }
                ]
            },
            {
                "id": "summarize",
                "name": "Summarize Results",
                "type": "transform",
                "transform": "extract",
                "input": "{{step_parallel_checks_result}}",
                "output": "summary",
                "fields": ["parallel_0", "parallel_1", "parallel_2"]
            },
            {
                "id": "log_summary",
                "name": "Log Summary",
                "type": "log",
                "message": "System check complete. Repositories: {{step_parallel_checks_result.parallel_0.count}}, Tasks: {{step_parallel_checks_result.parallel_1.count}}",
                "level": "info"
            }
        ]
    }
    
    complex_result = await tool_system.call_tool(
        "orchestrate_workflows",
        action="create",
        workflow_config=complex_workflow
    )
    
    if complex_result['success']:
        complex_workflow_id = complex_result['workflow_id']
        print(f"✓ Created complex workflow: {complex_workflow_id}")
        
        # Execute it
        print("\n6. Executing complex workflow...")
        exec_result = await tool_system.call_tool(
            "orchestrate_workflows",
            action="execute",
            workflow_id=complex_workflow_id
        )
        
        if exec_result['success']:
            print("✓ Complex workflow executed successfully")
        else:
            print(f"✗ Complex workflow failed: {exec_result.get('error')}")
    
    # Test 7: Use an example workflow
    print("\n7. Using example workflow 'analyze_and_fix'...")
    example_result = await tool_system.call_tool(
        "orchestrate_workflows",
        action="create",
        use_example="analyze_and_fix"
    )
    
    if example_result['success']:
        print(f"✓ Created example workflow: {example_result['workflow']['name']}")
        print(f"  Description: {example_result['workflow']['description']}")
        print(f"  Steps: {len(example_result['workflow']['steps'])}")


async def test_workflow_with_loops():
    """Test workflow with loop functionality."""
    print("\n\nTesting Workflow with Loops")
    print("=" * 50)
    
    tool_system = ToolCallingSystem()
    
    # Create a workflow that processes a list of items
    loop_workflow = {
        "name": "Process Multiple Items",
        "description": "Process a list of items using loops",
        "steps": [
            {
                "id": "prepare_data",
                "name": "Prepare Data",
                "type": "transform",
                "transform": "extract",
                "input": {
                    "items": ["item1", "item2", "item3"],
                    "prefix": "processed_"
                },
                "output": "data"
            },
            {
                "id": "process_items",
                "name": "Process Each Item",
                "type": "loop",
                "items": "{{data.items}}",
                "item_var": "current_item",
                "body": {
                    "type": "log",
                    "message": "Processing item {{loop_index}}: {{current_item}}",
                    "level": "info"
                }
            }
        ]
    }
    
    result = await tool_system.call_tool(
        "orchestrate_workflows",
        action="create",
        workflow_config=loop_workflow
    )
    
    if result['success']:
        workflow_id = result['workflow_id']
        print(f"✓ Created loop workflow: {workflow_id}")
        
        # Execute it
        exec_result = await tool_system.call_tool(
            "orchestrate_workflows",
            action="execute",
            workflow_id=workflow_id
        )
        
        if exec_result['success']:
            print("✓ Loop workflow executed successfully")
            print(f"  Processed {exec_result['workflow_result']['workflow']['steps'][1]['result']['data']['iterations']} items")


if __name__ == "__main__":
    asyncio.run(test_workflow_orchestrator())
    asyncio.run(test_workflow_with_loops())