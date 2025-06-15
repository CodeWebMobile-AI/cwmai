#!/usr/bin/env python3
"""Test the fixes we made to the AI system."""

import asyncio
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from scripts.task_manager import TaskManager, TaskType
from scripts.god_mode_controller import GodModeController, GodModeConfig, IntensityLevel
from scripts.ai_brain import AIBrain

async def test_feature_task_handling():
    """Test that feature tasks are handled correctly."""
    print("Testing feature task handling...")
    
    # Initialize AI brain
    ai_brain = AIBrain()
    
    # Initialize god mode controller
    config = GodModeConfig(
        intensity=IntensityLevel.MODERATE,
        enable_self_modification=False,  # Disable for testing
        enable_multi_repo=True,
        enable_predictive=False,
        enable_quantum=False,
        max_parallel_operations=1
    )
    
    controller = GodModeController(config, ai_brain)
    
    # Create a test feature task
    test_task = {
        'type': TaskType.FEATURE,
        'title': '[Test] Implement 2FA for admin accounts',
        'description': 'Add two-factor authentication for admin users',
        'priority': 'high',
        'source': 'test'
    }
    
    # Test the helper methods
    print(f"Feature needs new project: {controller._feature_needs_new_project(test_task)}")
    print(f"Generated project name: {controller._generate_project_name_from_feature(test_task)}")
    
    # Test swarm intelligence
    print("\nTesting real swarm intelligence...")
    swarm_result = await controller.swarm.process_task_swarm(test_task)
    
    if swarm_result and 'collective_review' in swarm_result:
        print(f"Swarm analysis completed in {swarm_result.get('duration_seconds', 0):.2f} seconds")
        print(f"Top suggestions: {swarm_result['collective_review'].get('top_suggestions', [])}")
    else:
        print("Swarm analysis failed or incomplete")
    
    print("\nTest completed!")

if __name__ == "__main__":
    asyncio.run(test_feature_task_handling())