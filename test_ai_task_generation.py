#!/usr/bin/env python3
"""Test script to verify AI-powered task generation is working correctly."""

import asyncio
import os
import sys
from datetime import datetime

# Add scripts directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'scripts'))

from scripts.task_manager import TaskManager, TaskType
from scripts.enhanced_work_generator import EnhancedWorkGenerator
from scripts.state_manager import StateManager
from scripts.ai_brain import IntelligentAIBrain


async def test_task_manager_generation():
    """Test TaskManager AI content generation."""
    print("\n=== Testing TaskManager AI Content Generation ===")
    
    # Initialize task manager
    task_manager = TaskManager()
    
    # Test if AI content generator was initialized
    if task_manager.ai_content_generator:
        print("✓ AI content generator initialized in TaskManager")
    else:
        print("✗ AI content generator NOT initialized in TaskManager")
        return
    
    # Test generation for different task types
    print("\nGenerating sample tasks...")
    
    # Generate a few tasks
    tasks = task_manager.generate_tasks(max_tasks=3)
    
    for i, task in enumerate(tasks):
        print(f"\n--- Task {i+1} ---")
        print(f"Type: {task.get('type')}")
        print(f"Title: {task.get('title')}")
        print(f"Repository: {task.get('repository', 'None')}")
        print(f"Description preview: {task.get('description', '')[:200]}...")
        
        # Check if it looks like a template
        title = task.get('title', '').lower()
        template_indicators = [
            'add real-time notifications system',
            'create advanced search functionality',
            'implement data export feature',
            'add multi-language support',
            'create api rate limiting'
        ]
        
        is_template = any(indicator in title for indicator in template_indicators)
        if is_template:
            print("⚠️  WARNING: This appears to be a template-based task!")
        else:
            print("✓ This appears to be AI-generated content")


async def test_enhanced_work_generator():
    """Test EnhancedWorkGenerator AI content generation."""
    print("\n\n=== Testing EnhancedWorkGenerator AI Content Generation ===")
    
    # Initialize components
    state_manager = StateManager()
    system_state = state_manager.load_state()
    ai_brain = IntelligentAIBrain(system_state, {})
    
    # Initialize work generator
    work_generator = EnhancedWorkGenerator(ai_brain=ai_brain, system_state=system_state)
    
    # Test if AI content generator was initialized
    if work_generator.ai_content_generator:
        print("✓ AI content generator initialized in EnhancedWorkGenerator")
    else:
        print("✗ AI content generator NOT initialized in EnhancedWorkGenerator")
        return
    
    # Generate some work items
    print("\nGenerating work items...")
    work_items = await work_generator.generate_work_batch(target_count=3)
    
    for i, work_item in enumerate(work_items):
        print(f"\n--- Work Item {i+1} ---")
        print(f"Type: {work_item.task_type}")
        print(f"Title: {work_item.title}")
        print(f"Repository: {work_item.repository}")
        print(f"Description preview: {work_item.description[:200]}...")
        print(f"AI Generated: {work_item.metadata.get('ai_generated', False)}")
        
        # Check if it looks like a template
        title_lower = work_item.title.lower()
        template_patterns = [
            'update {repo}',
            'create api documentation for',
            'add unit tests for',
            'optimize database queries in'
        ]
        
        is_template = any(pattern.replace('{repo}', '') in title_lower for pattern in template_patterns)
        if is_template:
            print("⚠️  WARNING: This appears to be a template-based work item!")
        else:
            print("✓ This appears to be AI-generated content")


async def main():
    """Run all tests."""
    print("Starting AI Task Generation Tests")
    print(f"Timestamp: {datetime.now()}")
    
    # Check environment
    print("\nEnvironment Check:")
    print(f"ANTHROPIC_API_KEY exists: {bool(os.getenv('ANTHROPIC_API_KEY'))}")
    print(f"Working directory: {os.getcwd()}")
    
    try:
        # Run tests
        await test_task_manager_generation()
        await test_enhanced_work_generator()
        
        print("\n\n=== Test Summary ===")
        print("If you see AI-generated content above, the fix is working!")
        print("If you see template warnings, the system is falling back to templates.")
        
    except Exception as e:
        print(f"\nError during testing: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())