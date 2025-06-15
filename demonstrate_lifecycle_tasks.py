#!/usr/bin/env python3
"""Demonstration of lifecycle-aware task generation in action."""

import asyncio
import os
import sys
from datetime import datetime, timedelta

# Add scripts directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'scripts'))

from scripts.project_lifecycle_analyzer import ProjectLifecycleAnalyzer


async def demonstrate():
    """Show how the system generates different tasks for different project stages."""
    
    print("\n=== Lifecycle-Aware Task Generation Demonstration ===\n")
    print("This system generates tasks based on where a project is in its lifecycle.")
    print("No hardcoded templates - everything is contextual!\n")
    
    # Initialize analyzer
    analyzer = ProjectLifecycleAnalyzer()
    
    # Example 1: Brand New Project
    print("="*60)
    print("Example 1: BRAND NEW PROJECT (3 days old)")
    print("="*60)
    
    new_project = {
        "repository": "my-new-app",
        "basic_info": {
            "created_at": (datetime.now() - timedelta(days=3)).isoformat() + "Z",
            "language": "JavaScript",
            "open_issues_count": 0
        },
        "health_metrics": {"health_score": 100, "recent_commits": 5},
        "code_analysis": {"documentation": [], "test_coverage": "unknown"},
        "issues_analysis": {"total_open": 0, "bug_count": 0, "feature_requests": 0},
        "recent_activity": {"active_contributors": 1}
    }
    
    result = await analyzer.analyze_project_stage(new_project)
    print(f"\nDetected Stage: {result['current_stage'].upper()}")
    print("\nAppropriate Tasks for This Stage:")
    for task in result['appropriate_task_types']:
        print(f"  • {task}")
    print("\nWhy these tasks? The project needs foundation work!")
    
    # Example 2: Active Development
    print("\n" + "="*60)
    print("Example 2: ACTIVE DEVELOPMENT PROJECT (3 months old)")
    print("="*60)
    
    active_project = {
        "repository": "growing-app",
        "basic_info": {
            "created_at": (datetime.now() - timedelta(days=90)).isoformat() + "Z",
            "language": "Python",
            "open_issues_count": 15
        },
        "health_metrics": {"health_score": 75, "recent_commits": 50},
        "code_analysis": {
            "documentation": ["README.md"],
            "test_coverage": "has_tests",
            "config_files": [".github/workflows/ci.yml"]
        },
        "issues_analysis": {
            "total_open": 15,
            "bug_count": 3,
            "feature_requests": 10
        },
        "recent_activity": {"active_contributors": 4}
    }
    
    result = await analyzer.analyze_project_stage(active_project)
    print(f"\nDetected Stage: {result['current_stage'].upper()}")
    print("\nAppropriate Tasks for This Stage:")
    for task in result['appropriate_task_types']:
        print(f"  • {task}")
    print("\nWhy these tasks? The project is growing rapidly and needs features!")
    
    # Example 3: Mature Project
    print("\n" + "="*60)
    print("Example 3: MATURE PROJECT (1 year old)")
    print("="*60)
    
    mature_project = {
        "repository": "stable-api",
        "basic_info": {
            "created_at": (datetime.now() - timedelta(days=365)).isoformat() + "Z",
            "language": "Java",
            "open_issues_count": 3
        },
        "health_metrics": {"health_score": 90, "recent_commits": 5},
        "code_analysis": {
            "documentation": ["README.md", "API.md", "CONTRIBUTING.md"],
            "test_coverage": "has_tests",
            "config_files": [".github/workflows/ci.yml", "docker-compose.yml"]
        },
        "issues_analysis": {
            "total_open": 3,
            "bug_count": 2,
            "feature_requests": 1
        },
        "recent_activity": {"active_contributors": 2}
    }
    
    result = await analyzer.analyze_project_stage(mature_project)
    print(f"\nDetected Stage: {result['current_stage'].upper()}")
    print("\nAppropriate Tasks for This Stage:")
    for task in result['appropriate_task_types']:
        print(f"  • {task}")
    print("\nWhy these tasks? The project is stable and needs maintenance!")
    
    # Show how transition planning works
    print("\n" + "="*60)
    print("TRANSITION PLANNING")
    print("="*60)
    
    print("\nEach project gets a plan to reach the next stage:")
    
    # Show transition for new project
    new_result = await analyzer.analyze_project_stage(new_project)
    transition = new_result['transition_plan']
    if transition.get('next_stage'):
        print(f"\n{new_project['repository']} → {transition['next_stage']}")
        print("Required tasks:")
        for task in transition.get('required_tasks', [])[:3]:
            print(f"  • {task}")
    
    # Summary
    print("\n" + "="*60)
    print("KEY BENEFITS")
    print("="*60)
    
    print("\n1. CONTEXTUAL TASKS: Every task matches the project's current needs")
    print("2. NATURAL PROGRESSION: Projects advance through stages systematically")
    print("3. NO WASTE: No premature optimization or inappropriate tasks")
    print("4. SMART PRIORITIZATION: Projects at critical stages get attention")
    print("5. CONTINUOUS IMPROVEMENT: The system learns what works")
    
    print("\n" + "="*60)
    print("LIFECYCLE STAGES")
    print("="*60)
    
    stages = [
        ("INCEPTION", "0-30 days", "Foundation & setup"),
        ("EARLY DEVELOPMENT", "15-90 days", "Core features"),
        ("ACTIVE DEVELOPMENT", "30-180 days", "Rapid growth"),
        ("GROWTH", "90-365 days", "Scaling & optimization"),
        ("MATURE", "180+ days", "Stability & maintenance"),
        ("MAINTENANCE", "365+ days", "Updates & fixes")
    ]
    
    print("\nThe system recognizes these project stages:\n")
    for stage, age, focus in stages:
        print(f"{stage:20} Age: {age:15} Focus: {focus}")


if __name__ == "__main__":
    asyncio.run(demonstrate())