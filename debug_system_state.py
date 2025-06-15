#!/usr/bin/env python3
"""Debug system state to see what projects are loaded."""

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'scripts'))

from state_manager import StateManager
from redis_state_adapter import RedisEnabledStateManager
import asyncio

def debug_state():
    """Debug system state."""
    print("=== DEBUGGING SYSTEM STATE ===\n")
    
    # Try Redis-enabled state manager first
    try:
        state_manager = RedisEnabledStateManager()
        print("Using Redis-enabled state manager")
        
        # Initialize Redis
        asyncio.run(state_manager.initialize_redis())
        
        # Load state
        state = state_manager.load_state()
    except Exception as e:
        print(f"Redis state manager failed: {e}")
        print("Using file-based state manager")
        state_manager = StateManager()
        state = state_manager.load_state()
    
    # Check projects
    projects = state.get('projects', {})
    print(f"\nProjects in state: {len(projects)}")
    
    for project_name, project_data in projects.items():
        print(f"\n  Project: {project_name}")
        print(f"    Name: {project_data.get('name', 'Unknown')}")
        print(f"    Description: {project_data.get('description', 'No description')[:80]}...")
        print(f"    Health Score: {project_data.get('health_score', 0)}")
        print(f"    Created: {project_data.get('created_at', 'Unknown')}")
        
        # Check if this is one of our problematic repos
        if project_name in ['ai-creative-studio', 'moderncms-with-ai-powered-content-recommendations']:
            print(f"    >>> FOUND PROBLEMATIC REPOSITORY: {project_name}")
            print(f"    Full data: {project_data}")
    
    # Check repository discovery info
    repo_discovery = state.get('repository_discovery', {})
    print(f"\n\nRepository Discovery Info:")
    print(f"  Discovery Source: {repo_discovery.get('discovery_source', 'Unknown')}")
    print(f"  Last Discovery: {repo_discovery.get('last_discovery', 'Never')}")
    print(f"  Repositories Found: {repo_discovery.get('repositories_found', 0)}")
    print(f"  Organization: {repo_discovery.get('organization', 'Unknown')}")

if __name__ == "__main__":
    debug_state()