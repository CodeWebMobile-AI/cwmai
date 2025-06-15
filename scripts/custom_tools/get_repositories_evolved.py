"""
AI-Generated Tool: get_repositories_evolved
Description: Get repositories with optional filtering and sorting
Generated: 2025-06-15
Requirements: Get list of repositories with filtering by name/language
"""

from typing import List, Dict, Any

__description__ = "Get repositories with optional filtering and sorting"
__parameters__ = {
    "filter": {
        "type": "string",
        "description": "Optional filter string to match repository names or languages",
        "required": False
    },
    "limit": {
        "type": "integer",
        "description": "Maximum number of repositories to return",
        "required": False,
        "default": 10
    }
}
__examples__ = [
    {
        "command": "get_repositories_evolved", 
        "description": "Get all repositories"
    },
    {
        "command": "get_repositories_evolved",
        "parameters": {"filter": "python", "limit": 5},
        "description": "Get top 5 Python repositories"
    }
]


async def get_repositories_evolved(**kwargs) -> List[Dict[str, Any]]:
    """Get repositories with optional filtering."""
    from scripts.state_manager import StateManager
    
    # Extract parameters with defaults
    filter_str = kwargs.get('filter', None)
    limit = kwargs.get('limit', 10)
    
    state_manager = StateManager()
    
    # Force reload to get fresh state
    if hasattr(state_manager, "force_reload_state"):
        state = state_manager.force_reload_state()
    else:
        state = state_manager.load_state()
    
    repositories = []
    for repo_name, repo_data in state.get('projects', {}).items():
        if filter_str:
            # Apply filter
            if filter_str.lower() not in repo_name.lower() and filter_str.lower() not in repo_data.get('language', '').lower():
                continue
        
        repositories.append({
            'name': repo_name,
            'description': repo_data.get('description', ''),
            'language': repo_data.get('language', ''),
            'stars': repo_data.get('stars', 0),
            'issues': repo_data.get('open_issues_count', 0)
        })
    
    # Sort by stars and limit
    repositories.sort(key=lambda x: x['stars'], reverse=True)
    return repositories[:limit]