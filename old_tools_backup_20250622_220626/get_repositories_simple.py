"""
Simple Get Repositories Tool with Local Mode Support

This is a simplified version that automatically uses StorageFactory
to work with both GitHub and local repositories.
"""

from typing import List, Dict, Any, Optional
import os
import logging
from scripts.storage_factory import StorageFactory
from scripts.local_github_simulator import LocalGitHubSimulator
from scripts.github_issue_creator import GitHubIssueCreator
from scripts.state_manager import StateManager

logging.basicConfig(level=logging.INFO)

__description__ = "Get all repositories from GitHub or local storage"
__parameters__ = {
    "return_dict": {
        "type": "boolean",
        "description": "If True, returns list of dicts with repo info. If False, returns list of repo names.",
        "required": False,
        "default": True
    }
}

class GetRepositoriesSimple:
    """Simple repository getter that works with both GitHub and local mode."""
    
    def __init__(self):
        """Initialize with automatic storage detection."""
        self.logger = logging.getLogger(__name__)
        self.storage_factory = StorageFactory()
        self.storage_mode = self.storage_factory.storage_mode
        
        if self.storage_mode == "local":
            self.local_simulator = LocalGitHubSimulator()
            self.logger.info("Using local repository storage")
        else:
            self.state_manager = StateManager()
            self.logger.info("Using GitHub repository storage")
    
    def execute(self, return_dict: bool = True) -> List[Dict[str, Any]] | List[str]:
        """Get all repositories.
        
        Args:
            return_dict: If True, returns list of dicts with repo info. If False, returns list of repo names.
        
        Returns:
            List of repositories
        """
        if self.storage_mode == "local":
            return self._get_local_repositories(return_dict)
        else:
            return self._get_github_repositories(return_dict)
    
    def _get_local_repositories(self, return_dict: bool = True) -> List[Dict[str, Any]] | List[str]:
        """Get repositories from local storage."""
        try:
            repos = self.local_simulator.list_repositories()
            
            if return_dict:
                repo_data = []
                for repo in repos:
                    repo_info = {
                        'name': repo.name,
                        'full_name': repo.full_name,
                        'description': repo.description or '',
                        'owner': {
                            'login': repo.full_name.split('/')[0] if '/' in repo.full_name else 'local_user',
                            'type': 'User'
                        },
                        'created_at': repo.created_at.isoformat() if hasattr(repo, 'created_at') else None,
                        'updated_at': repo.updated_at.isoformat() if hasattr(repo, 'updated_at') else None,
                        'html_url': repo.html_url if hasattr(repo, 'html_url') else f"file://{repo.path}"
                    }
                    repo_data.append(repo_info)
                return repo_data
            else:
                return [repo.name for repo in repos]
                
        except Exception as e:
            self.logger.error(f"Error getting local repositories: {e}")
            return []
    
    def _get_github_repositories(self, return_dict: bool = True) -> List[Dict[str, Any]] | List[str]:
        """Get repositories from GitHub/state manager."""
        try:
            state = self.state_manager.load_state()
            repositories = []
            
            for repo_name, repo_data in state.get('projects', {}).items():
                if repo_data is None:
                    continue
                    
                if return_dict:
                    repositories.append({
                        'name': repo_name,
                        'full_name': f"{repo_data.get('owner', 'CodeWebMobile-AI')}/{repo_name}",
                        'description': repo_data.get('description', ''),
                        'owner': {
                            'login': repo_data.get('owner', 'CodeWebMobile-AI'),
                            'type': 'Organization'
                        },
                        'created_at': repo_data.get('created_at'),
                        'updated_at': repo_data.get('pushed_at'),
                        'html_url': repo_data.get('html_url', f"https://github.com/{repo_data.get('owner', 'CodeWebMobile-AI')}/{repo_name}")
                    })
                else:
                    repositories.append(repo_name)
                    
            return repositories
            
        except Exception as e:
            self.logger.error(f"Error getting GitHub repositories: {e}")
            return []


# Make it directly callable
def get_repositories_simple(return_dict: bool = True) -> List[Dict[str, Any]] | List[str]:
    """Get all repositories from current storage mode."""
    tool = GetRepositoriesSimple()
    return tool.execute(return_dict)