"""
Repository Exclusion Module

Centralized configuration and utilities for excluding specific repositories
from all system flows. This ensures clean separation between the AI brain
(CWMAI) and the projects it manages.
"""

import json
import os
from typing import List, Set


class RepositoryExclusion:
    """Manages repository exclusion configuration and utilities."""
    
    # Base repositories to completely exclude from all flows
    _BASE_EXCLUDED_REPOS: Set[str] = {
        'codewebmobile-ai/cwmai',
        'cwmai',
        'cwmai.git',
        'codewebmobile-ai/.github',
        '.github'
    }
    
    # Dynamic exclusion list that includes base exclusions + deleted repos
    EXCLUDED_REPOS: Set[str] = _BASE_EXCLUDED_REPOS.copy()
    
    @classmethod
    def _load_deleted_repos(cls) -> None:
        """Load deleted repositories from persistent storage."""
        exclusion_file = os.path.join(os.path.dirname(__file__), "deleted_repos_exclusion.json")
        if os.path.exists(exclusion_file):
            try:
                with open(exclusion_file, 'r') as f:
                    deleted_repos = json.load(f)
                    for repo_info in deleted_repos:
                        if isinstance(repo_info, dict) and 'repository' in repo_info:
                            repo_name = repo_info['repository']
                            # Add both full name and short name
                            cls.EXCLUDED_REPOS.add(repo_name)
                            cls.EXCLUDED_REPOS.add(repo_name.lower())
                            if '/' in repo_name:
                                cls.EXCLUDED_REPOS.add(repo_name.split('/')[-1])
                                cls.EXCLUDED_REPOS.add(repo_name.split('/')[-1].lower())
                    print(f"Loaded {len(deleted_repos)} deleted repositories into exclusion list")
            except Exception as e:
                print(f"Warning: Failed to load deleted repos exclusion file: {e}")
    
    @classmethod
    def is_excluded_repo(cls, repo_identifier: str) -> bool:
        """Check if a repository should be excluded from all flows.
        
        Args:
            repo_identifier: Repository name, URL, or identifier
            
        Returns:
            True if repository should be excluded
        """
        if not repo_identifier:
            return False
        
        # Normalize the identifier
        normalized = cls._normalize_repo_identifier(repo_identifier)
        
        # Check against all exclusion patterns
        return normalized in cls.EXCLUDED_REPOS
    
    @classmethod
    def filter_excluded_repos(cls, repo_list: List[str]) -> List[str]:
        """Filter out excluded repositories from a list.
        
        Args:
            repo_list: List of repository identifiers
            
        Returns:
            Filtered list with excluded repositories removed
        """
        return [repo for repo in repo_list if not cls.is_excluded_repo(repo)]
    
    @classmethod
    def filter_excluded_repos_dict(cls, repo_dict: dict) -> dict:
        """Filter out excluded repositories from a dictionary.
        
        Args:
            repo_dict: Dictionary with repository identifiers as keys
            
        Returns:
            Filtered dictionary with excluded repositories removed
        """
        return {
            key: value for key, value in repo_dict.items()
            if not cls.is_excluded_repo(key)
        }
    
    @classmethod
    def should_process_repo(cls, repo_identifier: str) -> bool:
        """Check if a repository should be processed by the system.
        
        Args:
            repo_identifier: Repository name, URL, or identifier
            
        Returns:
            True if repository should be processed (not excluded)
        """
        return not cls.is_excluded_repo(repo_identifier)
    
    @classmethod
    def _normalize_repo_identifier(cls, identifier: str) -> str:
        """Normalize repository identifier for comparison.
        
        Args:
            identifier: Raw repository identifier
            
        Returns:
            Normalized identifier
        """
        if not identifier:
            return ""
        
        # Convert to lowercase for case-insensitive matching
        normalized = identifier.lower().strip()
        
        # Remove common prefixes/suffixes
        prefixes_to_remove = [
            'https://github.com/',
            'http://github.com/',
            'git@github.com:',
            'github.com/',
            'www.github.com/'
        ]
        
        for prefix in prefixes_to_remove:
            if normalized.startswith(prefix):
                normalized = normalized[len(prefix):]
                break
        
        # Remove .git suffix
        if normalized.endswith('.git'):
            normalized = normalized[:-4]
        
        # Remove trailing slashes
        normalized = normalized.rstrip('/')
        
        return normalized
    
    @classmethod
    def get_exclusion_reason(cls, repo_identifier: str) -> str:
        """Get human-readable explanation for why a repository is excluded.
        
        Args:
            repo_identifier: Repository identifier
            
        Returns:
            Explanation string
        """
        if cls.is_excluded_repo(repo_identifier):
            if 'cwmai' in repo_identifier.lower():
                return ("CWMAI repository is excluded because it's managed by the "
                       "self-improvement system, not the production flows")
            elif '.github' in repo_identifier.lower():
                return (".github repository is excluded because it contains organization-level "
                       "configuration and workflows, not application code")
        
        return "Repository is not excluded"
    
    @classmethod
    def validate_exclusion_config(cls) -> bool:
        """Validate the exclusion configuration.
        
        Returns:
            True if configuration is valid
        """
        # Ensure we have exclusions defined
        if not cls.EXCLUDED_REPOS:
            print("Warning: No repositories are excluded")
            return False
        
        # Check for duplicate entries (shouldn't happen with sets, but good to verify)
        normalized_repos = set()
        for repo in cls.EXCLUDED_REPOS:
            normalized = cls._normalize_repo_identifier(repo)
            if normalized in normalized_repos:
                print(f"Warning: Duplicate exclusion entry for {repo}")
            normalized_repos.add(normalized)
        
        print(f"Repository exclusion configuration valid: {len(cls.EXCLUDED_REPOS)} repositories excluded")
        return True
    
    @classmethod
    def get_excluded_repos_list(cls) -> List[str]:
        """Get list of all excluded repositories.
        
        Returns:
            List of excluded repository identifiers
        """
        return list(cls.EXCLUDED_REPOS)
    
    @classmethod
    def add_excluded_repo(cls, repo_identifier: str) -> bool:
        """Add a repository to the exclusion list.
        
        Args:
            repo_identifier: Repository to exclude
            
        Returns:
            True if added successfully
        """
        if not repo_identifier:
            return False
        
        normalized = cls._normalize_repo_identifier(repo_identifier)
        if normalized not in cls.EXCLUDED_REPOS:
            cls.EXCLUDED_REPOS.add(normalized)
            print(f"Added {repo_identifier} to exclusion list")
            return True
        
        print(f"Repository {repo_identifier} already excluded")
        return False
    
    @classmethod
    def remove_excluded_repo(cls, repo_identifier: str) -> bool:
        """Remove a repository from the exclusion list.
        
        Args:
            repo_identifier: Repository to remove from exclusion
            
        Returns:
            True if removed successfully
        """
        if not repo_identifier:
            return False
        
        normalized = cls._normalize_repo_identifier(repo_identifier)
        if normalized in cls.EXCLUDED_REPOS:
            cls.EXCLUDED_REPOS.remove(normalized)
            print(f"Removed {repo_identifier} from exclusion list")
            return True
        
        print(f"Repository {repo_identifier} was not excluded")
        return False


# Convenience functions for common operations
def is_excluded_repo(repo_identifier: str) -> bool:
    """Check if repository is excluded (convenience function)."""
    return RepositoryExclusion.is_excluded_repo(repo_identifier)


def filter_excluded_repos(repo_list: List[str]) -> List[str]:
    """Filter excluded repositories from list (convenience function)."""
    return RepositoryExclusion.filter_excluded_repos(repo_list)


def should_process_repo(repo_identifier: str) -> bool:
    """Check if repository should be processed (convenience function)."""
    return RepositoryExclusion.should_process_repo(repo_identifier)


def get_exclusion_reason(repo_identifier: str) -> str:
    """Get exclusion reason (convenience function)."""
    return RepositoryExclusion.get_exclusion_reason(repo_identifier)


# Initialize and validate configuration on import
RepositoryExclusion._load_deleted_repos()  # Load deleted repos first
RepositoryExclusion.validate_exclusion_config()