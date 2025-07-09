"""
Storage Factory

Factory classes for creating appropriate storage implementations
based on the configured storage mode (local or GitHub).
"""

import os
import logging
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from scripts.local_github_simulator import LocalGitHubSimulator

from scripts.storage_mode import StorageMode


class IssueCreatorFactory:
    """Factory for creating issue creator instances."""
    
    @staticmethod
    def create(storage_mode: str = None, enable_claude_execution: bool = None):
        """Create appropriate issue creator based on storage mode.
        
        Args:
            storage_mode: Override storage mode (optional)
            enable_claude_execution: Enable Claude Code CLI execution (local mode only)
            
        Returns:
            IssueCreator instance (LocalIssueCreator or GitHubIssueCreator)
        """
        mode = StorageMode.LOCAL  # ALWAYS use local mode
        
        logger = logging.getLogger(__name__)
        logger.info(f"Creating issue creator for mode: {mode}")
        
        if mode == StorageMode.LOCAL:
            # Import here to avoid circular import
            from scripts.local_issue_creator import LocalIssueCreator
            # Default to True if not specified
            if enable_claude_execution is None:
                enable_claude_execution = os.getenv('ENABLE_CLAUDE_EXECUTION', 'true').lower() == 'true'
            logger.info(f"Local issue creator with Claude execution: {enable_claude_execution}")
            return LocalIssueCreator(enable_claude_execution=enable_claude_execution)
        else:
            # Import here to avoid circular import
            from scripts.github_issue_creator import GitHubIssueCreator
            return GitHubIssueCreator(storage_mode=mode)


class ProjectCreatorFactory:
    """Factory for creating project creator instances."""
    
    @staticmethod
    def create(github_token: str = None, ai_brain=None, storage_mode: str = None):
        """Create appropriate project creator based on storage mode.
        
        Args:
            github_token: GitHub token (required for GitHub mode)
            ai_brain: AI brain instance
            storage_mode: Override storage mode (optional)
            
        Returns:
            ProjectCreator instance
        """
        mode = StorageMode.LOCAL  # ALWAYS use local mode
        
        logger = logging.getLogger(__name__)
        logger.info(f"Creating project creator for mode: {mode}")
        
        # Import here to avoid circular import
        from scripts.project_creator import ProjectCreator
        
        # ProjectCreator now handles both local and GitHub modes internally
        if mode == StorageMode.LOCAL:
            # For local mode, github_token is optional
            return ProjectCreator(github_token=None, ai_brain=ai_brain)
        else:
            # For GitHub mode, try to get token if not provided
            if not github_token:
                github_token = os.getenv('GITHUB_TOKEN') or os.getenv('CLAUDE_PAT')
            return ProjectCreator(github_token=github_token, ai_brain=ai_brain)


class TaskManagerFactory:
    """Factory for creating task manager instances."""
    
    @staticmethod
    def create(github_token: str = None, repository: str = None, storage_mode: str = None):
        """Create appropriate task manager based on storage mode.
        
        Args:
            github_token: GitHub token (required for GitHub mode)
            repository: Target repository
            storage_mode: Override storage mode (optional)
            
        Returns:
            TaskManager instance
        """
        mode = StorageMode.LOCAL  # ALWAYS use local mode
        
        logger = logging.getLogger(__name__)
        logger.info(f"Creating task manager for mode: {mode}")
        
        # Import here to avoid circular import
        from scripts.task_manager import TaskManager
        
        if mode == StorageMode.LOCAL:
            # For local mode, we can use the regular TaskManager but
            # it will work with local issue creator
            return TaskManager(github_token=None, repository=repository)
        else:
            if not github_token:
                github_token = os.getenv('GITHUB_TOKEN') or os.getenv('CLAUDE_PAT')
            return TaskManager(github_token=github_token, repository=repository)


class StorageFactory:
    """Main factory for all storage-related instances."""
    
    def __init__(self, storage_mode: str = None):
        """Initialize with storage mode.
        
        Args:
            storage_mode: Storage mode (local or github)
        """
        self.storage_mode = storage_mode or os.getenv('STORAGE_MODE', StorageMode.GITHUB)
        self.logger = logging.getLogger(__name__)
        
        # Normalize storage mode to lowercase
        self.storage_mode = self.storage_mode.lower()
        
        self.logger.info(f"StorageFactory initialized with mode: {self.storage_mode}")
        
        # Validate storage mode
        if self.storage_mode not in [StorageMode.LOCAL, StorageMode.GITHUB]:
            self.logger.warning(f"Invalid storage mode: {self.storage_mode}, defaulting to GitHub")
            self.storage_mode = StorageMode.GITHUB
    
    def create_issue_creator(self):
        """Create issue creator for current storage mode."""
        return IssueCreatorFactory.create(self.storage_mode)
    
    def create_project_creator(self, github_token: str = None, ai_brain=None):
        """Create project creator for current storage mode."""
        return ProjectCreatorFactory.create(github_token, ai_brain, self.storage_mode)
    
    def create_task_manager(self, github_token: str = None, repository: str = None):
        """Create task manager for current storage mode."""
        return TaskManagerFactory.create(github_token, repository, self.storage_mode)
    
    def get_simulator(self) -> Optional["LocalGitHubSimulator"]:
        """Get local GitHub simulator if in local mode.
        
        Returns:
            LocalGitHubSimulator instance or None
        """
        if self.storage_mode == StorageMode.LOCAL:
            # Import here to avoid circular import
            from scripts.local_github_simulator import LocalGitHubSimulator
            return LocalGitHubSimulator()
        return None
    
    def is_local_mode(self) -> bool:
        """Check if running in local mode.
        
        Returns:
            True if in local mode
        """
        return self.storage_mode == StorageMode.LOCAL
    
    def is_github_mode(self) -> bool:
        """Check if running in GitHub mode.
        
        Returns:
            True if in GitHub mode
        """
        return self.storage_mode == StorageMode.GITHUB