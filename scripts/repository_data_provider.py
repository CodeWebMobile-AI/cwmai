"""
Repository Data Provider Interface and Implementations

This module provides a clean abstraction for accessing repository data from
different sources (GitHub API, local filesystem, etc.) following the 
Dependency Inversion Principle.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
from dataclasses import dataclass
import os
from pathlib import Path


@dataclass
class RepositoryInfo:
    """Unified repository information structure."""
    name: str
    full_name: str
    description: Optional[str]
    language: Optional[str]
    created_at: datetime
    updated_at: datetime
    size: int
    default_branch: str
    topics: List[str]
    has_issues: bool
    open_issues_count: int
    has_wiki: bool
    has_pages: bool
    fork: bool
    private: bool
    html_url: str
    clone_url: str


@dataclass
class FileContent:
    """Unified file content structure."""
    path: str
    content: str
    encoding: str = 'utf-8'
    size: int = 0


@dataclass
class IssueInfo:
    """Unified issue information structure."""
    number: int
    title: str
    body: Optional[str]
    state: str
    labels: List[str]
    created_at: datetime
    updated_at: datetime
    author: str
    assignee: Optional[str]


@dataclass
class CommitInfo:
    """Unified commit information structure."""
    sha: str
    message: str
    author: str
    date: datetime
    files_changed: int


class RepositoryDataProvider(ABC):
    """Abstract base class for repository data providers."""
    
    @abstractmethod
    def get_repository(self, repo_name: str) -> RepositoryInfo:
        """Get basic repository information."""
        pass
    
    @abstractmethod
    def get_file_content(self, repo_name: str, file_path: str) -> Optional[FileContent]:
        """Get content of a specific file."""
        pass
    
    @abstractmethod
    def list_files(self, repo_name: str, path: str = "") -> List[str]:
        """List files in a directory."""
        pass
    
    @abstractmethod
    def get_issues(self, repo_name: str, state: str = "open", 
                   labels: Optional[List[str]] = None) -> List[IssueInfo]:
        """Get repository issues."""
        pass
    
    @abstractmethod
    def get_recent_commits(self, repo_name: str, limit: int = 30) -> List[CommitInfo]:
        """Get recent commits."""
        pass
    
    @abstractmethod
    def get_languages(self, repo_name: str) -> Dict[str, int]:
        """Get language statistics."""
        pass
    
    @abstractmethod
    def get_contributors(self, repo_name: str) -> List[Dict[str, Any]]:
        """Get repository contributors."""
        pass
    
    @abstractmethod
    def file_exists(self, repo_name: str, file_path: str) -> bool:
        """Check if a file exists in the repository."""
        pass


class GitHubDataProvider(RepositoryDataProvider):
    """GitHub API data provider implementation."""
    
    def __init__(self, github_client):
        """Initialize with GitHub client."""
        self.github = github_client
        self._repo_cache = {}
    
    def _get_repo_object(self, repo_name: str):
        """Get cached repository object."""
        if repo_name not in self._repo_cache:
            self._repo_cache[repo_name] = self.github.get_repo(repo_name)
        return self._repo_cache[repo_name]
    
    def get_repository(self, repo_name: str) -> RepositoryInfo:
        """Get basic repository information from GitHub."""
        repo = self._get_repo_object(repo_name)
        
        return RepositoryInfo(
            name=repo.name,
            full_name=repo.full_name,
            description=repo.description,
            language=repo.language,
            created_at=repo.created_at,
            updated_at=repo.updated_at,
            size=repo.size,
            default_branch=repo.default_branch,
            topics=list(repo.get_topics()),
            has_issues=repo.has_issues,
            open_issues_count=repo.open_issues_count,
            has_wiki=repo.has_wiki,
            has_pages=repo.has_pages,
            fork=repo.fork,
            private=repo.private,
            html_url=repo.html_url,
            clone_url=repo.clone_url
        )
    
    def get_file_content(self, repo_name: str, file_path: str) -> Optional[FileContent]:
        """Get file content from GitHub."""
        try:
            repo = self._get_repo_object(repo_name)
            file_content = repo.get_contents(file_path)
            
            return FileContent(
                path=file_path,
                content=file_content.decoded_content.decode('utf-8'),
                encoding='utf-8',
                size=file_content.size
            )
        except Exception:
            return None
    
    def list_files(self, repo_name: str, path: str = "") -> List[str]:
        """List files in a GitHub repository directory."""
        try:
            repo = self._get_repo_object(repo_name)
            contents = repo.get_contents(path)
            
            files = []
            for content in contents:
                if content.type == "file":
                    files.append(content.path)
                elif content.type == "dir":
                    # Recursively list files in subdirectories
                    files.extend(self.list_files(repo_name, content.path))
            
            return files
        except Exception:
            return []
    
    def get_issues(self, repo_name: str, state: str = "open", 
                   labels: Optional[List[str]] = None) -> List[IssueInfo]:
        """Get issues from GitHub."""
        repo = self._get_repo_object(repo_name)
        
        issues = []
        github_issues = repo.get_issues(state=state, labels=labels)
        
        for issue in github_issues:
            issues.append(IssueInfo(
                number=issue.number,
                title=issue.title,
                body=issue.body,
                state=issue.state,
                labels=[label.name for label in issue.labels],
                created_at=issue.created_at,
                updated_at=issue.updated_at,
                author=issue.user.login if issue.user else "unknown",
                assignee=issue.assignee.login if issue.assignee else None
            ))
        
        return issues
    
    def get_recent_commits(self, repo_name: str, limit: int = 30) -> List[CommitInfo]:
        """Get recent commits from GitHub."""
        repo = self._get_repo_object(repo_name)
        
        commits = []
        for commit in repo.get_commits()[:limit]:
            commits.append(CommitInfo(
                sha=commit.sha,
                message=commit.commit.message,
                author=commit.commit.author.name,
                date=commit.commit.author.date,
                files_changed=len(commit.files) if commit.files else 0
            ))
        
        return commits
    
    def get_languages(self, repo_name: str) -> Dict[str, int]:
        """Get language statistics from GitHub."""
        repo = self._get_repo_object(repo_name)
        return repo.get_languages()
    
    def get_contributors(self, repo_name: str) -> List[Dict[str, Any]]:
        """Get contributors from GitHub."""
        repo = self._get_repo_object(repo_name)
        
        contributors = []
        for contributor in repo.get_contributors():
            contributors.append({
                'login': contributor.login,
                'contributions': contributor.contributions,
                'avatar_url': contributor.avatar_url,
                'html_url': contributor.html_url
            })
        
        return contributors
    
    def file_exists(self, repo_name: str, file_path: str) -> bool:
        """Check if a file exists on GitHub."""
        try:
            repo = self._get_repo_object(repo_name)
            repo.get_contents(file_path)
            return True
        except Exception:
            return False


class LocalDataProvider(RepositoryDataProvider):
    """Local filesystem data provider implementation."""
    
    def __init__(self, local_simulator):
        """Initialize with local simulator."""
        self.simulator = local_simulator
        self.base_path = Path(local_simulator.repos_path)
    
    def get_repository(self, repo_name: str) -> RepositoryInfo:
        """Get repository information from local filesystem."""
        org, name = repo_name.split('/')
        repo = self.simulator.get_repository(org, name)
        
        if not repo:
            raise ValueError(f"Repository {repo_name} not found")
        
        # Get repository path
        repo_path = self.base_path / org / name
        
        # Calculate size (in KB)
        total_size = sum(
            f.stat().st_size for f in repo_path.rglob('*') if f.is_file()
        ) // 1024
        
        return RepositoryInfo(
            name=name,
            full_name=repo_name,
            description=repo.description,
            language="Unknown",  # Would need analysis
            created_at=repo.created_at,
            updated_at=repo.updated_at,
            size=total_size,
            default_branch="main",
            topics=[],
            has_issues=True,
            open_issues_count=len(repo.issues),
            has_wiki=False,
            has_pages=False,
            fork=False,
            private=False,
            html_url=f"file://{repo_path}",
            clone_url=f"file://{repo_path}"
        )
    
    def get_file_content(self, repo_name: str, file_path: str) -> Optional[FileContent]:
        """Get file content from local filesystem."""
        org, name = repo_name.split('/')
        full_path = self.base_path / org / name / file_path
        
        if not full_path.exists():
            return None
        
        try:
            content = full_path.read_text(encoding='utf-8')
            return FileContent(
                path=file_path,
                content=content,
                encoding='utf-8',
                size=full_path.stat().st_size
            )
        except Exception:
            return None
    
    def list_files(self, repo_name: str, path: str = "") -> List[str]:
        """List files in local repository."""
        org, name = repo_name.split('/')
        base_path = self.base_path / org / name
        search_path = base_path / path if path else base_path
        
        files = []
        if search_path.exists():
            for file_path in search_path.rglob('*'):
                if file_path.is_file():
                    relative_path = file_path.relative_to(base_path)
                    files.append(str(relative_path))
        
        return files
    
    def get_issues(self, repo_name: str, state: str = "open", 
                   labels: Optional[List[str]] = None) -> List[IssueInfo]:
        """Get issues from local storage."""
        org, name = repo_name.split('/')
        repo = self.simulator.get_repository(org, name)
        
        if not repo:
            return []
        
        issues = []
        for issue in repo.issues:
            # Filter by state
            if state != "all" and issue.state != state:
                continue
            
            # Filter by labels if specified
            if labels:
                issue_labels = [label['name'] for label in issue.labels]
                if not any(label in issue_labels for label in labels):
                    continue
            
            issues.append(IssueInfo(
                number=issue.number,
                title=issue.title,
                body=issue.body,
                state=issue.state,
                labels=[label['name'] for label in issue.labels],
                created_at=issue.created_at,
                updated_at=issue.updated_at,
                author=issue.author,
                assignee=issue.assignee
            ))
        
        return issues
    
    def get_recent_commits(self, repo_name: str, limit: int = 30) -> List[CommitInfo]:
        """Get recent commits from local repository."""
        # For local repos, we might not have commit history
        # This is a simplified implementation
        return []
    
    def get_languages(self, repo_name: str) -> Dict[str, int]:
        """Analyze languages in local repository."""
        org, name = repo_name.split('/')
        repo_path = self.base_path / org / name
        
        language_stats = {}
        
        # Simple file extension based analysis
        extension_map = {
            '.py': 'Python',
            '.js': 'JavaScript',
            '.jsx': 'JavaScript',
            '.ts': 'TypeScript',
            '.tsx': 'TypeScript',
            '.php': 'PHP',
            '.java': 'Java',
            '.go': 'Go',
            '.rs': 'Rust',
            '.cpp': 'C++',
            '.c': 'C',
            '.cs': 'C#',
            '.rb': 'Ruby',
            '.swift': 'Swift',
            '.kt': 'Kotlin'
        }
        
        for file_path in repo_path.rglob('*'):
            if file_path.is_file():
                ext = file_path.suffix.lower()
                if ext in extension_map:
                    lang = extension_map[ext]
                    language_stats[lang] = language_stats.get(lang, 0) + 1
        
        return language_stats
    
    def get_contributors(self, repo_name: str) -> List[Dict[str, Any]]:
        """Get contributors (not available for local repos)."""
        return [{
            'login': 'local-user',
            'contributions': 1,
            'avatar_url': '',
            'html_url': ''
        }]
    
    def file_exists(self, repo_name: str, file_path: str) -> bool:
        """Check if file exists in local repository."""
        org, name = repo_name.split('/')
        full_path = self.base_path / org / name / file_path
        return full_path.exists()