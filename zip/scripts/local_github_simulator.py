"""
Local GitHub Simulator

Simulates GitHub API functionality using local file system storage.
Provides a drop-in replacement for GitHub API calls without rate limits.
"""

import json
import os
import shutil
import subprocess
import base64
from datetime import datetime, timezone
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
import logging
import uuid


class LocalRepository:
    """Represents a local repository with GitHub-like properties."""
    
    def __init__(self, name: str, full_name: str, description: str = "", path: str = ""):
        self.name = name
        self.full_name = full_name
        self.description = description
        self.path = path
        self.created_at = datetime.now(timezone.utc)
        self.updated_at = datetime.now(timezone.utc)
        self.default_branch = "main"
        self.private = False
        self.html_url = f"file://{path}"
        self.language = None  # Language detection would require file analysis
        self.languages_url = None
        self.open_issues_count = 0
        self.forks_count = 0
        self.stargazers_count = 0
    
    def get_contents(self, path: str):
        """Get file contents from local repository.
        
        Args:
            path: Path to file relative to repository root
            
        Returns:
            Mock content object with GitHub API-like interface
            
        Raises:
            FileNotFoundError: If file doesn't exist
        """
        full_path = Path(self.path) / path
        if full_path.exists() and full_path.is_file():
            # Return a mock content object
            content = full_path.read_text()
            encoded_content = base64.b64encode(content.encode()).decode()
            return type('MockContent', (), {
                'content': encoded_content,
                'encoding': 'base64',
                'decoded_content': content,
                'path': path,
                'name': Path(path).name,
                'size': full_path.stat().st_size,
                'type': 'file',
                'sha': None  # Mock SHA for GitHub API compatibility
            })()
        else:
            raise FileNotFoundError(f"'{path}' not found in repository")
    
    def update_file(self, path: str, message: str, content: str, sha: str = None):
        """Update or create a file in the repository.
        
        Args:
            path: Path to file relative to repository root
            message: Commit message
            content: File content (base64 encoded)
            sha: SHA of existing file (ignored in local mode)
            
        Returns:
            Mock commit object
        """
        full_path = Path(self.path) / path
        
        # Ensure parent directory exists
        full_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Decode content if base64 encoded
        try:
            decoded_content = base64.b64decode(content).decode('utf-8')
        except:
            # If decoding fails, assume it's plain text
            decoded_content = content
        
        # Write file
        with open(full_path, 'w') as f:
            f.write(decoded_content)
        
        # Git add and commit
        try:
            subprocess.run(["git", "add", path], cwd=self.path, check=True, capture_output=True)
            subprocess.run(
                ["git", "commit", "-m", message], 
                cwd=self.path, 
                check=True, 
                capture_output=True
            )
        except subprocess.CalledProcessError:
            # If commit fails (no changes), that's okay
            pass
        
        # Update timestamp
        self.updated_at = datetime.now(timezone.utc)
        
        # Return mock commit object
        return type('MockCommit', (), {
            'sha': str(uuid.uuid4()),
            'commit': type('MockCommitDetails', (), {
                'message': message,
                'author': {'name': 'Local System', 'email': 'local@system'},
                'committer': {'name': 'Local System', 'email': 'local@system'}
            })(),
            'content': type('MockContent', (), {
                'path': path,
                'size': full_path.stat().st_size,
                'sha': str(uuid.uuid4())
            })()
        })()
    
    def create_file(self, path: str, message: str, content: str):
        """Create a new file in the repository.
        
        Args:
            path: Path to file relative to repository root
            message: Commit message
            content: File content (base64 encoded)
            
        Returns:
            Mock commit object
        """
        return self.update_file(path, message, content)
    
    def get_issue(self, number: int):
        """Get an issue by number.
        
        Args:
            number: Issue number
            
        Returns:
            Issue object or raises exception
        """
        issues_dir = Path(self.path) / "issues"
        issue_file = issues_dir / f"issue-{number}.json"
        
        if not issue_file.exists():
            raise ValueError(f"Issue #{number} not found")
            
        with open(issue_file, 'r') as f:
            issue_data = json.loads(f.read())
            
        # Create LocalIssue object
        from . import local_github_simulator
        issue = local_github_simulator.LocalIssue(
            issue_data["number"],
            issue_data["title"],
            issue_data["body"],
            issue_data.get("labels", [])
        )
        issue.state = issue_data["state"]
        return issue
    
    def get_issues(self, state="open", labels=None):
        """Get repository issues.
        
        Args:
            state: Issue state filter
            labels: Label filter
            
        Returns:
            List of issues
        """
        # Delegate to simulator
        from . import local_github_simulator
        simulator = local_github_simulator.LocalGitHubSimulator()
        return simulator.get_issues(self.full_name, state, labels)
    
    def create_issue(self, title: str, body: str = "", labels: list = None):
        """Create a new issue.
        
        Args:
            title: Issue title
            body: Issue body
            labels: Issue labels
            
        Returns:
            Created issue
        """
        # Delegate to simulator
        from . import local_github_simulator
        simulator = local_github_simulator.LocalGitHubSimulator()
        return simulator.create_issue(self.full_name, title, body, labels)
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "name": self.name,
            "full_name": self.full_name,
            "description": self.description,
            "path": self.path,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "default_branch": self.default_branch,
            "private": self.private,
            "html_url": self.html_url,
            "language": self.language,
            "languages_url": self.languages_url,
            "open_issues_count": self.open_issues_count,
            "forks_count": self.forks_count,
            "stargazers_count": self.stargazers_count
        }


class LocalIssue:
    """Represents a local issue with GitHub-like properties."""
    
    def __init__(self, number: int, title: str, body: str, labels: List[str] = None):
        self.number = number
        self.title = title
        self.body = body
        self.labels = labels or []
        self.state = "open"
        self.created_at = datetime.now(timezone.utc)
        self.updated_at = datetime.now(timezone.utc)
        self.closed_at = None
        self.assignee = None
        self.html_url = f"local://issue/{number}"
        self.id = str(uuid.uuid4())
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "id": self.id,
            "number": self.number,
            "title": self.title,
            "body": self.body,
            "labels": self.labels,
            "state": self.state,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "closed_at": self.closed_at.isoformat() if self.closed_at else None,
            "assignee": self.assignee,
            "html_url": self.html_url
        }


class LocalGitHubSimulator:
    """Simulates GitHub API using local file system."""
    
    def __init__(self, base_path: str = None):
        """Initialize the local GitHub simulator.
        
        Args:
            base_path: Base directory for local repositories (defaults to ./local_repos)
        """
        # Use relative path - look for local_repos in current working directory
        if base_path:
            self.base_path = Path(base_path)
        else:
            # Default to local_repos in the current working directory
            # Use script location instead of working directory
            script_dir = Path(__file__).parent.parent
            # Since we're in zip/scripts, parent.parent is the zip directory
            self.base_path = script_dir / "local_repos"
        self.repos_path = self.base_path / "repositories"
        self.state_path = self.base_path / "state"
        self.templates_path = self.base_path / "templates"
        
        self.logger = logging.getLogger(__name__)
        
        # Ensure directories exist
        self.repos_path.mkdir(parents=True, exist_ok=True)
        self.state_path.mkdir(parents=True, exist_ok=True)
        self.templates_path.mkdir(parents=True, exist_ok=True)
        
        # Load state
        self._load_state()
        
    def _load_state(self):
        """Load state from JSON files."""
        # Load repositories
        repos_file = self.state_path / "repositories.json"
        if repos_file.exists():
            with open(repos_file, 'r') as f:
                self.repos_state = json.load(f)
        else:
            self.repos_state = {
                "repositories": [],
                "total_count": 0,
                "last_updated": None,
                "metadata": {"version": "1.0", "storage_mode": "local"}
            }
            
    def _save_state(self):
        """Save state to JSON files."""
        # Save repositories state
        repos_file = self.state_path / "repositories.json"
        self.repos_state["last_updated"] = datetime.now(timezone.utc).isoformat()
        with open(repos_file, 'w') as f:
            json.dump(self.repos_state, f, indent=2)
            
    def create_repository(self, name: str, description: str = "", 
                         organization: str = "CodeWebMobile-AI", 
                         template: str = None) -> LocalRepository:
        """Create a new local repository.
        
        Args:
            name: Repository name
            description: Repository description
            organization: Organization name
            template: Template to copy (e.g., 'react-starter-kit')
            
        Returns:
            LocalRepository object
        """
        full_name = f"{organization}/{name}"
        repo_path = self.repos_path / organization / name
        
        # Check if repository already exists
        if repo_path.exists():
            raise ValueError(f"Repository {full_name} already exists at {repo_path}")
        
        # Log existing repositories before creation
        org_path = self.repos_path / organization
        if org_path.exists():
            existing = [p.name for p in org_path.iterdir() if p.is_dir()]
            self.logger.info(f"Existing repositories in {organization}: {existing}")
            
        # Create repository directory
        repo_path.mkdir(parents=True, exist_ok=True)
        self.logger.info(f"Created repository directory: {repo_path}")
        
        # Initialize git repository
        try:
            # If template is specified, copy template files first
            if template:
                template_path = self.templates_path / template
                if template_path.exists() and template_path.is_dir():
                    self.logger.info(f"Copying template from {template_path} to {repo_path}")
                    
                    # Copy all files from template to new repository
                    for item in template_path.iterdir():
                        if item.name == '.git':
                            continue  # Skip .git directory
                        
                        if item.is_file():
                            shutil.copy2(item, repo_path / item.name)
                        elif item.is_dir():
                            shutil.copytree(item, repo_path / item.name, dirs_exist_ok=True)
                    
                    self.logger.info(f"✅ Successfully copied template '{template}' to {repo_path}")
                else:
                    self.logger.warning(f"Template '{template}' not found at {template_path}")
            
            # Initialize git repository
            subprocess.run(["git", "init"], cwd=repo_path, check=True, capture_output=True)
            
            # Create or update README
            readme_path = repo_path / "README.md"
            if not readme_path.exists():
                with open(readme_path, 'w') as f:
                    f.write(f"# {name}\n\n{description}\n\n")
                    f.write("This repository was created by the Local GitHub Simulator.\n")
                
            # Create issues directory
            issues_dir = repo_path / "issues"
            issues_dir.mkdir(exist_ok=True)
            
            # Create metadata file
            metadata = {
                "name": name,
                "full_name": full_name,
                "description": description,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "updated_at": datetime.now(timezone.utc).isoformat(),
                "next_issue_number": 1,
                "organization": organization,
                "template": template
            }
            
            metadata_path = repo_path / "metadata.json"
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
                
            # Initial git commit
            subprocess.run(["git", "add", "."], cwd=repo_path, check=True, capture_output=True)
            subprocess.run(
                ["git", "commit", "-m", "Initial commit"], 
                cwd=repo_path, 
                check=True, 
                capture_output=True
            )
            
            if template and not (self.templates_path / template).exists():
                self.logger.warning(
                    f"⚠️  Created repository at {repo_path} but template '{template}' not found. "
                    f"Please ensure template exists at {self.templates_path / template}"
                )
            
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Failed to initialize git repository: {e}")
            # Clean up on failure
            shutil.rmtree(repo_path)
            raise
            
        # Create repository object
        repo = LocalRepository(name, full_name, description, str(repo_path))
        
        # Update state
        self.repos_state["repositories"].append(repo.to_dict())
        self.repos_state["total_count"] += 1
        self._save_state()
        
        # Verify repository still exists after all operations
        if not repo_path.exists():
            raise RuntimeError(f"Repository {full_name} was deleted after creation!")
            
        # Log final state
        org_path = self.repos_path / organization
        if org_path.exists():
            final_repos = [p.name for p in org_path.iterdir() if p.is_dir()]
            self.logger.info(f"Final repositories in {organization}: {final_repos}")
        
        self.logger.info(f"Created local repository: {full_name}")
        return repo
        
    def get_repository(self, full_name: str) -> Optional[LocalRepository]:
        """Get a repository by full name.
        
        Args:
            full_name: Full repository name (org/repo)
            
        Returns:
            LocalRepository object or None if not found
        """
        for repo_dict in self.repos_state["repositories"]:
            # Handle both old and new formats
            if "full_name" in repo_dict:
                repo_full_name = repo_dict["full_name"]
            else:
                # Construct full_name from organization and name
                org = repo_dict.get("organization", "CodeWebMobile-AI")
                repo_full_name = f"{org}/{repo_dict['name']}"
            
            if repo_full_name == full_name:
                repo = LocalRepository(
                    repo_dict["name"],
                    repo_full_name,
                    repo_dict.get("description", ""),
                    repo_dict["path"]
                )
                return repo
        return None  # Repository not found
        
    def create_issue(self, repo_full_name: str, title: str, body: str, 
                    labels: List[str] = None) -> LocalIssue:
        """Create a new issue in a repository.
        
        Args:
            repo_full_name: Full repository name (org/repo)
            title: Issue title
            body: Issue body
            labels: List of labels
            
        Returns:
            LocalIssue object
        """
        # Parse repository path
        parts = repo_full_name.split('/')
        if len(parts) == 2:
            org, repo_name = parts
        else:
            repo_name = repo_full_name
            org = "CodeWebMobile-AI"
            
        repo_path = self.repos_path / org / repo_name
        
        if not repo_path.exists():
            raise ValueError(f"Repository {repo_full_name} does not exist")
            
        # Load repository metadata
        metadata_path = repo_path / "metadata.json"
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
            
        # Get next issue number
        issue_number = metadata.get("next_issue_number", 1)
        metadata["next_issue_number"] = issue_number + 1
        
        # Save updated metadata
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
            
        # Create issue object
        issue = LocalIssue(issue_number, title, body, labels)
        
        # Save issue to file
        issues_dir = repo_path / "issues"
        issue_file = issues_dir / f"issue-{issue_number}.json"
        
        with open(issue_file, 'w') as f:
            json.dump(issue.to_dict(), f, indent=2)
            
        self.logger.info(f"Created issue #{issue_number} in {repo_full_name}")
        return issue
        
    def get_issues(self, repo_full_name: str, state: str = "open", 
                  labels: List[str] = None) -> List[LocalIssue]:
        """Get issues for a repository.
        
        Args:
            repo_full_name: Full repository name (org/repo)
            state: Issue state filter ('open', 'closed', or 'all')
            labels: Label filter
            
        Returns:
            List of LocalIssue objects
        """
        # Parse repository path
        parts = repo_full_name.split('/')
        if len(parts) == 2:
            org, repo_name = parts
        else:
            repo_name = repo_full_name
            org = "CodeWebMobile-AI"
            
        repo_path = self.repos_path / org / repo_name
        
        if not repo_path.exists():
            return []
            
        issues_dir = repo_path / "issues"
        if not issues_dir.exists():
            return []
            
        issues = []
        
        # Read all issue files
        for issue_file in issues_dir.glob("issue-*.json"):
            with open(issue_file, 'r') as f:
                issue_data = json.load(f)
                
            # Apply filters
            if state != "all" and issue_data["state"] != state:
                continue
                
            if labels:
                issue_labels = set(issue_data.get("labels", []))
                if not any(label in issue_labels for label in labels):
                    continue
                    
            # Create issue object
            issue = LocalIssue(
                issue_data["number"],
                issue_data["title"],
                issue_data["body"],
                issue_data.get("labels", [])
            )
            issue.state = issue_data["state"]
            
            # Parse datetime fields with timezone handling
            try:
                created_at_str = issue_data["created_at"]
                self.logger.debug(f"Parsing created_at: {created_at_str}")
                issue.created_at = datetime.fromisoformat(created_at_str)
                
                # Ensure timezone awareness
                if issue.created_at.tzinfo is None:
                    self.logger.debug(f"created_at is naive, adding UTC timezone")
                    issue.created_at = issue.created_at.replace(tzinfo=timezone.utc)
            except Exception as e:
                self.logger.error(f"Error parsing created_at: {e}")
                issue.created_at = datetime.now(timezone.utc)
            
            try:
                updated_at_str = issue_data["updated_at"]
                issue.updated_at = datetime.fromisoformat(updated_at_str)
                
                # Ensure timezone awareness
                if issue.updated_at.tzinfo is None:
                    issue.updated_at = issue.updated_at.replace(tzinfo=timezone.utc)
            except Exception as e:
                self.logger.error(f"Error parsing updated_at: {e}")
                issue.updated_at = datetime.now(timezone.utc)
            
            if issue_data.get("closed_at"):
                try:
                    closed_at_str = issue_data["closed_at"]
                    issue.closed_at = datetime.fromisoformat(closed_at_str)
                    
                    # Ensure timezone awareness
                    if issue.closed_at.tzinfo is None:
                        issue.closed_at = issue.closed_at.replace(tzinfo=timezone.utc)
                except Exception as e:
                    self.logger.error(f"Error parsing closed_at: {e}")
                    issue.closed_at = None
                
            issues.append(issue)
            
        # Sort by creation date (newest first)
        issues.sort(key=lambda x: x.created_at, reverse=True)
        
        return issues
        
    def update_issue(self, repo_full_name: str, issue_number: int, 
                    state: str = None, labels: List[str] = None) -> LocalIssue:
        """Update an existing issue.
        
        Args:
            repo_full_name: Full repository name (org/repo)
            issue_number: Issue number
            state: New state ('open' or 'closed')
            labels: New labels list
            
        Returns:
            Updated LocalIssue object
        """
        # Parse repository path
        parts = repo_full_name.split('/')
        if len(parts) == 2:
            org, repo_name = parts
        else:
            repo_name = repo_full_name
            org = "CodeWebMobile-AI"
            
        repo_path = self.repos_path / org / repo_name
        issues_dir = repo_path / "issues"
        issue_file = issues_dir / f"issue-{issue_number}.json"
        
        if not issue_file.exists():
            raise ValueError(f"Issue #{issue_number} not found in {repo_full_name}")
            
        # Load issue data
        with open(issue_file, 'r') as f:
            issue_data = json.load(f)
            
        # Update fields
        if state:
            issue_data["state"] = state
            if state == "closed" and not issue_data.get("closed_at"):
                issue_data["closed_at"] = datetime.now(timezone.utc).isoformat()
            elif state == "open":
                issue_data["closed_at"] = None
                
        if labels is not None:
            issue_data["labels"] = labels
            
        issue_data["updated_at"] = datetime.now(timezone.utc).isoformat()
        
        # Save updated issue
        with open(issue_file, 'w') as f:
            json.dump(issue_data, f, indent=2)
            
        # Create and return issue object
        issue = LocalIssue(
            issue_data["number"],
            issue_data["title"],
            issue_data["body"],
            issue_data.get("labels", [])
        )
        issue.state = issue_data["state"]
        issue.created_at = datetime.fromisoformat(issue_data["created_at"])
        issue.updated_at = datetime.fromisoformat(issue_data["updated_at"])
        if issue_data.get("closed_at"):
            issue.closed_at = datetime.fromisoformat(issue_data["closed_at"])
            
        return issue
    
    def add_issue_comment(self, repo_full_name: str, issue_number: int, comment_text: str) -> Dict[str, Any]:
        """Add a comment to an issue.
        
        Args:
            repo_full_name: Full repository name (org/repo)
            issue_number: Issue number
            comment_text: Comment text
            
        Returns:
            Comment data dictionary
        """
        # Parse repository path
        parts = repo_full_name.split('/')
        if len(parts) == 2:
            org, repo_name = parts
        else:
            repo_name = repo_full_name
            org = "CodeWebMobile-AI"
            
        repo_path = self.repos_path / org / repo_name
        issues_dir = repo_path / "issues"
        issue_file = issues_dir / f"issue-{issue_number}.json"
        
        if not issue_file.exists():
            raise ValueError(f"Issue #{issue_number} not found in {repo_full_name}")
            
        # Load issue data
        with open(issue_file, 'r') as f:
            issue_data = json.load(f)
            
        # Add comments array if not exists
        if "comments" not in issue_data:
            issue_data["comments"] = []
            
        # Create comment
        comment = {
            "id": len(issue_data["comments"]) + 1,
            "body": comment_text,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "user": "ai-system"
        }
        
        issue_data["comments"].append(comment)
        issue_data["updated_at"] = datetime.now(timezone.utc).isoformat()
        
        # Save updated issue
        with open(issue_file, 'w') as f:
            json.dump(issue_data, f, indent=2)
            
        return comment
        
    def list_repositories(self, organization: str = None) -> List[LocalRepository]:
        """List all repositories or repositories for an organization.
        
        Args:
            organization: Filter by organization (optional)
            
        Returns:
            List of LocalRepository objects
        """
        repositories = []
        
        for repo_dict in self.repos_state["repositories"]:
            # Handle both old and new formats
            if "full_name" in repo_dict:
                full_name = repo_dict["full_name"]
            else:
                # Construct full_name from organization and name
                org = repo_dict.get("organization", "CodeWebMobile-AI")
                full_name = f"{org}/{repo_dict['name']}"
            
            if organization:
                repo_org = full_name.split('/')[0]
                if repo_org != organization:
                    continue
                    
            repo = LocalRepository(
                repo_dict["name"],
                full_name,
                repo_dict.get("description", ""),
                repo_dict["path"]
            )
            repositories.append(repo)
            
        return repositories
        
    def delete_repository(self, full_name: str) -> bool:
        """Delete a repository (move to archives).
        
        Args:
            full_name: Full repository name (org/repo)
            
        Returns:
            True if successful
        """
        repo = self.get_repository(full_name)
        if not repo:
            return False
            
        # Move to archives instead of deleting
        archive_path = self.base_path / "archives" / datetime.now().strftime("%Y%m%d")
        archive_path.mkdir(parents=True, exist_ok=True)
        
        src_path = Path(repo.path)
        dst_path = archive_path / src_path.name
        
        shutil.move(str(src_path), str(dst_path))
        
        # Update state
        self.repos_state["repositories"] = [
            r for r in self.repos_state["repositories"] 
            if r["full_name"] != full_name
        ]
        self.repos_state["total_count"] -= 1
        self._save_state()
        
        self.logger.info(f"Archived repository: {full_name}")
        return True