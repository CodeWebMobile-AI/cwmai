"""
State Manager Module

Handles system state persistence, loading, and I/O operations for the autonomous AI system.
Provides comprehensive state management with both local and remote GitHub repository persistence.
"""

import json
import os
import base64
from datetime import datetime, timezone
from typing import Dict, Any, Optional, List
from github import Github
from .security_validator import security_validator, safe_json_load


class StateManager:
    """Manages system state persistence and I/O operations."""
    
    DEFAULT_STATE = {
        "charter": {
            "primary_goal": "innovation",
            "secondary_goal": "community_engagement", 
            "constraints": ["maintain_quality", "ensure_security"]
        },
        "projects": {
            "sample-project": {
                "health_score": 85,
                "last_checked": datetime.now(timezone.utc).isoformat(),
                "action_history": [
                    {
                        "action": "initial_setup",
                        "details": "Project initialized with default configuration",
                        "outcome": "success_merged",
                        "timestamp": datetime.now(timezone.utc).isoformat()
                    }
                ],
                "metrics": {
                    "stars": 0,
                    "forks": 0,
                    "issues_open": 0,
                    "pull_requests_open": 0
                }
            }
        },
        "system_performance": {
            "total_cycles": 0,
            "successful_actions": 0,
            "failed_actions": 0,
            "learning_metrics": {
                "decision_accuracy": 0.0,
                "resource_efficiency": 0.0,
                "goal_achievement": 0.0
            }
        },
        "task_queue": [],
        "external_context": {
            "last_updated": datetime.now(timezone.utc).isoformat(),
            "market_trends": [],
            "security_alerts": [],
            "technology_updates": []
        },
        "version": "1.0.0",
        "last_updated": datetime.now(timezone.utc).isoformat()
    }
    
    def __init__(self, local_path: str = "system_state.json", repo_path: str = "system_state.json"):
        """Initialize the StateManager.
        
        Args:
            local_path: Local file path for state storage
            repo_path: Repository path for remote state storage
        """
        self.local_path = local_path
        self.repo_path = repo_path
        self.github_token = os.getenv('CLAUDE_PAT') or os.getenv('GITHUB_TOKEN')
        self.repo_name = os.getenv('GITHUB_REPOSITORY', 'CodeWebMobile-AI/cwmai')
        self.organization = 'CodeWebMobile-AI'
        
    def discover_organization_repositories(self) -> List[Dict[str, Any]]:
        """Discover all repositories in the CodeWebMobile-AI organization.
        
        Returns:
            List of repository information dictionaries
        """
        if not self.github_token:
            print("Warning: GitHub token not available for repository discovery")
            return []
            
        try:
            g = Github(self.github_token)
            org = g.get_organization(self.organization)
            repositories = []
            
            print(f"Discovering repositories in {self.organization} organization...")
            
            for repo in org.get_repos():
                # Skip archived or disabled repositories
                if repo.archived or repo.disabled:
                    continue
                    
                try:
                    # Get repository metrics and health data
                    repo_data = {
                        'name': repo.name,
                        'full_name': repo.full_name,
                        'description': repo.description or f"Repository: {repo.name}",
                        'url': repo.html_url,
                        'clone_url': repo.clone_url,
                        'language': repo.language,
                        'private': repo.private,
                        'created_at': repo.created_at.isoformat() if repo.created_at else None,
                        'updated_at': repo.updated_at.isoformat() if repo.updated_at else None,
                        'pushed_at': repo.pushed_at.isoformat() if repo.pushed_at else None,
                        'size': repo.size,
                        'metrics': {
                            'stars': repo.stargazers_count,
                            'forks': repo.forks_count,
                            'watchers': repo.watchers_count,
                            'issues_open': repo.open_issues_count,
                            'has_issues': repo.has_issues,
                            'has_projects': repo.has_projects,
                            'has_wiki': repo.has_wiki
                        },
                        'topics': list(repo.get_topics()),
                        'default_branch': repo.default_branch,
                        'health_score': self._calculate_repository_health_score(repo),
                        'discovered_at': datetime.now(timezone.utc).isoformat()
                    }
                    
                    # Get recent activity summary
                    repo_data['recent_activity'] = self._get_repository_activity_summary(repo)
                    
                    repositories.append(repo_data)
                    print(f"✓ Discovered repository: {repo.name}")
                    
                except Exception as e:
                    print(f"Error processing repository {repo.name}: {e}")
                    continue
            
            print(f"Successfully discovered {len(repositories)} repositories")
            return repositories
            
        except Exception as e:
            print(f"Error discovering organization repositories: {e}")
            return []
    
    def _calculate_repository_health_score(self, repo) -> float:
        """Calculate a health score for a repository based on various metrics.
        
        Args:
            repo: GitHub repository object
            
        Returns:
            Health score between 0.0 and 100.0
        """
        try:
            score = 50.0  # Base score
            
            # Recent activity (pushed within last 30 days)
            if repo.pushed_at:
                days_since_push = (datetime.now(timezone.utc) - repo.pushed_at.replace(tzinfo=timezone.utc)).days
                if days_since_push <= 7:
                    score += 20
                elif days_since_push <= 30:
                    score += 10
                elif days_since_push <= 90:
                    score += 5
            
            # Documentation and setup
            if repo.has_wiki:
                score += 5
            if repo.description:
                score += 5
                
            # Community engagement
            if repo.stargazers_count > 0:
                score += min(10, repo.stargazers_count)
            if repo.forks_count > 0:
                score += min(5, repo.forks_count)
                
            # Project management
            if repo.has_issues:
                score += 5
            if repo.has_projects:
                score += 5
                
            # Size and content (reasonable size indicates active development)
            if repo.size > 100:  # Has some content
                score += 5
                
            return min(100.0, max(0.0, score))
            
        except Exception as e:
            print(f"Error calculating health score for {repo.name}: {e}")
            return 50.0
    
    def _get_repository_activity_summary(self, repo) -> Dict[str, Any]:
        """Get a summary of recent repository activity.
        
        Args:
            repo: GitHub repository object
            
        Returns:
            Dictionary with activity summary
        """
        try:
            activity = {
                'recent_commits': 0,
                'recent_issues': 0,
                'recent_prs': 0,
                'contributors_count': 0,
                'last_commit_date': None,
                'active_branches': 0
            }
            
            # Get recent commits (last 30 days)
            try:
                from datetime import timedelta
                since_date = datetime.now(timezone.utc) - timedelta(days=30)
                commits = list(repo.get_commits(since=since_date))
                activity['recent_commits'] = len(commits)
                if commits:
                    activity['last_commit_date'] = commits[0].commit.author.date.isoformat()
            except:
                pass
            
            # Get contributors count
            try:
                contributors = list(repo.get_contributors())
                activity['contributors_count'] = len(contributors)
            except:
                pass
                
            # Count branches
            try:
                branches = list(repo.get_branches())
                activity['active_branches'] = len(branches)
            except:
                pass
            
            return activity
            
        except Exception as e:
            print(f"Error getting activity summary for {repo.name}: {e}")
            return {
                'recent_commits': 0,
                'recent_issues': 0,
                'recent_prs': 0,
                'contributors_count': 0,
                'last_commit_date': None,
                'active_branches': 0
            }
    
    def load_state_with_repository_discovery(self) -> Dict[str, Any]:
        """Load system state and automatically discover organization repositories.
        
        Returns:
            Dict containing the system state with discovered repositories
        """
        # Load existing state
        state = self.load_state()
        
        # Discover repositories and update state
        discovered_repos = self.discover_organization_repositories()
        
        if discovered_repos:
            # Clear sample project and replace with real repositories
            state['projects'] = {}
            
            for repo_data in discovered_repos:
                project_id = repo_data['name']
                state['projects'][project_id] = {
                    'name': repo_data['name'],
                    'full_name': repo_data['full_name'],
                    'description': repo_data['description'],
                    'url': repo_data['url'],
                    'clone_url': repo_data['clone_url'],
                    'language': repo_data['language'],
                    'health_score': repo_data['health_score'],
                    'last_checked': repo_data['discovered_at'],
                    'metrics': repo_data['metrics'],
                    'recent_activity': repo_data['recent_activity'],
                    'topics': repo_data['topics'],
                    'default_branch': repo_data['default_branch'],
                    'type': 'github_repository',
                    'status': 'active',
                    'action_history': [
                        {
                            "action": "repository_discovered",
                            "details": f"Repository {repo_data['name']} discovered and integrated into system",
                            "outcome": "success_discovered",
                            "timestamp": repo_data['discovered_at']
                        }
                    ]
                }
            
            # Update state metadata
            state['repository_discovery'] = {
                'last_discovery': datetime.now(timezone.utc).isoformat(),
                'repositories_found': len(discovered_repos),
                'discovery_source': 'github_organization',
                'organization': self.organization
            }
            
            # Save updated state
            self.save_state_locally(state)
            
            print(f"✓ Integrated {len(discovered_repos)} repositories into system state")
        else:
            print("No repositories discovered, keeping existing state")
        
        return state

    def load_state(self) -> Dict[str, Any]:
        """Load system state with fallback hierarchy.
        
        Tries local file first, then remote repository, then creates default state.
        
        Returns:
            Dict containing the system state
        """
        # Try loading from local file first
        if os.path.exists(self.local_path):
            try:
                with open(self.local_path, 'r') as f:
                    # Use secure JSON loading with validation
                    state = safe_json_load(f.read())
                    self._validate_and_migrate_state(state)
                    return state
            except (json.JSONDecodeError, ValueError, KeyError) as e:
                print(f"Error loading local state: {e}")
        
        # Try loading from remote repository
        try:
            state = self._load_from_repo()
            if state:
                self._validate_and_migrate_state(state)
                # Save locally for next step in workflow
                self.save_state_locally(state)
                return state
        except Exception as e:
            print(f"Error loading remote state: {e}")
        
        # Create and return default state
        print("Creating new default state")
        default_state = self.DEFAULT_STATE.copy()
        default_state["last_updated"] = datetime.now(timezone.utc).isoformat()
        self.save_state_locally(default_state)
        return default_state
    
    def _load_from_repo(self) -> Optional[Dict[str, Any]]:
        """Load state from GitHub repository.
        
        Returns:
            Dict containing the state or None if not found
        """
        if not self.github_token:
            raise ValueError("GitHub token not available")
            
        g = Github(self.github_token)
        repo = g.get_repo(self.repo_name)
        
        try:
            file_content = repo.get_contents(self.repo_path)
            if file_content.encoding == 'base64':
                content = base64.b64decode(file_content.content).decode('utf-8')
            else:
                content = file_content.content
            
            # Use secure JSON loading with validation
            return safe_json_load(content)
        except Exception as e:
            print(f"State file not found in repository: {e}")
            return None
    
    def save_state_locally(self, state_data: Dict[str, Any]) -> None:
        """Save state to local file.
        
        Args:
            state_data: State dictionary to save
        """
        state_data["last_updated"] = datetime.now(timezone.utc).isoformat()
        
        with open(self.local_path, 'w') as f:
            json.dump(state_data, f, indent=2, sort_keys=True)
    
    def save_state_to_repo(self, state_data: Optional[Dict[str, Any]] = None) -> None:
        """Save state to GitHub repository.
        
        Args:
            state_data: State dictionary to save. If None, loads from local file.
        """
        if not self.github_token:
            print("GitHub token not available for remote save")
            return
            
        if state_data is None:
            if os.path.exists(self.local_path):
                with open(self.local_path, 'r') as f:
                    state_data = json.load(f)
            else:
                print("No local state file to save")
                return
        
        try:
            g = Github(self.github_token)
            repo = g.get_repo(self.repo_name)
            
            state_data["last_updated"] = datetime.now(timezone.utc).isoformat()
            content = json.dumps(state_data, indent=2, sort_keys=True)
            
            try:
                # Try to update existing file
                file_content = repo.get_contents(self.repo_path)
                repo.update_file(
                    path=self.repo_path,
                    message="Update AI system state",
                    content=content,
                    sha=file_content.sha
                )
            except:
                # Create new file if it doesn't exist
                repo.create_file(
                    path=self.repo_path,
                    message="Create AI system state",
                    content=content
                )
            
            print("State successfully saved to repository")
            
        except Exception as e:
            print(f"Error saving state to repository: {e}")
    
    def _validate_and_migrate_state(self, state: Dict[str, Any]) -> None:
        """Validate and migrate state structure if needed.
        
        Args:
            state: State dictionary to validate and migrate
        """
        # Ensure all required top-level keys exist
        required_keys = ["charter", "projects", "system_performance", "task_queue"]
        for key in required_keys:
            if key not in state:
                state[key] = self.DEFAULT_STATE[key].copy()
        
        # Ensure charter has required fields
        charter_keys = ["primary_goal", "secondary_goal", "constraints"]
        for key in charter_keys:
            if key not in state["charter"]:
                state["charter"][key] = self.DEFAULT_STATE["charter"][key]
        
        # Ensure system_performance has required structure
        if "learning_metrics" not in state["system_performance"]:
            state["system_performance"]["learning_metrics"] = self.DEFAULT_STATE["system_performance"]["learning_metrics"].copy()
        
        # Add version if missing
        if "version" not in state:
            state["version"] = "1.0.0"
    
    def load_workflow_state(self) -> Dict[str, Any]:
        """Load minimal state optimized for GitHub Actions workflow.
        
        Returns lightweight state with only essential data for CI performance.
        
        Returns:
            Minimal state dictionary for workflow execution
        """
        try:
            # Load full state first
            full_state = self.load_state()
            
            # Extract only essential data for workflow
            workflow_state = {
                'charter': full_state.get('charter', {
                    'purpose': 'autonomous_ai_development',
                    'version': 'workflow_v1',
                    'created_at': datetime.now(timezone.utc).isoformat()
                }),
                'projects': self._get_essential_projects(full_state.get('projects', {})),
                'system_performance': self._get_essential_performance(full_state.get('system_performance', {})),
                'last_updated': datetime.now(timezone.utc).isoformat(),
                'environment': 'github_actions'
            }
            
            print("Loaded workflow-optimized state")
            return workflow_state
            
        except Exception as e:
            print(f"Failed to load workflow state: {e}, using minimal defaults")
            return self._get_minimal_workflow_state()
    
    def load_production_state(self) -> Dict[str, Any]:
        """Load complete state for production environment.
        
        Loads all available state data including history, metrics, and configurations.
        
        Returns:
            Complete state dictionary for production use
        """
        try:
            # Load full state with all historical data
            state = self.load_state()
            
            # Add production-specific metadata
            state.update({
                'environment': 'production',
                'full_history_loaded': True,
                'last_production_load': datetime.now(timezone.utc).isoformat()
            })
            
            print("Loaded complete production state")
            return state
            
        except Exception as e:
            print(f"Failed to load production state: {e}")
            raise
    
    def _get_essential_projects(self, projects: Dict[str, Any]) -> Dict[str, Any]:
        """Extract essential project data for workflow."""
        essential_projects = {}
        
        for project_id, project_data in projects.items():
            if isinstance(project_data, dict):
                # Keep only essential project fields
                essential_projects[project_id] = {
                    'name': project_data.get('name', project_id),
                    'status': project_data.get('status', 'unknown'),
                    'health_score': project_data.get('health_score', 0.5),
                    'type': project_data.get('type', 'unknown'),
                    'last_updated': project_data.get('last_updated')
                }
        
        return essential_projects
    
    def _get_essential_performance(self, performance: Dict[str, Any]) -> Dict[str, Any]:
        """Extract essential performance metrics for workflow."""
        return {
            'success_rate': performance.get('success_rate', 0.0),
            'total_tasks_completed': performance.get('total_tasks_completed', 0),
            'avg_completion_time': performance.get('avg_completion_time', 0),
            'last_updated': performance.get('last_updated')
        }
    
    def _get_minimal_workflow_state(self) -> Dict[str, Any]:
        """Get minimal default state for workflow fallback."""
        return {
            'charter': {
                'purpose': 'autonomous_ai_development_fallback',
                'version': 'minimal_v1',
                'created_at': datetime.now(timezone.utc).isoformat()
            },
            'projects': {},
            'system_performance': {
                'success_rate': 0.0,
                'total_tasks_completed': 0
            },
            'environment': 'github_actions_fallback',
            'minimal_mode': True
        }
    
