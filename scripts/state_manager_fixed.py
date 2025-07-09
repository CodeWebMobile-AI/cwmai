# Copy the content from state_manager.py but fix all MCP methods

import json
import os
from typing import Dict, List, Any, Optional
from datetime import datetime, timezone
import logging
from pathlib import Path
import time
from github import Github, Auth
import subprocess
import uuid

# Import MCP client and integration (optional)
try:
    from scripts.mcp_client import MCPClient
    from scripts.mcp_integration import MCPIntegration
    MCP_AVAILABLE = True
except ImportError:
    MCP_AVAILABLE = False


class StateManager:
    """Manages the persistent state of the AI system."""
    
    def __init__(self, state_file: str = "system_state.json", use_mcp: bool = False):
        """Initialize the state manager.
        
        Args:
            state_file: Path to the state file
            use_mcp: Whether to use MCP integration (default: False, disables Redis)
        """
        self.state_file = state_file
        self.backup_dir = "state_backups"
        self.logger = logging.getLogger(__name__)
        self._use_mcp = use_mcp and MCP_AVAILABLE
        self.mcp = None
        
        # Create backup directory if it doesn't exist
        os.makedirs(self.backup_dir, exist_ok=True)
        
        # Initialize MCP if requested and available
        if self._use_mcp and MCP_AVAILABLE:
            try:
                # MCP Integration disabled to remove Redis dependency
                self.logger.info("MCP integration disabled to remove Redis dependency")
                self._use_mcp = False
            except Exception as e:
                self.logger.warning(f"Could not initialize MCP integration: {e}")
                self._use_mcp = False
    
    def load_state(self) -> Dict[str, Any]:
        """Load the system state from file."""
        if os.path.exists(self.state_file):
            try:
                with open(self.state_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                self.logger.error(f"Error loading state: {e}")
                return self._get_default_state()
        return self._get_default_state()
    
    def save_state(self, state: Dict[str, Any]) -> bool:
        """Save the system state to file with backup."""
        try:
            # Create backup of current state
            if os.path.exists(self.state_file):
                timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
                backup_file = os.path.join(self.backup_dir, f"state_backup_{timestamp}.json")
                with open(self.state_file, 'r') as f:
                    with open(backup_file, 'w') as b:
                        b.write(f.read())
                
                # Keep only last 10 backups
                self._cleanup_old_backups()
            
            # Save new state
            with open(self.state_file, 'w') as f:
                json.dump(state, f, indent=2, default=str)
            
            return True
        except Exception as e:
            self.logger.error(f"Error saving state: {e}")
            return False
    
    def update_state(self, updates: Dict[str, Any]) -> bool:
        """Update specific fields in the state."""
        state = self.load_state()
        
        # Deep merge updates into state
        for key, value in updates.items():
            if isinstance(value, dict) and key in state and isinstance(state[key], dict):
                state[key].update(value)
            else:
                state[key] = value
        
        # Update timestamp
        state['last_updated'] = datetime.now(timezone.utc).isoformat()
        
        return self.save_state(state)
    
    def add_project(self, project_data: Dict[str, Any]) -> bool:
        """Add or update a project in the state."""
        state = self.load_state()
        
        if 'projects' not in state:
            state['projects'] = {}
        
        project_id = project_data.get('id', project_data.get('name', str(uuid.uuid4())))
        state['projects'][project_id] = project_data
        state['projects'][project_id]['last_updated'] = datetime.now(timezone.utc).isoformat()
        
        return self.save_state(state)
    
    def get_state(self) -> Dict[str, Any]:
        """Get the current system state."""
        return self.load_state()
    
    def save_state_locally(self, state: Dict[str, Any]) -> bool:
        """Save state locally (alias for save_state for compatibility)."""
        return self.save_state(state)
    
    def _get_default_state(self) -> Dict[str, Any]:
        """Get the default initial state."""
        return {
            "charter": {
                "primary_goal": "Create innovative projects that solve real-world problems",
                "values": [
                    "Innovation",
                    "Quality",
                    "User-centric design",
                    "Continuous improvement",
                    "Ethical AI development"
                ],
                "constraints": [
                    "Work within API rate limits",
                    "Maintain code quality standards",
                    "Ensure security best practices",
                    "Create maintainable solutions"
                ]
            },
            "projects": {},
            "repositories": {},
            "system_performance": {
                "total_cycles": 0,
                "successful_actions": 0,
                "failed_actions": 0,
                "start_time": datetime.now(timezone.utc).isoformat()
            },
            "task_queue": [],
            "completed_tasks": [],
            "last_updated": datetime.now(timezone.utc).isoformat()
        }
    
    def _cleanup_old_backups(self, keep_count: int = 10):
        """Clean up old backup files, keeping only the most recent ones."""
        try:
            backup_files = sorted([
                f for f in os.listdir(self.backup_dir)
                if f.startswith("state_backup_") and f.endswith(".json")
            ])
            
            if len(backup_files) > keep_count:
                for old_backup in backup_files[:-keep_count]:
                    os.remove(os.path.join(self.backup_dir, old_backup))
        except Exception as e:
            self.logger.warning(f"Error cleaning up old backups: {e}")
    
    def load_state_with_repository_discovery(self) -> Dict[str, Any]:
        """Load state and discover repositories from the organization.
        
        This method loads the system state and also discovers all repositories
        from the CodeWebMobile-AI organization on GitHub.
        
        Returns:
            Dict containing the system state with discovered repositories
        """
        # Load existing state
        state = self.load_state()
        
        # Get GitHub token
        github_token = os.getenv('GITHUB_TOKEN') or os.getenv('CLAUDE_PAT')
        if not github_token:
            self.logger.warning("No GitHub token found, skipping repository discovery")
            return state
        
        try:
            # Initialize GitHub client
            auth = Auth.Token(github_token)
            g = Github(auth=auth)
            
            # Get organization
            org_name = "CodeWebMobile-AI"
            org = g.get_organization(org_name)
            
            # Discover all repositories
            discovered_repos = {}
            
            print(f"Discovering repositories from {org_name}...")
            
            for repo in org.get_repos():
                repo_data = {
                    'id': repo.name,
                    'name': repo.name,
                    'full_name': repo.full_name,
                    'description': repo.description or '',
                    'url': repo.html_url,
                    'clone_url': repo.clone_url,
                    'language': repo.language,
                    'created_at': repo.created_at.isoformat() if repo.created_at else None,
                    'updated_at': repo.updated_at.isoformat() if repo.updated_at else None,
                    'topics': list(repo.get_topics()),
                    'open_issues': repo.open_issues_count,
                    'status': 'active',
                    'health_score': self._calculate_health_score(repo),
                    'last_checked': datetime.now(timezone.utc).isoformat(),
                    'metrics': {
                        'stars': repo.stargazers_count,
                        'forks': repo.forks_count,
                        'issues_open': repo.open_issues_count,
                        'size': repo.size
                    }
                }
                
                discovered_repos[repo.name] = repo_data
            
            # Update state with discovered repositories
            if 'projects' not in state:
                state['projects'] = {}
            
            # Merge discovered repos into projects
            for repo_name, repo_data in discovered_repos.items():
                if repo_name not in state['projects']:
                    state['projects'][repo_name] = repo_data
                    print(f"  + Added new repository: {repo_name}")
                else:
                    # Update existing repo data
                    state['projects'][repo_name].update(repo_data)
            
            # Save updated state
            self.save_state(state)
            
            print(f"✓ Discovered {len(discovered_repos)} repositories")
            
        except Exception as e:
            self.logger.error(f"Error discovering repositories: {e}")
        
        return state
    
    def _calculate_health_score(self, repo) -> int:
        """Calculate a health score for a repository.
        
        Args:
            repo: GitHub repository object
            
        Returns:
            Health score from 0-100
        """
        score = 50  # Base score
        
        # Recent activity bonus
        if repo.updated_at:
            days_since_update = (datetime.now(timezone.utc) - repo.updated_at.replace(tzinfo=timezone.utc)).days
            if days_since_update < 7:
                score += 20
            elif days_since_update < 30:
                score += 10
        
        # Has description
        if repo.description:
            score += 10
        
        # Has topics
        if list(repo.get_topics()):
            score += 10
        
        # Low issue count
        if repo.open_issues_count < 5:
            score += 10
        elif repo.open_issues_count > 20:
            score -= 10
        
        return max(0, min(100, score))
    
    def is_project_active(self, project_name: str) -> bool:
        """Check if a project is active.
        
        Args:
            project_name: Name of the project
            
        Returns:
            True if the project is active
        """
        state = self.load_state()
        project = state.get('projects', {}).get(project_name, {})
        return project.get('status') == 'active'
    
    def get_active_projects(self) -> List[Dict[str, Any]]:
        """Get all active projects.
        
        Returns:
            List of active project dictionaries
        """
        state = self.load_state()
        return [
            project for project in state.get('projects', {}).values()
            if project.get('status') == 'active'
        ]
    
    def update_project_health(self, project_name: str, health_data: Dict[str, Any]) -> bool:
        """Update the health data for a project.
        
        Args:
            project_name: Name of the project
            health_data: Health metrics and scores
            
        Returns:
            True if update was successful
        """
        state = self.load_state()
        
        if project_name in state.get('projects', {}):
            state['projects'][project_name]['health'] = health_data
            state['projects'][project_name]['last_health_check'] = datetime.now(timezone.utc).isoformat()
            return self.save_state(state)
        
        return False
    
    def record_action(self, action_type: str, success: bool, details: Dict[str, Any] = None) -> bool:
        """Record a system action for performance tracking.
        
        Args:
            action_type: Type of action performed
            success: Whether the action was successful
            details: Additional details about the action
            
        Returns:
            True if recording was successful
        """
        state = self.load_state()
        
        if 'system_performance' not in state:
            state['system_performance'] = {
                'total_cycles': 0,
                'successful_actions': 0,
                'failed_actions': 0
            }
        
        state['system_performance']['total_cycles'] += 1
        
        if success:
            state['system_performance']['successful_actions'] += 1
        else:
            state['system_performance']['failed_actions'] += 1
        
        # Add to action history
        if 'action_history' not in state:
            state['action_history'] = []
        
        state['action_history'].append({
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'action_type': action_type,
            'success': success,
            'details': details or {}
        })
        
        # Keep only last 1000 actions
        if len(state['action_history']) > 1000:
            state['action_history'] = state['action_history'][-1000:]
        
        return self.save_state(state)
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get system performance metrics.
        
        Returns:
            Dictionary of performance metrics
        """
        state = self.load_state()
        perf = state.get('system_performance', {})
        
        total = perf.get('total_cycles', 0)
        successful = perf.get('successful_actions', 0)
        
        return {
            'total_actions': total,
            'successful_actions': successful,
            'failed_actions': perf.get('failed_actions', 0),
            'success_rate': successful / total if total > 0 else 0,
            'uptime': self._calculate_uptime(state)
        }
    
    def _calculate_uptime(self, state: Dict[str, Any]) -> str:
        """Calculate system uptime.
        
        Args:
            state: Current system state
            
        Returns:
            Formatted uptime string
        """
        start_time = state.get('system_performance', {}).get('start_time')
        if not start_time:
            return "Unknown"
        
        try:
            start = datetime.fromisoformat(start_time.replace('Z', '+00:00'))
            now = datetime.now(timezone.utc)
            delta = now - start
            
            days = delta.days
            hours = delta.seconds // 3600
            minutes = (delta.seconds % 3600) // 60
            
            return f"{days}d {hours}h {minutes}m"
        except:
            return "Unknown"
    
    # MCP Integration Methods (all disabled)
    async def query_state(self, query: str) -> Dict[str, Any]:
        """Query the system state using natural language."""
        return {"answer": "MCP is currently disabled"}
    
    async def find_related_projects(self, criteria: str) -> List[Dict[str, Any]]:
        """Find projects matching criteria."""
        return []
    
    async def analyze_state_patterns(self) -> Dict[str, Any]:
        """Analyze patterns in system state."""
        return {"message": "MCP is currently disabled"}
    
    async def get_project_recommendations(self, project_name: str) -> Dict[str, Any]:
        """Get AI-powered recommendations for a specific project."""
        return {"message": "MCP is currently disabled"}
    
    async def predict_future_state(self) -> Dict[str, Any]:
        """Predict future system state based on current trends."""
        return {"message": "MCP is currently disabled"}
    
    # Basic search methods (non-MCP)
    def _basic_project_search(self, criteria: str) -> List[Dict[str, Any]]:
        """Basic project search without MCP."""
        state = self.load_state()
        projects = state.get('projects', {})
        criteria_lower = criteria.lower()
        
        matching_projects = []
        for name, project in projects.items():
            # Search in name, description, language, and topics
            searchable_text = ' '.join([
                name.lower(),
                project.get('description', '').lower(),
                project.get('language', '').lower(),
                ' '.join(project.get('topics', []))
            ])
            
            if criteria_lower in searchable_text:
                matching_projects.append({
                    'name': name,
                    'match_reason': 'Text match in project data',
                    'project': project
                })
        
        return matching_projects
    
    def _extract_recent_patterns(self) -> List[str]:
        """Extract recent patterns from action history."""
        state = self.load_state()
        history = state.get('action_history', [])
        
        if not history:
            return []
        
        # Get last 50 actions
        recent = history[-50:]
        
        # Count action types
        action_counts = {}
        for action in recent:
            action_type = action.get('action_type', 'unknown')
            action_counts[action_type] = action_counts.get(action_type, 0) + 1
        
        # Find patterns
        patterns = []
        for action_type, count in action_counts.items():
            if count > 5:
                patterns.append(f"Frequent {action_type} ({count} times)")
        
        return patterns