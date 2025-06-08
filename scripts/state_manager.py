"""
State Manager Module

Handles system state persistence, loading, and I/O operations for the autonomous AI system.
Provides comprehensive state management with both local and remote GitHub repository persistence.
"""

import json
import os
import base64
from datetime import datetime, timezone
from typing import Dict, Any, Optional
from github import Github


class StateManager:
    """Manages system state persistence and I/O operations."""
    
    DEFAULT_STATE = {
        "charter": {
            "primary_goal": "innovation",
            "secondary_goal": "community_engagement", 
            "constraints": ["stay_within_budget", "maintain_quality"]
        },
        "api_budget": {
            "monthly_limit_usd": 100,
            "monthly_usage_usd": 0,
            "last_reset": datetime.now(timezone.utc).isoformat()
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
        self.github_token = os.getenv('GITHUB_TOKEN')
        self.repo_name = os.getenv('GITHUB_REPOSITORY', 'CodeWebMobile-AI/cwmai')
        
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
                    state = json.load(f)
                    self._validate_and_migrate_state(state)
                    return state
            except (json.JSONDecodeError, KeyError) as e:
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
            
            return json.loads(content)
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
        required_keys = ["charter", "api_budget", "projects", "system_performance", "task_queue"]
        for key in required_keys:
            if key not in state:
                state[key] = self.DEFAULT_STATE[key].copy()
        
        # Ensure charter has required fields
        charter_keys = ["primary_goal", "secondary_goal", "constraints"]
        for key in charter_keys:
            if key not in state["charter"]:
                state["charter"][key] = self.DEFAULT_STATE["charter"][key]
        
        # Ensure api_budget has required fields
        budget_keys = ["monthly_limit_usd", "monthly_usage_usd"]
        for key in budget_keys:
            if key not in state["api_budget"]:
                state["api_budget"][key] = self.DEFAULT_STATE["api_budget"][key]
        
        # Ensure system_performance has required structure
        if "learning_metrics" not in state["system_performance"]:
            state["system_performance"]["learning_metrics"] = self.DEFAULT_STATE["system_performance"]["learning_metrics"].copy()
        
        # Add version if missing
        if "version" not in state:
            state["version"] = "1.0.0"
    
    def update_budget_usage(self, cost_usd: float) -> None:
        """Update API budget usage.
        
        Args:
            cost_usd: Cost to add to monthly usage
        """
        state = self.load_state()
        state["api_budget"]["monthly_usage_usd"] += cost_usd
        self.save_state_locally(state)
    
    def get_budget_remaining(self) -> float:
        """Get remaining budget in USD.
        
        Returns:
            Remaining budget amount
        """
        state = self.load_state()
        return state["api_budget"]["monthly_limit_usd"] - state["api_budget"]["monthly_usage_usd"]
    
    def is_budget_available(self, required_amount: float = 10.0) -> bool:
        """Check if sufficient budget is available.
        
        Args:
            required_amount: Minimum required budget
            
        Returns:
            True if budget is available
        """
        return self.get_budget_remaining() >= required_amount