"""
Unit tests for StateManager module.

Tests state persistence, loading, GitHub operations, and error handling.
Follows AAA pattern (Arrange, Act, Assert) with comprehensive coverage.
"""

import pytest
import json
import os
from unittest.mock import Mock, patch, mock_open, MagicMock
from datetime import datetime, timezone

from state_manager import StateManager


class TestStateManager:
    """Test suite for StateManager class."""

    def test_init_default_values(self):
        """Test StateManager initialization with default values."""
        # Arrange & Act
        manager = StateManager()
        
        # Assert
        assert manager.local_path == "system_state.json"
        assert manager.repo_path == "system_state.json"
        assert manager.repo_name == 'CodeWebMobile-AI/cwmai'
        assert manager.organization == 'CodeWebMobile-AI'

    def test_init_custom_values(self):
        """Test StateManager initialization with custom values."""
        # Arrange
        local_path = "custom_state.json"
        repo_path = "custom/repo/state.json"
        
        # Act
        manager = StateManager(local_path=local_path, repo_path=repo_path)
        
        # Assert
        assert manager.local_path == local_path
        assert manager.repo_path == repo_path

    @patch.dict(os.environ, {'CLAUDE_PAT': 'test_token'})
    def test_init_with_github_token(self):
        """Test StateManager initialization with GitHub token."""
        # Arrange & Act
        manager = StateManager()
        
        # Assert
        assert manager.github_token == 'test_token'

    @patch.dict(os.environ, {'GITHUB_REPOSITORY': 'custom-org/custom-repo'})
    def test_init_with_custom_repo(self):
        """Test StateManager initialization with custom repository."""
        # Arrange & Act
        manager = StateManager()
        
        # Assert
        assert manager.repo_name == 'custom-org/custom-repo'

    def test_default_state_structure(self):
        """Test default state contains all required components."""
        # Arrange & Act
        default_state = StateManager.DEFAULT_STATE
        
        # Assert
        assert "charter" in default_state
        assert "projects" in default_state
        assert "system_performance" in default_state
        assert "task_queue" in default_state
        assert "external_context" in default_state
        assert "version" in default_state
        assert "last_updated" in default_state

    def test_default_state_charter_structure(self):
        """Test default state charter has required fields."""
        # Arrange & Act
        charter = StateManager.DEFAULT_STATE["charter"]
        
        # Assert
        assert "primary_goal" in charter
        assert "secondary_goal" in charter
        assert "constraints" in charter
        assert isinstance(charter["constraints"], list)

    def test_default_state_performance_metrics(self):
        """Test default state performance metrics structure."""
        # Arrange & Act
        performance = StateManager.DEFAULT_STATE["system_performance"]
        
        # Assert
        assert "total_cycles" in performance
        assert "successful_actions" in performance
        assert "failed_actions" in performance
        assert "learning_metrics" in performance
        
        learning_metrics = performance["learning_metrics"]
        assert "decision_accuracy" in learning_metrics
        assert "resource_efficiency" in learning_metrics
        assert "goal_achievement" in learning_metrics

    @patch('state_manager.Github')
    def test_discover_organization_repositories_success(self, mock_github_class):
        """Test successful repository discovery."""
        # Arrange
        manager = StateManager()
        manager.github_token = 'test_token'
        
        mock_github = Mock()
        mock_org = Mock()
        mock_repo = Mock()
        
        mock_repo.name = 'test-repo'
        mock_repo.full_name = 'test-org/test-repo'
        mock_repo.archived = False
        mock_repo.disabled = False
        mock_repo.stargazers_count = 5
        mock_repo.forks_count = 2
        mock_repo.open_issues_count = 1
        mock_repo.get_pulls.return_value = []
        
        mock_org.get_repos.return_value = [mock_repo]
        mock_github.get_organization.return_value = mock_org
        mock_github_class.return_value = mock_github
        
        # Act
        repositories = manager.discover_organization_repositories()
        
        # Assert
        assert len(repositories) == 1
        assert repositories[0]['name'] == 'test-repo'
        assert repositories[0]['full_name'] == 'test-org/test-repo'
        assert repositories[0]['stars'] == 5
        assert repositories[0]['forks'] == 2
        assert repositories[0]['open_issues'] == 1

    def test_discover_organization_repositories_no_token(self):
        """Test repository discovery without GitHub token."""
        # Arrange
        manager = StateManager()
        manager.github_token = None
        
        # Act
        repositories = manager.discover_organization_repositories()
        
        # Assert
        assert repositories == []

    @patch('state_manager.Github')
    def test_discover_organization_repositories_with_archived(self, mock_github_class):
        """Test repository discovery filters out archived repositories."""
        # Arrange
        manager = StateManager()
        manager.github_token = 'test_token'
        
        mock_github = Mock()
        mock_org = Mock()
        
        # Create active and archived repos
        active_repo = Mock()
        active_repo.name = 'active-repo'
        active_repo.archived = False
        active_repo.disabled = False
        active_repo.stargazers_count = 3
        active_repo.forks_count = 1
        active_repo.open_issues_count = 0
        active_repo.get_pulls.return_value = []
        
        archived_repo = Mock()
        archived_repo.archived = True
        archived_repo.disabled = False
        
        mock_org.get_repos.return_value = [active_repo, archived_repo]
        mock_github.get_organization.return_value = mock_org
        mock_github_class.return_value = mock_github
        
        # Act
        repositories = manager.discover_organization_repositories()
        
        # Assert
        assert len(repositories) == 1
        assert repositories[0]['name'] == 'active-repo'

    @patch('state_manager.Github')
    def test_discover_organization_repositories_exception(self, mock_github_class):
        """Test repository discovery handles exceptions gracefully."""
        # Arrange
        manager = StateManager()
        manager.github_token = 'test_token'
        
        mock_github_class.side_effect = Exception("API Error")
        
        # Act
        repositories = manager.discover_organization_repositories()
        
        # Assert
        assert repositories == []

    @patch('builtins.open', new_callable=mock_open, read_data='{"test": "data"}')
    @patch('os.path.exists', return_value=True)
    def test_load_state_local_success(self, mock_exists, mock_file):
        """Test successful local state loading."""
        # Arrange
        manager = StateManager(local_path="test_state.json")
        
        # Act
        state = manager.load_state()
        
        # Assert
        assert state == {"test": "data"}
        mock_file.assert_called_once_with("test_state.json", 'r')

    @patch('os.path.exists', return_value=False)
    def test_load_state_no_local_file(self, mock_exists):
        """Test state loading when no local file exists."""
        # Arrange
        manager = StateManager(local_path="nonexistent.json")
        
        # Act
        state = manager.load_state()
        
        # Assert
        assert state == StateManager.DEFAULT_STATE

    @patch('builtins.open', new_callable=mock_open)
    @patch('os.path.exists', return_value=True)
    def test_load_state_invalid_json(self, mock_exists, mock_file):
        """Test state loading with invalid JSON."""
        # Arrange
        mock_file.return_value.read.return_value = "invalid json"
        manager = StateManager()
        
        # Act
        state = manager.load_state()
        
        # Assert
        assert state == StateManager.DEFAULT_STATE

    @patch('builtins.open', new_callable=mock_open)
    @patch('json.dump')
    def test_save_state_success(self, mock_json_dump, mock_file):
        """Test successful state saving."""
        # Arrange
        manager = StateManager(local_path="test_state.json")
        test_state = {"test": "data"}
        
        # Act
        result = manager.save_state(test_state)
        
        # Assert
        assert result is True
        mock_file.assert_called_once_with("test_state.json", 'w')
        mock_json_dump.assert_called_once()

    @patch('builtins.open', side_effect=PermissionError("Permission denied"))
    def test_save_state_permission_error(self, mock_file):
        """Test state saving with permission error."""
        # Arrange
        manager = StateManager()
        test_state = {"test": "data"}
        
        # Act
        result = manager.save_state(test_state)
        
        # Assert
        assert result is False

    def test_validate_state_structure_valid(self):
        """Test state validation with valid structure."""
        # Arrange
        manager = StateManager()
        valid_state = {
            "charter": {"primary_goal": "test"},
            "projects": {},
            "system_performance": {
                "total_cycles": 0,
                "successful_actions": 0,
                "failed_actions": 0,
                "learning_metrics": {}
            },
            "task_queue": [],
            "version": "1.0.0"
        }
        
        # Act
        is_valid = manager.validate_state_structure(valid_state)
        
        # Assert
        assert is_valid is True

    def test_validate_state_structure_missing_key(self):
        """Test state validation with missing required key."""
        # Arrange
        manager = StateManager()
        invalid_state = {"charter": {"primary_goal": "test"}}  # Missing other keys
        
        # Act
        is_valid = manager.validate_state_structure(invalid_state)
        
        # Assert
        assert is_valid is False

    def test_validate_state_structure_none(self):
        """Test state validation with None input."""
        # Arrange
        manager = StateManager()
        
        # Act
        is_valid = manager.validate_state_structure(None)
        
        # Assert
        assert is_valid is False

    def test_validate_state_structure_empty_dict(self):
        """Test state validation with empty dictionary."""
        # Arrange
        manager = StateManager()
        
        # Act
        is_valid = manager.validate_state_structure({})
        
        # Assert
        assert is_valid is False

    @patch('state_manager.Github')
    def test_github_operations_integration(self, mock_github_class):
        """Test GitHub file operations integration."""
        # Arrange
        manager = StateManager()
        manager.github_token = 'test_token'
        
        mock_github = Mock()
        mock_repo = Mock()
        mock_file = Mock()
        
        mock_file.content = b'{"existing": "data"}'
        mock_file.sha = 'existing_sha'
        mock_repo.get_contents.return_value = mock_file
        mock_repo.update_file.return_value = {'commit': {'sha': 'new_sha'}}
        mock_github.get_repo.return_value = mock_repo
        mock_github_class.return_value = mock_github
        
        test_state = {"new": "data"}
        
        # Act
        result = manager.save_state_to_github(test_state)
        
        # Assert
        assert result is True
        mock_repo.update_file.assert_called_once()

    def test_performance_metrics_calculation(self):
        """Test performance metrics calculation methods."""
        # Arrange
        manager = StateManager()
        test_state = {
            "system_performance": {
                "total_cycles": 100,
                "successful_actions": 80,
                "failed_actions": 20,
                "learning_metrics": {
                    "decision_accuracy": 0.8,
                    "resource_efficiency": 0.75,
                    "goal_achievement": 0.9
                }
            }
        }
        
        # Act
        success_rate = manager.calculate_success_rate(test_state)
        overall_performance = manager.calculate_overall_performance(test_state)
        
        # Assert
        assert success_rate == 0.8  # 80/100
        assert overall_performance > 0.8  # Should be average of learning metrics