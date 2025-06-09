#!/usr/bin/env python3
"""
Comprehensive unit tests for StateManager module.

Tests cover:
- State loading and saving operations
- Repository discovery functionality  
- Health score calculations
- Error handling and edge cases
- External dependency mocking
"""

import unittest
import json
import os
import tempfile
import shutil
from unittest.mock import Mock, patch, MagicMock, mock_open
from datetime import datetime, timezone
import sys

# Add scripts directory to path
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'scripts'))

from scripts.state_manager import StateManager


class TestStateManagerInitialization(unittest.TestCase):
    """Test StateManager initialization and basic properties."""
    
    def test_init_with_default_parameters(self):
        """Test StateManager initialization with default parameters."""
        # Arrange & Act
        state_manager = StateManager()
        
        # Assert
        self.assertEqual(state_manager.local_path, "system_state.json")
        self.assertEqual(state_manager.repo_path, "system_state.json")
        self.assertEqual(state_manager.repo_name, 'CodeWebMobile-AI/cwmai')
        self.assertEqual(state_manager.organization, 'CodeWebMobile-AI')
    
    def test_init_with_custom_parameters(self):
        """Test StateManager initialization with custom parameters."""
        # Arrange
        local_path = "custom_local.json"
        repo_path = "custom_repo.json"
        
        # Act
        state_manager = StateManager(local_path=local_path, repo_path=repo_path)
        
        # Assert
        self.assertEqual(state_manager.local_path, local_path)
        self.assertEqual(state_manager.repo_path, repo_path)
    
    @patch.dict(os.environ, {'CLAUDE_PAT': 'test_token', 'GITHUB_REPOSITORY': 'test/repo'})
    def test_init_with_environment_variables(self):
        """Test StateManager initialization with environment variables."""
        # Arrange & Act
        state_manager = StateManager()
        
        # Assert
        self.assertEqual(state_manager.github_token, 'test_token')
        self.assertEqual(state_manager.repo_name, 'test/repo')


class TestStateManagerDefaultState(unittest.TestCase):
    """Test StateManager default state structure."""
    
    def test_default_state_structure(self):
        """Test that default state has correct structure."""
        # Arrange
        default_state = StateManager.DEFAULT_STATE
        
        # Assert - Check top-level keys
        self.assertIn('charter', default_state)
        self.assertIn('projects', default_state)
        self.assertIn('system_performance', default_state)
        self.assertIn('task_queue', default_state)
        self.assertIn('external_context', default_state)
        self.assertIn('version', default_state)
        self.assertIn('last_updated', default_state)
    
    def test_default_state_charter_structure(self):
        """Test charter section of default state."""
        # Arrange
        charter = StateManager.DEFAULT_STATE['charter']
        
        # Assert
        self.assertEqual(charter['primary_goal'], 'innovation')
        self.assertEqual(charter['secondary_goal'], 'community_engagement')
        self.assertIsInstance(charter['constraints'], list)
        self.assertIn('maintain_quality', charter['constraints'])
        self.assertIn('ensure_security', charter['constraints'])
    
    def test_default_state_projects_structure(self):
        """Test projects section of default state."""
        # Arrange
        projects = StateManager.DEFAULT_STATE['projects']
        
        # Assert
        self.assertIn('sample-project', projects)
        sample_project = projects['sample-project']
        self.assertIn('health_score', sample_project)
        self.assertIn('last_checked', sample_project)
        self.assertIn('action_history', sample_project)
        self.assertIn('metrics', sample_project)
        self.assertEqual(sample_project['health_score'], 85)
    
    def test_default_state_system_performance_structure(self):
        """Test system performance section of default state."""
        # Arrange
        performance = StateManager.DEFAULT_STATE['system_performance']
        
        # Assert
        self.assertEqual(performance['total_cycles'], 0)
        self.assertEqual(performance['successful_actions'], 0)
        self.assertEqual(performance['failed_actions'], 0)
        self.assertIn('learning_metrics', performance)
        
        # Check learning metrics
        learning_metrics = performance['learning_metrics']
        self.assertEqual(learning_metrics['decision_accuracy'], 0.0)


class TestStateManagerRepositoryDiscovery(unittest.TestCase):
    """Test repository discovery functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.state_manager = StateManager()
    
    def test_discover_repositories_without_token(self):
        """Test repository discovery without GitHub token."""
        # Arrange
        self.state_manager.github_token = None
        
        # Act
        repositories = self.state_manager.discover_organization_repositories()
        
        # Assert
        self.assertEqual(repositories, [])
    
    @patch('scripts.state_manager.Github')
    def test_discover_repositories_with_token_success(self, mock_github):
        """Test successful repository discovery."""
        # Arrange
        self.state_manager.github_token = 'test_token'
        
        # Mock repository object
        mock_repo = Mock()
        mock_repo.name = 'test-repo'
        mock_repo.full_name = 'CodeWebMobile-AI/test-repo'
        mock_repo.description = 'Test repository'
        mock_repo.html_url = 'https://github.com/CodeWebMobile-AI/test-repo'
        mock_repo.clone_url = 'https://github.com/CodeWebMobile-AI/test-repo.git'
        mock_repo.language = 'Python'
        mock_repo.private = False
        mock_repo.archived = False
        mock_repo.disabled = False
        mock_repo.created_at = datetime.now(timezone.utc)
        mock_repo.updated_at = datetime.now(timezone.utc)
        mock_repo.pushed_at = datetime.now(timezone.utc)
        mock_repo.size = 1024
        mock_repo.stargazers_count = 5
        mock_repo.forks_count = 2
        mock_repo.watchers_count = 3
        mock_repo.open_issues_count = 1
        mock_repo.has_issues = True
        mock_repo.has_projects = True
        mock_repo.has_wiki = False
        mock_repo.default_branch = 'main'
        mock_repo.get_topics.return_value = ['python', 'testing']
        
        # Mock GitHub organization
        mock_org = Mock()
        mock_org.get_repos.return_value = [mock_repo]
        
        # Mock GitHub instance
        mock_github_instance = Mock()
        mock_github_instance.get_organization.return_value = mock_org
        mock_github.return_value = mock_github_instance
        
        # Mock health score calculation
        with patch.object(self.state_manager, '_calculate_repository_health_score', return_value=85.0):
            with patch.object(self.state_manager, '_get_repository_activity_summary', return_value={}):
                # Act
                repositories = self.state_manager.discover_organization_repositories()
        
        # Assert
        self.assertEqual(len(repositories), 1)
        repo_data = repositories[0]
        self.assertEqual(repo_data['name'], 'test-repo')
        self.assertEqual(repo_data['language'], 'Python')
        self.assertEqual(repo_data['metrics']['stars'], 5)
        self.assertEqual(repo_data['health_score'], 85.0)
        self.assertIn('discovered_at', repo_data)
    
    @patch('scripts.state_manager.Github')
    def test_discover_repositories_filters_archived(self, mock_github):
        """Test that archived repositories are filtered out."""
        # Arrange
        self.state_manager.github_token = 'test_token'
        
        # Mock archived repository
        mock_repo = Mock()
        mock_repo.archived = True
        mock_repo.disabled = False
        
        # Mock organization
        mock_org = Mock()
        mock_org.get_repos.return_value = [mock_repo]
        
        # Mock GitHub instance  
        mock_github_instance = Mock()
        mock_github_instance.get_organization.return_value = mock_org
        mock_github.return_value = mock_github_instance
        
        # Act
        repositories = self.state_manager.discover_organization_repositories()
        
        # Assert
        self.assertEqual(len(repositories), 0)
    
    @patch('scripts.state_manager.Github')
    def test_discover_repositories_handles_github_exception(self, mock_github):
        """Test handling of GitHub API exceptions."""
        # Arrange
        self.state_manager.github_token = 'test_token'
        mock_github.side_effect = Exception("GitHub API error")
        
        # Act
        repositories = self.state_manager.discover_organization_repositories()
        
        # Assert
        self.assertEqual(repositories, [])


class TestStateManagerHealthScore(unittest.TestCase):
    """Test repository health score calculation."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.state_manager = StateManager()
    
    def test_calculate_repository_health_score_with_valid_repo(self):
        """Test health score calculation with valid repository metrics."""
        # Arrange
        mock_repo = Mock()
        mock_repo.stargazers_count = 10
        mock_repo.forks_count = 5
        mock_repo.open_issues_count = 2
        mock_repo.updated_at = datetime.now(timezone.utc)
        
        # Act
        health_score = self.state_manager._calculate_repository_health_score(mock_repo)
        
        # Assert
        self.assertIsInstance(health_score, float)
        self.assertGreaterEqual(health_score, 0.0)
        self.assertLessEqual(health_score, 100.0)
    
    def test_calculate_repository_health_score_with_zero_metrics(self):
        """Test health score calculation with zero metrics."""
        # Arrange
        mock_repo = Mock()
        mock_repo.stargazers_count = 0
        mock_repo.forks_count = 0
        mock_repo.open_issues_count = 0
        mock_repo.updated_at = datetime.now(timezone.utc)
        
        # Act
        health_score = self.state_manager._calculate_repository_health_score(mock_repo)
        
        # Assert
        self.assertIsInstance(health_score, float)
        self.assertGreaterEqual(health_score, 0.0)


class TestStateManagerFileOperations(unittest.TestCase):
    """Test file operations for state persistence."""
    
    def setUp(self):
        """Set up test fixtures with temporary directory."""
        self.temp_dir = tempfile.mkdtemp()
        self.test_file = os.path.join(self.temp_dir, "test_state.json")
        self.state_manager = StateManager(local_path=self.test_file)
    
    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir)
    
    def test_load_state_with_existing_file(self):
        """Test loading state from existing file."""
        # Arrange
        test_state = {"test_key": "test_value", "version": "1.0.0"}
        with open(self.test_file, 'w') as f:
            json.dump(test_state, f)
        
        # Act
        loaded_state = self.state_manager.load_state()
        
        # Assert
        self.assertEqual(loaded_state["test_key"], "test_value")
        self.assertEqual(loaded_state["version"], "1.0.0")
    
    def test_load_state_with_missing_file(self):
        """Test loading state when file doesn't exist."""
        # Arrange - file doesn't exist
        
        # Act
        loaded_state = self.state_manager.load_state()
        
        # Assert - should return default state
        self.assertEqual(loaded_state["version"], "1.0.0")
        self.assertIn("charter", loaded_state)
        self.assertIn("projects", loaded_state)
    
    def test_load_state_with_corrupted_file(self):
        """Test loading state with corrupted JSON file."""
        # Arrange
        with open(self.test_file, 'w') as f:
            f.write("invalid json content {")
        
        # Act
        loaded_state = self.state_manager.load_state()
        
        # Assert - should return default state
        self.assertEqual(loaded_state["version"], "1.0.0")
        self.assertIn("charter", loaded_state)
    
    def test_save_state_creates_file(self):
        """Test that save_state creates file with correct content."""
        # Arrange
        test_state = {"test_key": "test_value", "version": "2.0.0"}
        
        # Act
        self.state_manager.save_state(test_state)
        
        # Assert
        self.assertTrue(os.path.exists(self.test_file))
        with open(self.test_file, 'r') as f:
            saved_state = json.load(f)
        self.assertEqual(saved_state["test_key"], "test_value")
        self.assertEqual(saved_state["version"], "2.0.0")
        self.assertIn("last_updated", saved_state)
    
    def test_save_state_updates_timestamp(self):
        """Test that save_state updates the last_updated timestamp."""
        # Arrange
        test_state = {"test_key": "test_value"}
        old_time = "2023-01-01T00:00:00+00:00"
        test_state["last_updated"] = old_time
        
        # Act
        self.state_manager.save_state(test_state)
        
        # Assert
        with open(self.test_file, 'r') as f:
            saved_state = json.load(f)
        self.assertNotEqual(saved_state["last_updated"], old_time)


class TestStateManagerEdgeCases(unittest.TestCase):
    """Test edge cases and error conditions."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.state_manager = StateManager()
    
    def test_github_token_from_environment_fallback(self):
        """Test GitHub token fallback from environment variables."""
        # Arrange & Act
        with patch.dict(os.environ, {'GITHUB_TOKEN': 'fallback_token'}, clear=True):
            state_manager = StateManager()
        
        # Assert
        self.assertEqual(state_manager.github_token, 'fallback_token')
    
    def test_github_token_priority_claude_pat_over_github_token(self):
        """Test that CLAUDE_PAT takes priority over GITHUB_TOKEN."""
        # Arrange & Act
        with patch.dict(os.environ, {'CLAUDE_PAT': 'claude_token', 'GITHUB_TOKEN': 'github_token'}):
            state_manager = StateManager()
        
        # Assert
        self.assertEqual(state_manager.github_token, 'claude_token')
    
    @patch('scripts.state_manager.Github')
    def test_discover_repositories_handles_repo_processing_error(self, mock_github):
        """Test handling of errors during individual repository processing."""
        # Arrange
        self.state_manager.github_token = 'test_token'
        
        # Mock repository that raises exception during processing
        mock_repo = Mock()
        mock_repo.name = 'problem-repo'
        mock_repo.archived = False
        mock_repo.disabled = False
        # Make accessing properties raise an exception
        mock_repo.full_name = Mock(side_effect=Exception("API error"))
        
        # Mock organization
        mock_org = Mock()
        mock_org.get_repos.return_value = [mock_repo]
        
        # Mock GitHub instance
        mock_github_instance = Mock()
        mock_github_instance.get_organization.return_value = mock_org
        mock_github.return_value = mock_github_instance
        
        # Act
        repositories = self.state_manager.discover_organization_repositories()
        
        # Assert - should return empty list but not crash
        self.assertEqual(repositories, [])


if __name__ == '__main__':
    # Run tests with verbose output
    unittest.main(verbosity=2)