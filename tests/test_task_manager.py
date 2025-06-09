"""
Unit tests for TaskManager module.

Tests task creation, management, state tracking, and GitHub integration.
Follows AAA pattern with comprehensive coverage of task lifecycle.
"""

import pytest
import json
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timezone, timedelta

# Import the enums and classes
from task_manager import TaskManager, TaskStatus, TaskPriority, TaskType


class TestTaskEnums:
    """Test task enumeration classes."""

    def test_task_status_values(self):
        """Test TaskStatus enum has all required values."""
        # Arrange & Act
        statuses = [status.value for status in TaskStatus]
        
        # Assert
        expected_statuses = [
            "pending", "assigned", "in_progress", "in_review", 
            "completed", "blocked", "failed", "cancelled"
        ]
        assert all(status in statuses for status in expected_statuses)

    def test_task_priority_values(self):
        """Test TaskPriority enum has all required values."""
        # Arrange & Act
        priorities = [priority.value for priority in TaskPriority]
        
        # Assert
        expected_priorities = ["critical", "high", "medium", "low"]
        assert all(priority in priorities for priority in expected_priorities)

    def test_task_type_values(self):
        """Test TaskType enum has required values."""
        # Arrange & Act
        types = [task_type.value for task_type in TaskType]
        
        # Assert
        expected_types = ["new_project", "feature", "bug_fix", "refactor"]
        assert all(task_type in types for task_type in expected_types)


class TestTaskManager:
    """Test suite for TaskManager class."""

    @patch('task_manager.StateManager')
    @patch('task_manager.IntelligentAIBrain')
    @patch('task_manager.ContextGatherer')
    def test_init_with_defaults(self, mock_context, mock_brain, mock_state):
        """Test TaskManager initialization with default parameters."""
        # Arrange & Act
        with patch.dict('os.environ', {'CLAUDE_PAT': 'test_token'}):
            manager = TaskManager()
        
        # Assert
        assert manager.github_token == 'test_token'
        assert manager.repo_name == 'CodeWebMobile-AI/cwmai'
        mock_state.assert_called_once()
        mock_brain.assert_called_once()
        mock_context.assert_called_once()

    @patch('task_manager.StateManager')
    @patch('task_manager.IntelligentAIBrain')
    @patch('task_manager.ContextGatherer')
    def test_init_with_custom_repo(self, mock_context, mock_brain, mock_state):
        """Test TaskManager initialization with custom repository."""
        # Arrange & Act
        with patch.dict('os.environ', {'GITHUB_REPOSITORY': 'custom/repo'}):
            manager = TaskManager()
        
        # Assert
        assert manager.repo_name == 'custom/repo'

    @patch('task_manager.Github')
    @patch('task_manager.StateManager')
    @patch('task_manager.IntelligentAIBrain')
    @patch('task_manager.ContextGatherer')
    def test_create_task_basic(self, mock_context, mock_brain, mock_state, mock_github_class):
        """Test basic task creation."""
        # Arrange
        manager = TaskManager()
        manager.github_token = 'test_token'
        
        mock_github = Mock()
        mock_repo = Mock()
        mock_issue = Mock()
        
        mock_issue.number = 123
        mock_issue.html_url = 'https://github.com/test/repo/issues/123'
        mock_repo.create_issue.return_value = mock_issue
        mock_github.get_repo.return_value = mock_repo
        mock_github_class.return_value = mock_github
        
        task_data = {
            'title': 'Test Task',
            'description': 'Test Description',
            'type': TaskType.FEATURE,
            'priority': TaskPriority.HIGH
        }
        
        # Act
        result = manager.create_task(task_data)
        
        # Assert
        assert result is not None
        assert result['issue_number'] == 123
        assert result['issue_url'] == 'https://github.com/test/repo/issues/123'

    @patch('task_manager.Github')
    @patch('task_manager.StateManager')
    @patch('task_manager.IntelligentAIBrain')
    @patch('task_manager.ContextGatherer')
    def test_create_task_without_github_token(self, mock_context, mock_brain, mock_state, mock_github_class):
        """Test task creation without GitHub token."""
        # Arrange
        manager = TaskManager()
        manager.github_token = None
        
        task_data = {
            'title': 'Test Task',
            'description': 'Test Description',
            'type': TaskType.FEATURE,
            'priority': TaskPriority.HIGH
        }
        
        # Act
        result = manager.create_task(task_data)
        
        # Assert
        assert result is None

    @patch('task_manager.StateManager')
    @patch('task_manager.IntelligentAIBrain')
    @patch('task_manager.ContextGatherer')
    def test_generate_task_id(self, mock_context, mock_brain, mock_state):
        """Test task ID generation."""
        # Arrange
        manager = TaskManager()
        
        # Act
        task_id_1 = manager._generate_task_id()
        task_id_2 = manager._generate_task_id()
        
        # Assert
        assert task_id_1.startswith('TASK-')
        assert task_id_2.startswith('TASK-')
        assert task_id_1 != task_id_2

    @patch('task_manager.StateManager')
    @patch('task_manager.IntelligentAIBrain')
    @patch('task_manager.ContextGatherer')
    def test_format_task_description(self, mock_context, mock_brain, mock_state):
        """Test task description formatting."""
        # Arrange
        manager = TaskManager()
        task_data = {
            'title': 'Test Task',
            'description': 'Basic description',
            'type': TaskType.FEATURE,
            'priority': TaskPriority.HIGH,
            'requirements': ['Requirement 1', 'Requirement 2'],
            'acceptance_criteria': ['Criteria 1', 'Criteria 2']
        }
        
        # Act
        formatted = manager._format_task_description(task_data)
        
        # Assert
        assert '@claude' in formatted
        assert 'Test Task' in formatted
        assert 'Basic description' in formatted
        assert 'Requirement 1' in formatted
        assert 'Criteria 1' in formatted

    @patch('task_manager.StateManager')
    @patch('task_manager.IntelligentAIBrain')
    @patch('task_manager.ContextGatherer')
    def test_calculate_task_priority_score(self, mock_context, mock_brain, mock_state):
        """Test task priority score calculation."""
        # Arrange
        manager = TaskManager()
        
        # Act
        critical_score = manager._calculate_task_priority_score(TaskPriority.CRITICAL)
        high_score = manager._calculate_task_priority_score(TaskPriority.HIGH)
        medium_score = manager._calculate_task_priority_score(TaskPriority.MEDIUM)
        low_score = manager._calculate_task_priority_score(TaskPriority.LOW)
        
        # Assert
        assert critical_score > high_score > medium_score > low_score
        assert critical_score == 100
        assert low_score == 25

    @patch('task_manager.Github')
    @patch('task_manager.StateManager')
    @patch('task_manager.IntelligentAIBrain')
    @patch('task_manager.ContextGatherer')
    def test_update_task_status(self, mock_context, mock_brain, mock_state, mock_github_class):
        """Test task status updating."""
        # Arrange
        manager = TaskManager()
        manager.github_token = 'test_token'
        
        mock_github = Mock()
        mock_repo = Mock()
        mock_issue = Mock()
        
        mock_repo.get_issue.return_value = mock_issue
        mock_github.get_repo.return_value = mock_repo
        mock_github_class.return_value = mock_github
        
        # Act
        result = manager.update_task_status(123, TaskStatus.IN_PROGRESS)
        
        # Assert
        assert result is True
        mock_issue.edit.assert_called_once()

    @patch('task_manager.Github')
    @patch('task_manager.StateManager')
    @patch('task_manager.IntelligentAIBrain')
    @patch('task_manager.ContextGatherer')
    def test_get_task_details(self, mock_context, mock_brain, mock_state, mock_github_class):
        """Test retrieving task details from GitHub."""
        # Arrange
        manager = TaskManager()
        manager.github_token = 'test_token'
        
        mock_github = Mock()
        mock_repo = Mock()
        mock_issue = Mock()
        
        mock_issue.number = 123
        mock_issue.title = 'Test Issue'
        mock_issue.body = 'Test Description'
        mock_issue.state = 'open'
        mock_issue.created_at = datetime.now(timezone.utc)
        mock_issue.updated_at = datetime.now(timezone.utc)
        mock_issue.labels = []
        
        mock_repo.get_issue.return_value = mock_issue
        mock_github.get_repo.return_value = mock_repo
        mock_github_class.return_value = mock_github
        
        # Act
        details = manager.get_task_details(123)
        
        # Assert
        assert details is not None
        assert details['number'] == 123
        assert details['title'] == 'Test Issue'

    @patch('task_manager.StateManager')
    @patch('task_manager.IntelligentAIBrain')
    @patch('task_manager.ContextGatherer')
    def test_validate_task_data_valid(self, mock_context, mock_brain, mock_state):
        """Test task data validation with valid data."""
        # Arrange
        manager = TaskManager()
        valid_task = {
            'title': 'Valid Task',
            'description': 'Valid description',
            'type': TaskType.FEATURE,
            'priority': TaskPriority.MEDIUM
        }
        
        # Act
        is_valid, errors = manager._validate_task_data(valid_task)
        
        # Assert
        assert is_valid is True
        assert len(errors) == 0

    @patch('task_manager.StateManager')
    @patch('task_manager.IntelligentAIBrain')
    @patch('task_manager.ContextGatherer')
    def test_validate_task_data_invalid(self, mock_context, mock_brain, mock_state):
        """Test task data validation with invalid data."""
        # Arrange
        manager = TaskManager()
        invalid_task = {
            'title': '',  # Empty title
            'description': 'Valid description',
            'type': 'INVALID_TYPE',  # Invalid type
            'priority': TaskPriority.MEDIUM
        }
        
        # Act
        is_valid, errors = manager._validate_task_data(invalid_task)
        
        # Assert
        assert is_valid is False
        assert len(errors) > 0
        assert any('title' in error.lower() for error in errors)

    @patch('task_manager.Github')
    @patch('task_manager.StateManager')
    @patch('task_manager.IntelligentAIBrain')
    @patch('task_manager.ContextGatherer')
    def test_list_open_tasks(self, mock_context, mock_brain, mock_state, mock_github_class):
        """Test listing open tasks from GitHub."""
        # Arrange
        manager = TaskManager()
        manager.github_token = 'test_token'
        
        mock_github = Mock()
        mock_repo = Mock()
        mock_issue = Mock()
        
        mock_issue.number = 123
        mock_issue.title = 'Open Task'
        mock_issue.state = 'open'
        mock_issue.labels = []
        
        mock_repo.get_issues.return_value = [mock_issue]
        mock_github.get_repo.return_value = mock_repo
        mock_github_class.return_value = mock_github
        
        # Act
        tasks = manager.list_open_tasks()
        
        # Assert
        assert len(tasks) == 1
        assert tasks[0]['number'] == 123
        assert tasks[0]['title'] == 'Open Task'

    @patch('task_manager.StateManager')
    @patch('task_manager.IntelligentAIBrain')
    @patch('task_manager.ContextGatherer')
    def test_estimate_task_complexity(self, mock_context, mock_brain, mock_state):
        """Test task complexity estimation."""
        # Arrange
        manager = TaskManager()
        
        simple_task = {
            'type': TaskType.BUG_FIX,
            'description': 'Fix typo in documentation'
        }
        
        complex_task = {
            'type': TaskType.NEW_PROJECT,
            'description': 'Build a complete e-commerce platform with payment integration'
        }
        
        # Act
        simple_complexity = manager._estimate_task_complexity(simple_task)
        complex_complexity = manager._estimate_task_complexity(complex_task)
        
        # Assert
        assert complex_complexity > simple_complexity
        assert simple_complexity >= 1
        assert complex_complexity <= 10

    @patch('task_manager.StateManager')
    @patch('task_manager.IntelligentAIBrain')
    @patch('task_manager.ContextGatherer')
    def test_task_age_calculation(self, mock_context, mock_brain, mock_state):
        """Test task age calculation."""
        # Arrange
        manager = TaskManager()
        old_date = datetime.now(timezone.utc) - timedelta(days=5)
        recent_date = datetime.now(timezone.utc) - timedelta(hours=2)
        
        # Act
        old_age = manager._calculate_task_age(old_date)
        recent_age = manager._calculate_task_age(recent_date)
        
        # Assert
        assert old_age > recent_age
        assert old_age == 5.0  # 5 days
        assert recent_age < 1.0  # Less than 1 day

    @patch('task_manager.Github')
    @patch('task_manager.StateManager')
    @patch('task_manager.IntelligentAIBrain')
    @patch('task_manager.ContextGatherer')
    def test_error_handling_github_api_failure(self, mock_context, mock_brain, mock_state, mock_github_class):
        """Test error handling when GitHub API fails."""
        # Arrange
        manager = TaskManager()
        manager.github_token = 'test_token'
        
        mock_github_class.side_effect = Exception("GitHub API Error")
        
        task_data = {
            'title': 'Test Task',
            'description': 'Test Description',
            'type': TaskType.FEATURE,
            'priority': TaskPriority.HIGH
        }
        
        # Act
        result = manager.create_task(task_data)
        
        # Assert
        assert result is None

    @patch('task_manager.StateManager')
    @patch('task_manager.IntelligentAIBrain')
    @patch('task_manager.ContextGatherer')
    def test_task_dependencies_handling(self, mock_context, mock_brain, mock_state):
        """Test task dependencies validation and handling."""
        # Arrange
        manager = TaskManager()
        
        task_with_deps = {
            'title': 'Dependent Task',
            'description': 'Task with dependencies',
            'type': TaskType.FEATURE,
            'priority': TaskPriority.MEDIUM,
            'dependencies': ['TASK-001', 'TASK-002']
        }
        
        # Act
        has_deps = manager._has_dependencies(task_with_deps)
        deps = manager._get_task_dependencies(task_with_deps)
        
        # Assert
        assert has_deps is True
        assert len(deps) == 2
        assert 'TASK-001' in deps
        assert 'TASK-002' in deps

    @patch('task_manager.StateManager')
    @patch('task_manager.IntelligentAIBrain')
    @patch('task_manager.ContextGatherer')
    def test_task_metrics_tracking(self, mock_context, mock_brain, mock_state):
        """Test task metrics calculation and tracking."""
        # Arrange
        manager = TaskManager()
        
        # Mock some task history
        manager.task_history = [
            {'status': TaskStatus.COMPLETED, 'created_at': datetime.now(timezone.utc) - timedelta(days=1)},
            {'status': TaskStatus.COMPLETED, 'created_at': datetime.now(timezone.utc) - timedelta(days=2)},
            {'status': TaskStatus.FAILED, 'created_at': datetime.now(timezone.utc) - timedelta(days=3)}
        ]
        
        # Act
        success_rate = manager.calculate_success_rate()
        avg_completion_time = manager.calculate_average_completion_time()
        
        # Assert
        assert success_rate > 0.0
        assert success_rate <= 1.0
        assert avg_completion_time >= 0.0