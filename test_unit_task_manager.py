#!/usr/bin/env python3
"""
Comprehensive unit tests for TaskManager module.

Tests cover:
- Task creation and lifecycle management
- Task status and priority enums
- GitHub integration
- State persistence
- Task validation and error handling
- External dependency mocking
"""

import unittest
import json
import os
import tempfile
import shutil
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timezone
import sys

# Add scripts directory to path
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'scripts'))

from scripts.task_manager import TaskManager, TaskStatus, TaskPriority, TaskType


class TestTaskManagerEnums(unittest.TestCase):
    """Test TaskManager enumeration classes."""
    
    def test_task_status_enum_values(self):
        """Test TaskStatus enum has correct values."""
        # Assert
        self.assertEqual(TaskStatus.PENDING.value, "pending")
        self.assertEqual(TaskStatus.ASSIGNED.value, "assigned")
        self.assertEqual(TaskStatus.IN_PROGRESS.value, "in_progress")
        self.assertEqual(TaskStatus.IN_REVIEW.value, "in_review")
        self.assertEqual(TaskStatus.COMPLETED.value, "completed")
        self.assertEqual(TaskStatus.BLOCKED.value, "blocked")
        self.assertEqual(TaskStatus.FAILED.value, "failed")
        self.assertEqual(TaskStatus.CANCELLED.value, "cancelled")
    
    def test_task_priority_enum_values(self):
        """Test TaskPriority enum has correct values."""
        # Assert
        self.assertEqual(TaskPriority.CRITICAL.value, "critical")
        self.assertEqual(TaskPriority.HIGH.value, "high")
        self.assertEqual(TaskPriority.MEDIUM.value, "medium")
        self.assertEqual(TaskPriority.LOW.value, "low")
    
    def test_task_type_enum_values(self):
        """Test TaskType enum has correct values."""
        # Assert
        self.assertEqual(TaskType.NEW_PROJECT.value, "new_project")
        self.assertEqual(TaskType.FEATURE.value, "feature")
        self.assertEqual(TaskType.BUG_FIX.value, "bug_fix")
        self.assertEqual(TaskType.REFACTOR.value, "refactor")
        self.assertEqual(TaskType.DOCUMENTATION.value, "documentation")
        self.assertEqual(TaskType.TESTING.value, "testing")
        self.assertEqual(TaskType.SECURITY.value, "security")
        self.assertEqual(TaskType.PERFORMANCE.value, "performance")
        self.assertEqual(TaskType.CODE_REVIEW.value, "code_review")
        self.assertEqual(TaskType.DEPENDENCY_UPDATE.value, "dependency_update")


class TestTaskManagerInitialization(unittest.TestCase):
    """Test TaskManager initialization."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.original_cwd = os.getcwd()
        os.chdir(self.temp_dir)
    
    def tearDown(self):
        """Clean up test fixtures."""
        os.chdir(self.original_cwd)
        shutil.rmtree(self.temp_dir)
    
    @patch.dict(os.environ, {'CLAUDE_PAT': 'test_token'})
    def test_init_with_token_from_environment(self):
        """Test TaskManager initialization with token from environment."""
        # Arrange & Act
        with patch('scripts.task_manager.Github'), \
             patch('scripts.task_manager.StateManager'):
            task_manager = TaskManager()
        
        # Assert
        self.assertEqual(task_manager.github_token, 'test_token')
        self.assertEqual(task_manager.repo_name, 'CodeWebMobile-AI/cwmai')
    
    def test_init_with_explicit_token(self):
        """Test TaskManager initialization with explicit token."""
        # Arrange
        token = 'explicit_token'
        
        # Act
        with patch('scripts.task_manager.Github'), \
             patch('scripts.task_manager.StateManager'):
            task_manager = TaskManager(github_token=token)
        
        # Assert
        self.assertEqual(task_manager.github_token, token)
    
    def test_init_without_token(self):
        """Test TaskManager initialization without token."""
        # Arrange & Act
        with patch('scripts.task_manager.StateManager'):
            task_manager = TaskManager()
        
        # Assert
        self.assertIsNone(task_manager.github_token)
        self.assertIsNone(task_manager.github)
        self.assertIsNone(task_manager.repo)
    
    @patch.dict(os.environ, {'GITHUB_REPOSITORY': 'test/repo'})
    def test_init_with_custom_repo_from_environment(self):
        """Test TaskManager initialization with custom repo from environment."""
        # Arrange & Act
        with patch('scripts.task_manager.StateManager'):
            task_manager = TaskManager()
        
        # Assert
        self.assertEqual(task_manager.repo_name, 'test/repo')


class TestTaskManagerStateOperations(unittest.TestCase):
    """Test TaskManager state loading and saving operations."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.original_cwd = os.getcwd()
        os.chdir(self.temp_dir)
    
    def tearDown(self):
        """Clean up test fixtures."""
        os.chdir(self.original_cwd)
        shutil.rmtree(self.temp_dir)
    
    def test_load_state_with_existing_file(self):
        """Test loading state from existing file."""
        # Arrange
        test_state = {
            "tasks": {"task1": {"title": "Test Task"}},
            "task_counter": 1500,
            "active_tasks": 2
        }
        with open("task_state.json", 'w') as f:
            json.dump(test_state, f)
        
        # Act
        with patch('scripts.task_manager.StateManager'):
            task_manager = TaskManager()
            loaded_state = task_manager._load_state()
        
        # Assert
        self.assertEqual(loaded_state["task_counter"], 1500)
        self.assertEqual(loaded_state["active_tasks"], 2)
        self.assertIn("task1", loaded_state["tasks"])
    
    def test_load_state_with_missing_file(self):
        """Test loading state when file doesn't exist."""
        # Arrange - no file exists
        
        # Act
        with patch('scripts.task_manager.StateManager'):
            task_manager = TaskManager()
            loaded_state = task_manager._load_state()
        
        # Assert - should return default state
        self.assertEqual(loaded_state["task_counter"], 1000)
        self.assertEqual(loaded_state["active_tasks"], 0)
        self.assertEqual(loaded_state["completed_today"], 0)
        self.assertEqual(loaded_state["success_rate"], 0.0)
        self.assertIn("tasks", loaded_state)
        self.assertIn("last_updated", loaded_state)
    
    def test_load_state_with_corrupted_file(self):
        """Test loading state with corrupted JSON file."""
        # Arrange
        with open("task_state.json", 'w') as f:
            f.write("invalid json {")
        
        # Act
        with patch('scripts.task_manager.StateManager'):
            task_manager = TaskManager()
            loaded_state = task_manager._load_state()
        
        # Assert - should return default state
        self.assertEqual(loaded_state["task_counter"], 1000)
        self.assertIn("tasks", loaded_state)
    
    def test_load_history_with_existing_file(self):
        """Test loading history from existing file."""
        # Arrange
        test_history = [
            {"task_id": "TASK-1001", "action": "created", "timestamp": "2025-01-01T00:00:00Z"}
        ]
        with open("task_history.json", 'w') as f:
            json.dump(test_history, f)
        
        # Act
        with patch('scripts.task_manager.StateManager'):
            task_manager = TaskManager()
            loaded_history = task_manager._load_history()
        
        # Assert
        self.assertEqual(len(loaded_history), 1)
        self.assertEqual(loaded_history[0]["task_id"], "TASK-1001")
    
    def test_load_history_with_missing_file(self):
        """Test loading history when file doesn't exist."""
        # Arrange - no file exists
        
        # Act
        with patch('scripts.task_manager.StateManager'):
            task_manager = TaskManager()
            loaded_history = task_manager._load_history()
        
        # Assert
        self.assertEqual(loaded_history, [])
    
    def test_save_state_updates_timestamp(self):
        """Test that _save_state updates the last_updated timestamp."""
        # Arrange
        with patch('scripts.task_manager.StateManager'):
            task_manager = TaskManager()
            old_timestamp = task_manager.state["last_updated"]
        
        # Act
        task_manager._save_state()
        
        # Assert
        self.assertNotEqual(task_manager.state["last_updated"], old_timestamp)
        self.assertTrue(os.path.exists("task_state.json"))
    
    def test_save_history_limits_entries(self):
        """Test that _save_history limits history to 1000 entries."""
        # Arrange
        with patch('scripts.task_manager.StateManager'):
            task_manager = TaskManager()
            # Create history with more than 1000 entries
            task_manager.history = [{"id": i} for i in range(1200)]
        
        # Act
        task_manager._save_history()
        
        # Assert
        self.assertEqual(len(task_manager.history), 1000)
        # Should keep the most recent entries
        self.assertEqual(task_manager.history[0]["id"], 200)
        self.assertEqual(task_manager.history[-1]["id"], 1199)


class TestTaskManagerTaskCreation(unittest.TestCase):
    """Test task creation functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.original_cwd = os.getcwd()
        os.chdir(self.temp_dir)
        
        with patch('scripts.task_manager.StateManager'):
            self.task_manager = TaskManager()
    
    def tearDown(self):
        """Clean up test fixtures."""
        os.chdir(self.original_cwd)
        shutil.rmtree(self.temp_dir)
    
    def test_generate_task_id_increments_counter(self):
        """Test that generate_task_id increments the counter."""
        # Arrange
        initial_counter = self.task_manager.state["task_counter"]
        
        # Act
        task_id1 = self.task_manager.generate_task_id()
        task_id2 = self.task_manager.generate_task_id()
        
        # Assert
        self.assertEqual(task_id1, f"TASK-{initial_counter + 1}")
        self.assertEqual(task_id2, f"TASK-{initial_counter + 2}")
        self.assertEqual(self.task_manager.state["task_counter"], initial_counter + 2)
    
    def test_create_task_with_minimal_parameters(self):
        """Test creating task with minimal required parameters."""
        # Arrange
        task_type = TaskType.FEATURE
        title = "Test Feature"
        description = "A test feature implementation"
        
        # Act
        task = self.task_manager.create_task(task_type, title, description)
        
        # Assert
        self.assertIn("id", task)
        self.assertEqual(task["type"], TaskType.FEATURE.value)
        self.assertEqual(task["title"], title)
        self.assertEqual(task["description"], description)
        self.assertEqual(task["priority"], TaskPriority.MEDIUM.value)
        self.assertEqual(task["status"], TaskStatus.PENDING.value)
        self.assertEqual(task["estimated_hours"], 4.0)
        self.assertIn("created_at", task)
        self.assertEqual(task["dependencies"], [])
        self.assertEqual(task["labels"], [])
    
    def test_create_task_with_all_parameters(self):
        """Test creating task with all parameters specified."""
        # Arrange
        task_type = TaskType.BUG_FIX
        title = "Fix Critical Bug"
        description = "Fix a critical bug in the system"
        priority = TaskPriority.CRITICAL
        dependencies = ["TASK-1001", "TASK-1002"]
        estimated_hours = 8.0
        labels = ["bug", "critical", "security"]
        
        # Act
        task = self.task_manager.create_task(
            task_type=task_type,
            title=title,
            description=description,
            priority=priority,
            dependencies=dependencies,
            estimated_hours=estimated_hours,
            labels=labels
        )
        
        # Assert
        self.assertEqual(task["type"], TaskType.BUG_FIX.value)
        self.assertEqual(task["priority"], TaskPriority.CRITICAL.value)
        self.assertEqual(task["dependencies"], dependencies)
        self.assertEqual(task["estimated_hours"], estimated_hours)
        self.assertEqual(task["labels"], labels)
    
    def test_create_task_generates_unique_ids(self):
        """Test that create_task generates unique task IDs."""
        # Arrange & Act
        task1 = self.task_manager.create_task(TaskType.FEATURE, "Task 1", "Description 1")
        task2 = self.task_manager.create_task(TaskType.BUG_FIX, "Task 2", "Description 2")
        task3 = self.task_manager.create_task(TaskType.TESTING, "Task 3", "Description 3")
        
        # Assert
        self.assertNotEqual(task1["id"], task2["id"])
        self.assertNotEqual(task2["id"], task3["id"])
        self.assertNotEqual(task1["id"], task3["id"])
    
    def test_create_task_timestamp_format(self):
        """Test that create_task generates correct timestamp format."""
        # Arrange & Act
        task = self.task_manager.create_task(TaskType.FEATURE, "Test", "Description")
        
        # Assert
        # Should be able to parse the timestamp
        timestamp = task["created_at"]
        parsed_time = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
        self.assertIsInstance(parsed_time, datetime)


class TestTaskManagerTaskLifecycle(unittest.TestCase):
    """Test task lifecycle management functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.original_cwd = os.getcwd()
        os.chdir(self.temp_dir)
        
        with patch('scripts.task_manager.StateManager'):
            self.task_manager = TaskManager()
    
    def tearDown(self):
        """Clean up test fixtures."""
        os.chdir(self.original_cwd)
        shutil.rmtree(self.temp_dir)
    
    def test_update_task_status_valid_transition(self):
        """Test updating task status with valid transition."""
        # Arrange
        task = self.task_manager.create_task(TaskType.FEATURE, "Test", "Description")
        task_id = task["id"]
        
        # Act
        result = self.task_manager.update_task_status(task_id, TaskStatus.IN_PROGRESS)
        
        # Assert
        self.assertTrue(result)
        updated_task = self.task_manager.get_task(task_id)
        self.assertEqual(updated_task["status"], TaskStatus.IN_PROGRESS.value)
        self.assertIn("updated_at", updated_task)
    
    def test_update_task_status_nonexistent_task(self):
        """Test updating status of non-existent task."""
        # Arrange
        nonexistent_id = "TASK-99999"
        
        # Act
        result = self.task_manager.update_task_status(nonexistent_id, TaskStatus.COMPLETED)
        
        # Assert
        self.assertFalse(result)
    
    def test_get_task_existing(self):
        """Test getting an existing task."""
        # Arrange
        task = self.task_manager.create_task(TaskType.FEATURE, "Test", "Description")
        task_id = task["id"]
        
        # Act
        retrieved_task = self.task_manager.get_task(task_id)
        
        # Assert
        self.assertIsNotNone(retrieved_task)
        self.assertEqual(retrieved_task["id"], task_id)
        self.assertEqual(retrieved_task["title"], "Test")
    
    def test_get_task_nonexistent(self):
        """Test getting a non-existent task."""
        # Arrange
        nonexistent_id = "TASK-99999"
        
        # Act
        retrieved_task = self.task_manager.get_task(nonexistent_id)
        
        # Assert
        self.assertIsNone(retrieved_task)
    
    def test_get_tasks_by_status(self):
        """Test getting tasks filtered by status."""
        # Arrange
        task1 = self.task_manager.create_task(TaskType.FEATURE, "Feature 1", "Description 1")
        task2 = self.task_manager.create_task(TaskType.BUG_FIX, "Bug Fix 1", "Description 2")
        task3 = self.task_manager.create_task(TaskType.TESTING, "Test 1", "Description 3")
        
        # Update one task to IN_PROGRESS
        self.task_manager.update_task_status(task2["id"], TaskStatus.IN_PROGRESS)
        
        # Act
        pending_tasks = self.task_manager.get_tasks_by_status(TaskStatus.PENDING)
        in_progress_tasks = self.task_manager.get_tasks_by_status(TaskStatus.IN_PROGRESS)
        
        # Assert
        self.assertEqual(len(pending_tasks), 2)
        self.assertEqual(len(in_progress_tasks), 1)
        self.assertEqual(in_progress_tasks[0]["id"], task2["id"])
    
    def test_get_tasks_by_priority(self):
        """Test getting tasks filtered by priority."""
        # Arrange
        task1 = self.task_manager.create_task(TaskType.FEATURE, "Feature 1", "Description 1", TaskPriority.HIGH)
        task2 = self.task_manager.create_task(TaskType.BUG_FIX, "Bug Fix 1", "Description 2", TaskPriority.LOW)
        task3 = self.task_manager.create_task(TaskType.TESTING, "Test 1", "Description 3", TaskPriority.HIGH)
        
        # Act
        high_priority_tasks = self.task_manager.get_tasks_by_priority(TaskPriority.HIGH)
        low_priority_tasks = self.task_manager.get_tasks_by_priority(TaskPriority.LOW)
        
        # Assert
        self.assertEqual(len(high_priority_tasks), 2)
        self.assertEqual(len(low_priority_tasks), 1)
        high_task_ids = [task["id"] for task in high_priority_tasks]
        self.assertIn(task1["id"], high_task_ids)
        self.assertIn(task3["id"], high_task_ids)


class TestTaskManagerStatistics(unittest.TestCase):
    """Test task statistics and reporting functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.original_cwd = os.getcwd()
        os.chdir(self.temp_dir)
        
        with patch('scripts.task_manager.StateManager'):
            self.task_manager = TaskManager()
    
    def tearDown(self):
        """Clean up test fixtures."""
        os.chdir(self.original_cwd)
        shutil.rmtree(self.temp_dir)
    
    def test_get_task_statistics(self):
        """Test getting task statistics."""
        # Arrange
        task1 = self.task_manager.create_task(TaskType.FEATURE, "Feature 1", "Description 1", TaskPriority.HIGH)
        task2 = self.task_manager.create_task(TaskType.BUG_FIX, "Bug Fix 1", "Description 2", TaskPriority.LOW)
        task3 = self.task_manager.create_task(TaskType.TESTING, "Test 1", "Description 3", TaskPriority.MEDIUM)
        
        # Update task statuses
        self.task_manager.update_task_status(task2["id"], TaskStatus.COMPLETED)
        self.task_manager.update_task_status(task3["id"], TaskStatus.IN_PROGRESS)
        
        # Act
        stats = self.task_manager.get_task_statistics()
        
        # Assert
        self.assertEqual(stats["total_tasks"], 3)
        self.assertEqual(stats["pending_tasks"], 1)
        self.assertEqual(stats["in_progress_tasks"], 1)
        self.assertEqual(stats["completed_tasks"], 1)
        self.assertIn("by_priority", stats)
        self.assertIn("by_type", stats)
    
    def test_get_task_statistics_empty(self):
        """Test getting task statistics with no tasks."""
        # Act
        stats = self.task_manager.get_task_statistics()
        
        # Assert
        self.assertEqual(stats["total_tasks"], 0)
        self.assertEqual(stats["pending_tasks"], 0)
        self.assertEqual(stats["in_progress_tasks"], 0)
        self.assertEqual(stats["completed_tasks"], 0)


class TestTaskManagerValidation(unittest.TestCase):
    """Test task validation functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.original_cwd = os.getcwd()
        os.chdir(self.temp_dir)
        
        with patch('scripts.task_manager.StateManager'):
            self.task_manager = TaskManager()
    
    def tearDown(self):
        """Clean up test fixtures."""
        os.chdir(self.original_cwd)
        shutil.rmtree(self.temp_dir)
    
    def test_validate_task_dependencies_valid(self):
        """Test validating task dependencies with valid dependencies."""
        # Arrange
        dependency_task = self.task_manager.create_task(TaskType.FEATURE, "Dependency", "Description")
        task_data = {
            "dependencies": [dependency_task["id"]]
        }
        
        # Act
        is_valid, errors = self.task_manager.validate_task_dependencies(task_data)
        
        # Assert
        self.assertTrue(is_valid)
        self.assertEqual(len(errors), 0)
    
    def test_validate_task_dependencies_invalid(self):
        """Test validating task dependencies with invalid dependencies."""
        # Arrange
        task_data = {
            "dependencies": ["TASK-99999"]  # Non-existent task
        }
        
        # Act
        is_valid, errors = self.task_manager.validate_task_dependencies(task_data)
        
        # Assert
        self.assertFalse(is_valid)
        self.assertGreater(len(errors), 0)
        self.assertIn("TASK-99999", errors[0])
    
    def test_validate_task_dependencies_empty(self):
        """Test validating task dependencies with empty dependencies list."""
        # Arrange
        task_data = {
            "dependencies": []
        }
        
        # Act
        is_valid, errors = self.task_manager.validate_task_dependencies(task_data)
        
        # Assert
        self.assertTrue(is_valid)
        self.assertEqual(len(errors), 0)


class TestTaskManagerErrorHandling(unittest.TestCase):
    """Test error handling in TaskManager."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.original_cwd = os.getcwd()
        os.chdir(self.temp_dir)
    
    def tearDown(self):
        """Clean up test fixtures."""
        os.chdir(self.original_cwd)
        shutil.rmtree(self.temp_dir)
    
    def test_create_task_with_invalid_enum_values(self):
        """Test creating task with invalid enum values."""
        # Arrange
        with patch('scripts.task_manager.StateManager'):
            task_manager = TaskManager()
        
        # Act & Assert
        with self.assertRaises(Exception):
            # This should fail due to type checking
            task_manager.create_task("invalid_type", "Title", "Description")
    
    def test_filesystem_operations_with_permission_error(self):
        """Test filesystem operations when permission is denied."""
        # Arrange
        with patch('scripts.task_manager.StateManager'):
            task_manager = TaskManager()
        
        # Mock file operations to raise permission error
        with patch('builtins.open', side_effect=PermissionError("Permission denied")):
            # Act - should handle permission error gracefully
            task_manager._save_state()
            task_manager._save_history()
            
            # Assert - should not raise exception
            # (actual behavior depends on implementation)


if __name__ == '__main__':
    # Run tests with verbose output
    unittest.main(verbosity=2)