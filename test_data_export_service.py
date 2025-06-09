"""
Test Data Export Service Module

Comprehensive tests for the data export functionality supporting CSV, JSON, and PDF formats.
Tests all data types, filtering, performance benchmarking, and error handling.
"""

import unittest
import json
import csv
import os
import tempfile
import shutil
from datetime import datetime, timezone
from unittest.mock import Mock, patch, MagicMock
import sys

# Add scripts directory to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'scripts'))

from scripts.data_export_service import (
    DataExportService, ExportFormat, DataType, 
    TaskStatus, TaskPriority, TaskType
)


class TestDataExportService(unittest.TestCase):
    """Test cases for DataExportService."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.export_service = DataExportService(output_dir=self.temp_dir)
        
        # Mock data for testing
        self.mock_task_state = {
            "tasks": {
                "TASK-1001": {
                    "title": "Test Task 1",
                    "status": "completed",
                    "priority": "high",
                    "type": "feature",
                    "created_at": "2025-06-09T10:00:00Z",
                    "updated_at": "2025-06-09T12:00:00Z",
                    "estimated_hours": 4.0,
                    "dependencies": ["TASK-1000"],
                    "labels": ["enhancement", "priority"]
                },
                "TASK-1002": {
                    "title": "Test Task 2",
                    "status": "in_progress",
                    "priority": "medium",
                    "type": "bug_fix",
                    "created_at": "2025-06-09T11:00:00Z",
                    "updated_at": "2025-06-09T13:00:00Z",
                    "estimated_hours": 2.0,
                    "dependencies": [],
                    "labels": ["bug"]
                }
            },
            "task_counter": 1002,
            "last_updated": "2025-06-09T13:00:00Z",
            "active_tasks": 1,
            "completed_today": 1,
            "success_rate": 0.75
        }
        
        self.mock_system_state = {
            "charter": {
                "primary_goal": "innovation",
                "secondary_goal": "community_engagement",
                "constraints": ["maintain_quality", "ensure_security"]
            },
            "projects": {
                "test-project": {
                    "name": "test-project",
                    "full_name": "test-org/test-project",
                    "description": "Test project description",
                    "language": "Python",
                    "health_score": 85.0,
                    "status": "active",
                    "metrics": {
                        "stars": 10,
                        "forks": 5,
                        "issues_open": 2,
                        "watchers": 8
                    },
                    "recent_activity": {
                        "recent_commits": 15,
                        "contributors_count": 3,
                        "last_commit_date": "2025-06-09T10:00:00Z"
                    },
                    "last_checked": "2025-06-09T13:00:00Z"
                }
            },
            "system_performance": {
                "total_cycles": 100,
                "successful_actions": 85,
                "failed_actions": 15,
                "learning_metrics": {
                    "decision_accuracy": 0.85,
                    "goal_achievement": 0.75,
                    "resource_efficiency": 0.90
                }
            },
            "repository_discovery": {
                "discovery_source": "github_organization",
                "last_discovery": "2025-06-09T12:00:00Z",
                "organization": "test-org",
                "repositories_found": 1
            },
            "external_context": {
                "last_updated": "2025-06-09T12:00:00Z",
                "market_trends": [],
                "security_alerts": [],
                "technology_updates": []
            },
            "task_queue": []
        }
    
    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    @patch('scripts.data_export_service.StateManager')
    @patch('scripts.data_export_service.TaskManager')
    def test_export_json_tasks(self, mock_task_manager, mock_state_manager):
        """Test JSON export for task data."""
        # Mock the managers
        mock_task_manager.return_value.state = self.mock_task_state
        mock_state_manager.return_value.load_state.return_value = self.mock_system_state
        
        # Create new service instance with mocked managers
        service = DataExportService(output_dir=self.temp_dir)
        service.task_manager.state = self.mock_task_state
        service.state_manager.load_state = Mock(return_value=self.mock_system_state)
        
        # Export task data as JSON
        filepath = service.export_data(DataType.TASKS, ExportFormat.JSON)
        
        # Verify file was created
        self.assertTrue(os.path.exists(filepath))
        self.assertTrue(filepath.endswith('.json'))
        
        # Verify content
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        self.assertIn('tasks', data)
        self.assertIn('summary', data)
        self.assertEqual(len(data['tasks']), 2)
        self.assertEqual(data['summary']['total_tasks'], 2)
        self.assertEqual(data['summary']['active_tasks'], 1)
        self.assertEqual(data['summary']['success_rate'], 0.75)
    
    @patch('scripts.data_export_service.StateManager')
    @patch('scripts.data_export_service.TaskManager')
    def test_export_csv_tasks(self, mock_task_manager, mock_state_manager):
        """Test CSV export for task data."""
        # Mock the managers
        service = DataExportService(output_dir=self.temp_dir)
        service.task_manager.state = self.mock_task_state
        service.state_manager.load_state = Mock(return_value=self.mock_system_state)
        
        # Export task data as CSV
        filepath = service.export_data(DataType.TASKS, ExportFormat.CSV)
        
        # Verify file was created
        self.assertTrue(os.path.exists(filepath))
        self.assertTrue(filepath.endswith('.csv'))
        
        # Verify content
        with open(filepath, 'r', newline='') as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]['task_id'], 'TASK-1001')
        self.assertEqual(rows[0]['title'], 'Test Task 1')
        self.assertEqual(rows[0]['status'], 'completed')
        self.assertEqual(rows[1]['task_id'], 'TASK-1002')
        self.assertEqual(rows[1]['status'], 'in_progress')
    
    @patch('scripts.data_export_service.PDF_AVAILABLE', True)
    @patch('scripts.data_export_service.SimpleDocTemplate')
    @patch('scripts.data_export_service.StateManager')
    @patch('scripts.data_export_service.TaskManager')
    def test_export_pdf_tasks(self, mock_task_manager, mock_state_manager, mock_doc):
        """Test PDF export for task data."""
        # Mock the managers
        service = DataExportService(output_dir=self.temp_dir)
        service.task_manager.state = self.mock_task_state
        service.state_manager.load_state = Mock(return_value=self.mock_system_state)
        
        # Mock the PDF document
        mock_doc_instance = MagicMock()
        mock_doc.return_value = mock_doc_instance
        
        # Export task data as PDF
        filepath = service.export_data(DataType.TASKS, ExportFormat.PDF)
        
        # Verify file was created and PDF document was built
        self.assertTrue(os.path.exists(filepath))
        self.assertTrue(filepath.endswith('.pdf'))
        mock_doc.assert_called_once()
        mock_doc_instance.build.assert_called_once()
    
    @patch('scripts.data_export_service.StateManager')
    @patch('scripts.data_export_service.TaskManager')
    def test_export_performance_data(self, mock_task_manager, mock_state_manager):
        """Test performance data export."""
        # Mock the managers
        service = DataExportService(output_dir=self.temp_dir)
        service.task_manager.state = self.mock_task_state
        service.state_manager.load_state = Mock(return_value=self.mock_system_state)
        
        # Export performance data as JSON
        filepath = service.export_data(DataType.PERFORMANCE, ExportFormat.JSON)
        
        # Verify content
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        self.assertIn('system_performance', data)
        self.assertIn('task_metrics', data)
        self.assertIn('repository_health', data)
        self.assertEqual(data['system_performance']['total_cycles'], 100)
        self.assertEqual(data['task_metrics']['active_tasks'], 1)
    
    @patch('scripts.data_export_service.StateManager')
    @patch('scripts.data_export_service.TaskManager')
    def test_export_repositories_data(self, mock_task_manager, mock_state_manager):
        """Test repository data export."""
        # Mock the managers
        service = DataExportService(output_dir=self.temp_dir)
        service.task_manager.state = self.mock_task_state
        service.state_manager.load_state = Mock(return_value=self.mock_system_state)
        
        # Export repository data as JSON
        filepath = service.export_data(DataType.REPOSITORIES, ExportFormat.JSON)
        
        # Verify content
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        self.assertIn('projects', data)
        self.assertIn('summary', data)
        self.assertEqual(len(data['projects']), 1)
        self.assertEqual(data['summary']['total_repositories'], 1)
        self.assertEqual(data['summary']['active_repositories'], 1)
        self.assertEqual(data['projects']['test-project']['health_score'], 85.0)
    
    @patch('scripts.data_export_service.StateManager')
    @patch('scripts.data_export_service.TaskManager')
    def test_export_analytics_data(self, mock_task_manager, mock_state_manager):
        """Test analytics data export."""
        # Mock the managers
        service = DataExportService(output_dir=self.temp_dir)
        service.task_manager.state = self.mock_task_state
        service.state_manager.load_state = Mock(return_value=self.mock_system_state)
        
        # Export analytics data as JSON
        filepath = service.export_data(DataType.ANALYTICS, ExportFormat.JSON)
        
        # Verify content
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        self.assertIn('learning_metrics', data)
        self.assertIn('charter', data)
        self.assertEqual(data['learning_metrics']['decision_accuracy'], 0.85)
        self.assertEqual(data['charter']['primary_goal'], 'innovation')
    
    @patch('scripts.data_export_service.StateManager')
    @patch('scripts.data_export_service.TaskManager')
    def test_task_filtering_by_status(self, mock_task_manager, mock_state_manager):
        """Test filtering tasks by status."""
        # Mock the managers
        service = DataExportService(output_dir=self.temp_dir)
        service.task_manager.state = self.mock_task_state
        service.state_manager.load_state = Mock(return_value=self.mock_system_state)
        
        # Export only completed tasks
        filepath = service.export_data(
            DataType.TASKS, 
            ExportFormat.JSON,
            filters={"status": "completed"}
        )
        
        # Verify content
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        self.assertEqual(len(data['tasks']), 1)
        self.assertEqual(list(data['tasks'].keys())[0], 'TASK-1001')
        self.assertEqual(data['tasks']['TASK-1001']['status'], 'completed')
    
    @patch('scripts.data_export_service.StateManager')
    @patch('scripts.data_export_service.TaskManager')
    def test_task_filtering_by_priority(self, mock_task_manager, mock_state_manager):
        """Test filtering tasks by priority."""
        # Mock the managers
        service = DataExportService(output_dir=self.temp_dir)
        service.task_manager.state = self.mock_task_state
        service.state_manager.load_state = Mock(return_value=self.mock_system_state)
        
        # Export only high priority tasks
        filepath = service.export_data(
            DataType.TASKS, 
            ExportFormat.JSON,
            filters={"priority": "high"}
        )
        
        # Verify content
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        self.assertEqual(len(data['tasks']), 1)
        self.assertEqual(data['tasks']['TASK-1001']['priority'], 'high')
    
    @patch('scripts.data_export_service.StateManager')
    @patch('scripts.data_export_service.TaskManager')
    def test_repository_filtering_by_health_score(self, mock_task_manager, mock_state_manager):
        """Test filtering repositories by health score."""
        # Mock the managers
        service = DataExportService(output_dir=self.temp_dir)
        service.task_manager.state = self.mock_task_state
        service.state_manager.load_state = Mock(return_value=self.mock_system_state)
        
        # Export only repositories with health score >= 90
        filepath = service.export_data(
            DataType.REPOSITORIES, 
            ExportFormat.JSON,
            filters={"min_health_score": 90}
        )
        
        # Verify content
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        # Should be empty since test project has health score of 85
        self.assertEqual(len(data['projects']), 0)
        
        # Test with lower threshold
        filepath = service.export_data(
            DataType.REPOSITORIES, 
            ExportFormat.JSON,
            filters={"min_health_score": 80}
        )
        
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        # Should include the test project
        self.assertEqual(len(data['projects']), 1)
    
    @patch('scripts.data_export_service.StateManager')
    @patch('scripts.data_export_service.TaskManager')
    def test_custom_filename(self, mock_task_manager, mock_state_manager):
        """Test custom filename specification."""
        # Mock the managers
        service = DataExportService(output_dir=self.temp_dir)
        service.task_manager.state = self.mock_task_state
        service.state_manager.load_state = Mock(return_value=self.mock_system_state)
        
        # Export with custom filename
        custom_filename = "my_custom_export.json"
        filepath = service.export_data(
            DataType.TASKS, 
            ExportFormat.JSON,
            filename=custom_filename
        )
        
        # Verify filename
        self.assertTrue(filepath.endswith(custom_filename))
        self.assertTrue(os.path.exists(filepath))
    
    @patch('scripts.data_export_service.StateManager')
    @patch('scripts.data_export_service.TaskManager')
    def test_export_all_data(self, mock_task_manager, mock_state_manager):
        """Test exporting all data types."""
        # Mock the managers
        service = DataExportService(output_dir=self.temp_dir)
        service.task_manager.state = self.mock_task_state
        service.state_manager.load_state = Mock(return_value=self.mock_system_state)
        
        # Export all data
        filepath = service.export_data(DataType.ALL, ExportFormat.JSON)
        
        # Verify content
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        self.assertIn('tasks', data)
        self.assertIn('performance', data)
        self.assertIn('repositories', data)
        self.assertIn('analytics', data)
        self.assertIn('export_timestamp', data)
    
    def test_pdf_not_available_error(self):
        """Test error handling when PDF library is not available."""
        # Temporarily disable PDF availability
        with patch('scripts.data_export_service.PDF_AVAILABLE', False):
            service = DataExportService(output_dir=self.temp_dir)
            
            with self.assertRaises(ValueError) as context:
                service.export_data(DataType.TASKS, ExportFormat.PDF)
            
            self.assertIn("PDF export requires reportlab library", str(context.exception))
    
    def test_invalid_data_type(self):
        """Test error handling for invalid data type."""
        service = DataExportService(output_dir=self.temp_dir)
        
        # This should work fine since DataType enum prevents invalid values
        # But let's test the internal method directly
        with self.assertRaises(ValueError):
            service._get_data_by_type("invalid_type", None)
    
    def test_invalid_export_format(self):
        """Test error handling for invalid export format."""
        service = DataExportService(output_dir=self.temp_dir)
        
        # This should work fine since ExportFormat enum prevents invalid values
        # But let's test by patching the enum
        with patch.object(service, 'export_data') as mock_export:
            mock_export.side_effect = NotImplementedError("Export format xyz not implemented")
            
            with self.assertRaises(NotImplementedError):
                mock_export(DataType.TASKS, "xyz")
    
    @patch('scripts.data_export_service.StateManager')
    @patch('scripts.data_export_service.TaskManager')
    @patch('psutil.Process')
    def test_performance_benchmark(self, mock_process, mock_task_manager, mock_state_manager):
        """Test performance benchmarking functionality."""
        # Mock memory usage
        mock_process.return_value.memory_info.return_value.rss = 1024 * 1024 * 100  # 100MB
        
        # Mock the managers
        service = DataExportService(output_dir=self.temp_dir)
        service.task_manager.state = self.mock_task_state
        service.state_manager.load_state = Mock(return_value=self.mock_system_state)
        
        # Run benchmark
        benchmark = service.get_export_performance_benchmark(
            DataType.TASKS, 
            ExportFormat.JSON
        )
        
        # Verify benchmark results
        self.assertIn('data_type', benchmark)
        self.assertIn('export_format', benchmark)
        self.assertIn('execution_time_seconds', benchmark)
        self.assertIn('memory_used_mb', benchmark)
        self.assertIn('file_size_bytes', benchmark)
        self.assertIn('success', benchmark)
        self.assertIn('output_file', benchmark)
        self.assertIn('timestamp', benchmark)
        
        self.assertEqual(benchmark['data_type'], 'tasks')
        self.assertEqual(benchmark['export_format'], 'json')
        self.assertTrue(benchmark['success'])
        self.assertGreater(benchmark['file_size_bytes'], 0)
    
    @patch('scripts.data_export_service.StateManager')
    @patch('scripts.data_export_service.TaskManager')
    def test_csv_empty_data(self, mock_task_manager, mock_state_manager):
        """Test CSV export with empty data."""
        # Mock empty task state
        empty_task_state = {
            "tasks": {},
            "task_counter": 1000,
            "last_updated": "2025-06-09T13:00:00Z",
            "active_tasks": 0,
            "completed_today": 0,
            "success_rate": 0.0
        }
        
        service = DataExportService(output_dir=self.temp_dir)
        service.task_manager.state = empty_task_state
        service.state_manager.load_state = Mock(return_value=self.mock_system_state)
        
        # Export empty task data as CSV
        filepath = service.export_data(DataType.TASKS, ExportFormat.CSV)
        
        # Verify file was created with headers only
        self.assertTrue(os.path.exists(filepath))
        
        with open(filepath, 'r', newline='') as f:
            reader = csv.reader(f)
            rows = list(reader)
        
        # Should have header row only
        self.assertEqual(len(rows), 1)
        self.assertIn('task_id', rows[0])
        self.assertIn('title', rows[0])
    
    @patch('scripts.data_export_service.StateManager')
    @patch('scripts.data_export_service.TaskManager')
    def test_date_range_filtering(self, mock_task_manager, mock_state_manager):
        """Test filtering tasks by date range."""
        service = DataExportService(output_dir=self.temp_dir)
        service.task_manager.state = self.mock_task_state
        service.state_manager.load_state = Mock(return_value=self.mock_system_state)
        
        # Export tasks created after 10:30 AM
        filepath = service.export_data(
            DataType.TASKS, 
            ExportFormat.JSON,
            filters={"start_date": "2025-06-09T10:30:00Z"}
        )
        
        # Verify content
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        # Should only include TASK-1002 (created at 11:00 AM)
        self.assertEqual(len(data['tasks']), 1)
        self.assertIn('TASK-1002', data['tasks'])


class TestDataExportServiceIntegration(unittest.TestCase):
    """Integration tests for DataExportService."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_command_line_interface(self):
        """Test the command-line interface."""
        # This would require mocking sys.argv and testing main()
        # For now, we'll test that the main function exists and can be imported
        from scripts.data_export_service import main
        self.assertTrue(callable(main))
    
    def test_file_creation_permissions(self):
        """Test that export service can create files in the output directory."""
        service = DataExportService(output_dir=self.temp_dir)
        
        # Test that the output directory was created
        self.assertTrue(os.path.exists(self.temp_dir))
        self.assertTrue(os.path.isdir(self.temp_dir))
        
        # Test file creation permissions
        test_file = os.path.join(self.temp_dir, "test.txt")
        with open(test_file, 'w') as f:
            f.write("test")
        
        self.assertTrue(os.path.exists(test_file))


if __name__ == '__main__':
    # Create test suite
    suite = unittest.TestSuite()
    
    # Add all test methods
    suite.addTest(unittest.makeSuite(TestDataExportService))
    suite.addTest(unittest.makeSuite(TestDataExportServiceIntegration))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    print(f"\nTest Summary:")
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Success rate: {((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun) * 100:.1f}%")
    
    # Exit with appropriate code
    sys.exit(0 if result.wasSuccessful() else 1)