#!/usr/bin/env python3
"""
Comprehensive unit tests for ContextGatherer module.

Tests cover:
- Context gathering operations
- AI-enhanced analysis integration
- Market trends research
- Security alerts processing
- Error handling and edge cases
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

from scripts.context_gatherer import ContextGatherer


class TestContextGathererInitialization(unittest.TestCase):
    """Test ContextGatherer initialization."""
    
    def test_init_with_default_parameters(self):
        """Test ContextGatherer initialization with default parameters."""
        # Arrange & Act
        gatherer = ContextGatherer()
        
        # Assert
        self.assertEqual(gatherer.output_path, "context.json")
        self.assertIsNone(gatherer.ai_brain)
    
    def test_init_with_custom_parameters(self):
        """Test ContextGatherer initialization with custom parameters."""
        # Arrange
        output_path = "custom_context.json"
        mock_ai_brain = Mock()
        
        # Act
        gatherer = ContextGatherer(output_path=output_path, ai_brain=mock_ai_brain)
        
        # Assert
        self.assertEqual(gatherer.output_path, output_path)
        self.assertEqual(gatherer.ai_brain, mock_ai_brain)


class TestContextGathererBasicFunctionality(unittest.TestCase):
    """Test basic context gathering functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.output_path = os.path.join(self.temp_dir, "test_context.json")
        self.gatherer = ContextGatherer(output_path=self.output_path)
    
    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir)
    
    def test_gather_context_basic_structure(self):
        """Test that gather_context returns correct basic structure."""
        # Arrange
        charter = {
            "primary_goal": "innovation",
            "secondary_goal": "community_engagement"
        }
        
        # Act
        context = self.gatherer.gather_context(charter)
        
        # Assert
        self.assertIn("timestamp", context)
        self.assertIn("charter_goals", context)
        self.assertIn("market_trends", context)
        self.assertIn("security_alerts", context)
        self.assertIn("technology_updates", context)
        self.assertIn("github_trending", context)
        self.assertIn("programming_news", context)
        
        # Check charter goals are correctly extracted
        self.assertEqual(context["charter_goals"], ["innovation", "community_engagement"])
    
    def test_gather_context_creates_file(self):
        """Test that gather_context saves context to file."""
        # Arrange
        charter = {"primary_goal": "testing"}
        
        # Act
        context = self.gatherer.gather_context(charter)
        
        # Assert
        self.assertTrue(os.path.exists(self.output_path))
        with open(self.output_path, 'r') as f:
            saved_context = json.load(f)
        self.assertEqual(saved_context["charter_goals"], ["testing", ""])
    
    def test_gather_context_with_missing_charter_goals(self):
        """Test gather_context with missing charter goal fields."""
        # Arrange
        charter = {}  # Empty charter
        
        # Act
        context = self.gatherer.gather_context(charter)
        
        # Assert
        self.assertEqual(context["charter_goals"], ["", ""])
    
    def test_gather_context_timestamp_format(self):
        """Test that context timestamp is in correct ISO format."""
        # Arrange
        charter = {"primary_goal": "testing"}
        
        # Act
        context = self.gatherer.gather_context(charter)
        
        # Assert
        # Should be able to parse the timestamp
        timestamp = context["timestamp"]
        parsed_time = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
        self.assertIsInstance(parsed_time, datetime)


class TestContextGathererInnovationTrends(unittest.TestCase):
    """Test innovation trends gathering functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.mock_ai_brain = Mock()
        self.gatherer = ContextGatherer(ai_brain=self.mock_ai_brain)
    
    def test_get_innovation_trends_without_ai_brain(self):
        """Test innovation trends gathering without AI brain."""
        # Arrange
        gatherer = ContextGatherer()  # No AI brain
        
        # Act
        trends = gatherer._get_innovation_trends()
        
        # Assert
        self.assertEqual(trends, [])
    
    def test_get_innovation_trends_with_successful_ai_response(self):
        """Test innovation trends gathering with successful AI response."""
        # Arrange
        expected_trends = [
            {
                'title': 'AI Revolution 2025',
                'snippet': 'Advanced AI capabilities transforming development',
                'url': 'ai-research'
            }
        ]
        
        # Mock AI response
        self.mock_ai_brain.generate_enhanced_response_sync.return_value = {
            'content': str(expected_trends)
        }
        
        # Act
        trends = self.gatherer._get_innovation_trends()
        
        # Assert
        self.assertEqual(len(trends), 1)
        self.assertEqual(trends[0]['title'], 'AI Revolution 2025')
        self.assertEqual(trends[0]['url'], 'ai-research')
        self.mock_ai_brain.generate_enhanced_response_sync.assert_called_once()
    
    def test_get_innovation_trends_with_ai_parsing_error(self):
        """Test innovation trends gathering when AI response parsing fails."""
        # Arrange
        # Mock AI response with unparseable content
        self.mock_ai_brain.generate_enhanced_response_sync.return_value = {
            'content': 'This is not valid Python literal'
        }
        
        # Act
        trends = self.gatherer._get_innovation_trends()
        
        # Assert
        self.assertEqual(len(trends), 1)
        self.assertEqual(trends[0]['title'], 'AI Technology Trends 2025')
        self.assertEqual(trends[0]['url'], 'ai-research')
        self.assertEqual(trends[0]['snippet'][:20], 'This is not valid Py')
    
    def test_get_innovation_trends_with_ai_exception(self):
        """Test innovation trends gathering when AI raises exception."""
        # Arrange
        self.mock_ai_brain.generate_enhanced_response_sync.side_effect = Exception("AI error")
        
        # Act
        trends = self.gatherer._get_innovation_trends()
        
        # Assert
        self.assertEqual(trends, [])
    
    def test_get_innovation_trends_with_empty_ai_response(self):
        """Test innovation trends gathering with empty AI response."""
        # Arrange
        self.mock_ai_brain.generate_enhanced_response_sync.return_value = None
        
        # Act
        trends = self.gatherer._get_innovation_trends()
        
        # Assert
        self.assertEqual(trends, [])


class TestContextGathererTechnologyUpdates(unittest.TestCase):
    """Test technology updates gathering functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.mock_ai_brain = Mock()
        self.gatherer = ContextGatherer(ai_brain=self.mock_ai_brain)
    
    def test_get_technology_updates_without_ai_brain(self):
        """Test technology updates gathering without AI brain."""
        # Arrange
        gatherer = ContextGatherer()  # No AI brain
        
        # Act
        updates = gatherer._get_technology_updates()
        
        # Assert
        self.assertEqual(updates, [])
    
    def test_get_technology_updates_with_successful_ai_response(self):
        """Test technology updates gathering with successful AI response."""
        # Arrange
        expected_updates = [
            {
                'title': 'Python 3.13 Released',
                'snippet': 'New features and performance improvements',
                'url': 'ai-research'
            }
        ]
        
        # Mock AI response
        self.mock_ai_brain.generate_enhanced_response_sync.return_value = {
            'content': str(expected_updates)
        }
        
        # Act
        updates = gatherer._get_technology_updates()
        
        # Assert
        self.assertEqual(len(updates), 1)
        self.assertEqual(updates[0]['title'], 'Python 3.13 Released')
        self.mock_ai_brain.generate_enhanced_response_sync.assert_called_once()
    
    def test_get_technology_updates_with_ai_exception(self):
        """Test technology updates gathering when AI raises exception."""
        # Arrange
        self.mock_ai_brain.generate_enhanced_response_sync.side_effect = Exception("AI error")
        
        # Act
        updates = self.gatherer._get_technology_updates()
        
        # Assert
        self.assertEqual(updates, [])


class TestContextGathererSecurityAlerts(unittest.TestCase):
    """Test security alerts gathering functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.mock_ai_brain = Mock()
        self.gatherer = ContextGatherer(ai_brain=self.mock_ai_brain)
    
    def test_get_security_alerts_without_ai_brain(self):
        """Test security alerts gathering without AI brain."""
        # Arrange
        gatherer = ContextGatherer()  # No AI brain
        
        # Act
        alerts = gatherer._get_security_alerts()
        
        # Assert
        self.assertEqual(alerts, [])
    
    def test_get_security_alerts_with_successful_ai_response(self):
        """Test security alerts gathering with successful AI response."""
        # Arrange
        expected_alerts = [
            {
                'title': 'CVE-2025-0001',
                'snippet': 'Critical security vulnerability in popular library',
                'url': 'ai-research'
            }
        ]
        
        # Mock AI response
        self.mock_ai_brain.generate_enhanced_response_sync.return_value = {
            'content': str(expected_alerts)
        }
        
        # Act
        alerts = self.gatherer._get_security_alerts()
        
        # Assert
        self.assertEqual(len(alerts), 1)
        self.assertEqual(alerts[0]['title'], 'CVE-2025-0001')
        self.mock_ai_brain.generate_enhanced_response_sync.assert_called_once()


class TestContextGathererProgrammingNews(unittest.TestCase):
    """Test programming news gathering functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.mock_ai_brain = Mock()
        self.gatherer = ContextGatherer(ai_brain=self.mock_ai_brain)
    
    def test_get_programming_news_without_ai_brain(self):
        """Test programming news gathering without AI brain."""
        # Arrange
        gatherer = ContextGatherer()  # No AI brain
        
        # Act
        news = gatherer._get_programming_news()
        
        # Assert
        self.assertEqual(news, [])
    
    def test_get_programming_news_with_successful_ai_response(self):
        """Test programming news gathering with successful AI response."""
        # Arrange
        expected_news = [
            {
                'title': 'New Framework Released',
                'snippet': 'Revolutionary new development framework',
                'url': 'ai-research'
            }
        ]
        
        # Mock AI response
        self.mock_ai_brain.generate_enhanced_response_sync.return_value = {
            'content': str(expected_news)
        }
        
        # Act
        news = self.gatherer._get_programming_news()
        
        # Assert
        self.assertEqual(len(news), 1)
        self.assertEqual(news[0]['title'], 'New Framework Released')
        self.mock_ai_brain.generate_enhanced_response_sync.assert_called_once()


class TestContextGathererGithubTrending(unittest.TestCase):
    """Test GitHub trending functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.mock_ai_brain = Mock()
        self.gatherer = ContextGatherer(ai_brain=self.mock_ai_brain)
    
    def test_get_github_trending_without_ai_brain(self):
        """Test GitHub trending gathering without AI brain."""
        # Arrange
        gatherer = ContextGatherer()  # No AI brain
        
        # Act
        trending = gatherer._get_github_trending()
        
        # Assert
        self.assertEqual(trending, [])
    
    def test_get_github_trending_with_successful_ai_response(self):
        """Test GitHub trending gathering with successful AI response."""
        # Arrange
        expected_trending = [
            {
                'title': 'awesome-ai-tools',
                'snippet': 'Curated list of AI development tools',
                'url': 'ai-research'
            }
        ]
        
        # Mock AI response
        self.mock_ai_brain.generate_enhanced_response_sync.return_value = {
            'content': str(expected_trending)
        }
        
        # Act
        trending = self.gatherer._get_github_trending()
        
        # Assert
        self.assertEqual(len(trending), 1)
        self.assertEqual(trending[0]['title'], 'awesome-ai-tools')
        self.mock_ai_brain.generate_enhanced_response_sync.assert_called_once()


class TestContextGathererGeneralTrends(unittest.TestCase):
    """Test general programming trends functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.gatherer = ContextGatherer()
    
    def test_get_general_programming_trends_returns_list(self):
        """Test that general programming trends returns a list."""
        # Act
        trends = self.gatherer._get_general_programming_trends()
        
        # Assert
        self.assertIsInstance(trends, list)


class TestContextGathererAIEnhancement(unittest.TestCase):
    """Test AI enhancement functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.mock_ai_brain = Mock()
        self.gatherer = ContextGatherer(ai_brain=self.mock_ai_brain)
    
    def test_enhance_context_with_ai_success(self):
        """Test successful AI context enhancement."""
        # Arrange
        original_context = {
            "market_trends": [{"title": "trend1"}],
            "security_alerts": []
        }
        
        enhanced_response = {
            'content': {
                'insights': ['AI-generated insight'],
                'priority_trends': ['High priority trend'],
                'risk_assessment': 'low'
            }
        }
        
        self.mock_ai_brain.generate_enhanced_response_sync.return_value = enhanced_response
        
        # Act
        enhanced_context = self.gatherer._enhance_context_with_ai(original_context)
        
        # Assert
        self.assertIn('ai_insights', enhanced_context)
        self.mock_ai_brain.generate_enhanced_response_sync.assert_called_once()
    
    def test_enhance_context_with_ai_exception(self):
        """Test AI enhancement when AI raises exception."""
        # Arrange
        original_context = {"market_trends": []}
        self.mock_ai_brain.generate_enhanced_response_sync.side_effect = Exception("AI error")
        
        # Act
        enhanced_context = self.gatherer._enhance_context_with_ai(original_context)
        
        # Assert
        # Should return original context when AI fails
        self.assertEqual(enhanced_context, original_context)


class TestContextGathererCharterBasedGathering(unittest.TestCase):
    """Test charter-based context gathering."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.mock_ai_brain = Mock()
        self.gatherer = ContextGatherer(ai_brain=self.mock_ai_brain)
    
    def test_gather_context_innovation_charter(self):
        """Test context gathering with innovation charter."""
        # Arrange
        charter = {"primary_goal": "innovation"}
        
        # Mock AI responses
        self.mock_ai_brain.generate_enhanced_response_sync.return_value = {
            'content': '[]'
        }
        
        # Act
        context = self.gatherer.gather_context(charter)
        
        # Assert
        self.assertIn("market_trends", context)
        self.assertIn("technology_updates", context)
        self.assertIn("github_trending", context)
    
    def test_gather_context_security_charter(self):
        """Test context gathering with security charter."""
        # Arrange
        charter = {"primary_goal": "security"}
        
        # Mock AI responses
        self.mock_ai_brain.generate_enhanced_response_sync.return_value = {
            'content': '[]'
        }
        
        # Act
        context = self.gatherer.gather_context(charter)
        
        # Assert
        self.assertIn("security_alerts", context)
    
    def test_gather_context_community_engagement_charter(self):
        """Test context gathering with community engagement charter."""
        # Arrange
        charter = {"primary_goal": "community_engagement"}
        
        # Mock AI responses
        self.mock_ai_brain.generate_enhanced_response_sync.return_value = {
            'content': '[]'
        }
        
        # Act
        context = self.gatherer.gather_context(charter)
        
        # Assert
        self.assertIn("programming_news", context)
        self.assertIn("github_trending", context)
    
    def test_gather_context_handles_exceptions_gracefully(self):
        """Test that gather_context handles exceptions gracefully."""
        # Arrange
        charter = {"primary_goal": "innovation"}
        
        # Mock AI brain to raise exception
        self.mock_ai_brain.generate_enhanced_response_sync.side_effect = Exception("Network error")
        
        # Act
        context = self.gatherer.gather_context(charter)
        
        # Assert
        self.assertIn("error", context)
        self.assertEqual(context["error"], "Network error")
        # Should still have basic structure
        self.assertIn("timestamp", context)
        self.assertIn("charter_goals", context)


class TestContextGathererFileOperations(unittest.TestCase):
    """Test file operations for context saving."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.output_path = os.path.join(self.temp_dir, "test_context.json")
        self.gatherer = ContextGatherer(output_path=self.output_path)
    
    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir)
    
    def test_save_context_creates_file(self):
        """Test that _save_context creates file with correct content."""
        # Arrange
        test_context = {
            "timestamp": "2025-01-01T00:00:00+00:00",
            "market_trends": []
        }
        
        # Act
        self.gatherer._save_context(test_context)
        
        # Assert
        self.assertTrue(os.path.exists(self.output_path))
        with open(self.output_path, 'r') as f:
            saved_context = json.load(f)
        self.assertEqual(saved_context["timestamp"], "2025-01-01T00:00:00+00:00")
    
    def test_save_context_handles_file_error(self):
        """Test that _save_context handles file write errors gracefully."""
        # Arrange
        # Use invalid path to cause write error
        self.gatherer.output_path = "/invalid/path/context.json"
        test_context = {"test": "data"}
        
        # Act & Assert - should not raise exception
        try:
            self.gatherer._save_context(test_context)
        except Exception:
            self.fail("_save_context should handle file errors gracefully")


if __name__ == '__main__':
    # Run tests with verbose output
    unittest.main(verbosity=2)